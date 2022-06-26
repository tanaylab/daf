"""
Write API for ``daf`` data sets.

.. note::

    Since ``daf`` aggressively caches data derived from the stored data, the only safe operation is to add new data to a
    data set (as this invalidates no caches). Even then one has to take care that no old `.StorageView` objects are used
    as these need not reflect the new data. Actually overwriting data is even worse as it will **not** invalidate any
    stale data in the caches. Deletion isn't even supported in the API.

    When overwriting or deletion is needed, the recommended idiom is to create a new `.DafWriter` with a separate
    ``storage`` to hold the data-we'll-want-to-overwrite-or-delete. After we are done with this stale data, we can
    simply discard this ``storage`` and create a fresh `.DafWriter` without it. This requires foresight, but is
    automated by functions such as `.DafWriter.adapter` and `.computation`.

.. todo::

    Track which views/caches refer to each base storage and automatically invalidate any cached data on change, and
    provide delete operations? This would massively complicate the implementation...
"""

# pylint: disable=duplicate-code

import sys
from contextlib import contextmanager
from inspect import Parameter
from inspect import signature
from textwrap import dedent
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generator
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Set
from typing import TypeVar
from typing import Union

import numpy as np
import scipy.sparse as sp  # type: ignore

from ..storage import AxisView
from ..storage import MemoryStorage
from ..storage import StorageChain
from ..storage import StorageReader
from ..storage import StorageView
from ..storage import StorageWriter
from ..storage import prefix
from ..storage import suffix
from ..typing import STR_DTYPE
from ..typing import AnyData
from ..typing import DenseInRows
from ..typing import DType
from ..typing import Vector
from ..typing import as_dense
from ..typing import as_matrix
from ..typing import as_vector
from ..typing import assert_data
from ..typing import be_dense_in_rows
from ..typing import be_matrix_in_rows
from ..typing import be_vector
from ..typing import dtype_of
from ..typing import freeze
from ..typing import has_dtype
from ..typing import is_dense_in_rows
from ..typing import is_matrix_in_rows
from ..typing import is_sparse_in_rows
from ..typing import optimize
from .readers import DafReader
from .readers import transpose_name

# pylint: enable=duplicate-code

__all__ = [
    "DafWriter",
    "BackAxis",
    "BackData",
    "COMPLETE_DATA",
    "computation",
]


class CompleteData:  # pylint: disable=too-few-public-methods
    """
    Specify that computed data must be complete (will not use axes that were sliced).
    """


#: Specify that computed data must be complete (will not use axes that were sliced).
COMPLETE_DATA = CompleteData()


class BackAxis(NamedTuple):
    """
    How to copy data axis from processing results into the original data set.
    """

    #: The simple name to copy the axis into in the original data set. By default the axis is not renamed.
    name: Optional[str] = None

    #: Whether the axis is not required to exist in the computed results.
    optional: bool = False

    #: Whether to (try to) copy the axis into the original data set even if the processing code failed with some
    #: exception.
    copy_on_error: bool = False


class BackData(NamedTuple):
    """
    How to copy data back from processing results into the original data set.
    """

    #: The simple name to copy the data into in the original data set. By default the data is not renamed.
    name: Optional[str] = None

    #: Whether to overwrite existing data in the original data set.
    default: Any = COMPLETE_DATA

    #: Whether the data is not required to exist in the computed results.
    optional: bool = False

    #: Whether to overwrite existing data in the original data set.
    overwrite: bool = False

    #: Whether to (try to) copy the data into the original data set even if the processing code failed with some
    #: exception.
    copy_on_error: bool = False


class DafWriter(DafReader):
    """
    Read-write access to a ``daf`` data set.
    """

    def __init__(
        self,
        storage: StorageWriter,
        *,
        base: Optional[StorageReader] = None,
        derived: Optional[StorageWriter] = None,
        name: str = ".daf#",
    ) -> None:
        """
        If the ``name`` starts with ``.``, it is appended to the ``base`` name. If the name ends with ``#``, we append
        the object id to it to make it unique.
        """
        super().__init__(base or storage, derived=derived, name=name)

        #: Where to store modifications to the data set. By default the ``base`` is also set to this. Specifying an
        #: explicit ``base`` allows, for example, to use a `.MemoryStorage` to hold modifications (such as additional
        #: type annotations), without actually modifying some read-only base `.FilesReader` cells atlas.
        self.storage = storage

        self.chain = StorageChain([self.derived, self.storage, self.base], name=self.name + ".chain")

        if self.storage is self.base:
            return

        for axis in self.chain.axis_names():
            entries: Optional[Vector] = None
            if not self.derived.has_axis(axis):
                entries = freeze(optimize(as_vector(self.chain.axis_entries(axis))))
                self.derived.create_axis(axis, entries)
            if not self.storage.has_axis(axis):
                entries = entries or freeze(optimize(as_vector(self.chain.axis_entries(axis))))
                self.storage.create_axis(axis, entries)

    def as_reader(self) -> "DafReader":
        """
        Return the data set as a `.DafReader`.

        This is a no-op (returns self) for "real" read-only data sets, but for writable data sets, it returns a "real"
        read-only wrapper object (that does not implement the writing methods). This ensures that the result can't be
        used to modify the data if passed by mistake to a function that takes a `.DafWriter`.
        """
        return DafReader(StorageChain([self.storage, self.base]), derived=self.derived, name=self.name + ".as_reader")

    def _view_base(
        self, axes: Optional[Mapping[str, Union[None, str, AnyData, AxisView]]], hide_implicit: bool, name: str
    ) -> StorageReader:
        return StorageChain(
            [self._derived_filtered(axes, hide_implicit, name), self.storage, self.base], name=name + ".chain"
        )

    def set_item(self, name: str, item: Any, *, overwrite: bool = False) -> None:
        """
        Set a ``name`` 0D data ``item``.

        If ``overwrite``, will silently overwrite an existing item of the same name, otherwise overwriting will fail.
        """
        assert overwrite or not self.has_item(
            name
        ), f"refuse to overwrite the item: {name} in the data set: {self.name}"
        self.storage.set_item(name, item, overwrite=True)

    def create_axis(self, axis: str, entries: AnyData) -> None:
        """
        Create a new ``axis`` and the unique ``entries`` identifying each entry along the axis.

        The ``entries`` must be `.is_frozen` and contain string data.

        It is always an error to overwrite an existing axis.

        .. note::

            We verify that the axis entries are unique. However, we can't guarantee that the entries will be unique for
            axes in arbitrary data accessed through some storage adapter (e.g., ``AnnData``).
        """
        assert not self.has_axis(axis), f"refuse to recreate the axis: {axis} in the data set: {self.name}"
        entries = freeze(optimize(as_vector(entries)))

        unique, counts = np.unique(entries, return_counts=True)
        repeated = np.argmax(counts)
        assert (
            counts[repeated] == 1
        ), f"duplicate entry: {unique[repeated]} in the entries for the axis: {axis} in the data set: {self.name}"

        self.storage.create_axis(axis, entries)
        self.derived.create_axis(axis, entries)

    def set_data1d(self, name: str, data1d: AnyData, *, overwrite: bool = False) -> None:
        """
        Set a ``name`` `.AnyData` data.

        The name must be in the format ``axis;name`` which uniquely identifies the 1D data.

        If ``overwrite``, will silently overwrite an existing 1D data of the same name, otherwise overwriting will fail.
        """
        assert overwrite or not self.has_data1d(
            name
        ), f"refuse to overwrite the 1D data: {name} in the data set: {self.name}"
        self.storage.set_vector(name, freeze(optimize(as_vector(data1d))), overwrite=True)

    def set_data2d(self, name: str, data2d: AnyData, *, overwrite: bool = False) -> None:
        """
        Set a ``name`` ``.AnyData`` data.

        The name must be in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data.

        If ``overwrite``, will silently overwrite an existing 2D data of the same name, otherwise overwriting will fail.
        """
        assert overwrite or not self.has_data2d(
            name
        ), f"refuse to overwrite the 2D data: {name} in the data set: {self.name}"

        matrix = as_matrix(data2d)
        if is_matrix_in_rows(matrix):
            matrix_in_rows = matrix
        else:
            matrix_in_rows = be_matrix_in_rows(matrix.transpose())
            name = transpose_name(name)

        self.storage.set_matrix(name, freeze(optimize(matrix_in_rows)), overwrite=True)

    @contextmanager
    def create_dense_in_rows(
        self, name: str, *, dtype: DType, overwrite: bool = False
    ) -> Generator[DenseInRows, None, None]:
        """
        Create an uninitialized `.ROW_MAJOR` .`DenseInRows` of some ``dtype`` to be set by the ``name`` in the data set,
        expecting the code to initialize it.

        The name must be in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data.

        Expected usage is:

        .. code:: python

            with data.create_dense_in_rows(name="rows_axis,columns_axis;name", dtype="...") as dense:
                # Here the dense is still not necessarily set inside the data set.
                # That is, one can't assume ``get_matrix`` will access it.
                # It is only available for filling in the values:
                dense[..., ...] = ...

            # Here the array is set inside the storage.
            # That is, one can use ``get_matrix`` to access it.

        This allows `.FilesWriter` to create the array on disk without first having to create an in-memory copy. By
        default (for other storage adapters), this just creates and returns an uninitialized in-memory 2D dense array,
        then sets it as the 2D data value.
        """
        with self.storage.create_dense_in_rows(name, dtype=dtype, overwrite=overwrite) as dense:
            yield dense

    @contextmanager
    def adapter(  # pylint: disable=too-many-locals
        self,
        *,
        axes: Optional[Mapping[str, Union[None, str, AnyData, AxisView]]] = None,
        data: Optional[Mapping[str, Optional[str]]] = None,
        name: str = ".adapter#",
        cache: Optional[StorageWriter] = None,
        storage: Optional[StorageWriter] = None,
        hide_implicit: bool = False,
        back_axes: Union[None, Collection[str], Mapping[str, Union[str, BackAxis]]] = None,
        back_data: Union[None, Collection[str], Mapping[str, Union[str, BackData]]] = None,
    ) -> Generator["DafWriter", None, None]:
        """
        Execute some code on a view of this data set; when done, transfer (some of) the results back into it.

        If the ``name`` starts with ``.``, it is appended to both the `.StorageView` and the `.DafWriter` names. If the
        name ends with ``#``, we append the object id to it to make it unique.

        This sets up a `.StorageView` to adapt the data set to the expectation of some processing code. It then uses
        this as the ``base`` for a `.DafWriter` which is provided to the processing code. By default this uses
        `.MemoryStorage` as the ``storage`` for the computed results. When the processing completes, (some of) the
        computed results are copied back into the original data set:

        * If ``back_axes`` is specified, it should list (some of) the new axes created by the processing code. Each of
          these will be copied into the original data set. If ``back_axes`` is a ``dict``, it provides either a name or
          a complete `.BackAxis` specifying exactly how to copy each axis back. Otherwise it is just a collection of the
          new axes to copy on success, preserving their name.

        * If ``back_data`` is specified, it should list (some of) the new data created (or modified) by the processing
          code. Each of these will be copied back into the original data set. If ``back_data`` is a ``dict``, it
          provides either a name of a complete `.BackData` specifying exactly how to copy each data back. Otherwise it
          is a just a collection of the data to copy on success, preserving the names and requiring that such data will
          not use any sliced axes.

        A contrived example might look like:

        .. code:: python

            rna = DafReader(...)

            # Assume the `rna` data set has `cell` and `gene` axes, and a per-cell-per-gene `UMIs` matrix.

            with rna.adapter(axes=dict(cell="x", gene="y", data={ "cell,gene:UMIs": "z" }), hide_implicit=True,
                             back_data=[ "x;mean", "y;variance" ]) as adapter:

                # The `adapter` data set has only `x` and `y` axes, and a per-x-per-y `z` matrix,
                # matching the expectations of `collect_stats`:

                collect_stats(adapter)

                # Assume `collect_stats` created `x;mean`, `x;variance`, `y;mean`, `y;variance` in `adapter`.
                # This has no effect on the `rna` data set (yet).

            # The `rna` data set now has additional `cell;mean` and `gene;variance` data copied from the above.
            # It does not contain `cell;variance` and `gene;mean`, as these were not requested to be copied.

        .. note::

            This idiom is key for creating an ecosystem of generic ``daf`` processing tools. Such tools can require and
            compute generic axes and data names, and still be applied to data that uses specific descriptive axes and
            data names. Often the same generic tool can be applied to the same data set in multiple ways, using
            different mappings between the specific names and the generic names.

        .. todo::

            Provide a more efficient implementation of ``DafWriter._copy_back_data2d`` (used by `.DafWriter.adapter`).
            The current implementation uses a few temporary buffers the size of the partial data. If this were
            implemented in a C/C++ extension it would avoid the temporary buffers, giving a significant performance
            boost for large data sizes. So far we have chosen to keep ``daf`` as a pure Python package so we suffer this
            inefficiency. Perhaps using ``numba`` would provide the efficiency while avoiding C/C++ extension code? Of
            course this really should be a part of ``numpy`` and/or ``scipy.sparse`` in the 1st place.
        """
        # pylint: disable=duplicate-code

        if name.startswith("."):
            name = self.name + name

        unique: Optional[List[None]] = None
        if name.endswith("#"):
            unique = []
            name = name + str(id(unique))

        # pylint: enable=duplicate-code

        view = StorageView(
            self.chain, axes=axes, data=data, cache=cache, hide_implicit=hide_implicit, name=name + ".base"
        )
        adapter = DafWriter(storage or MemoryStorage(name=name + ".storage"), base=view, name=name)

        if unique is not None:
            setattr(view, "__daf_unique__", unique)  # Prevent it from being garbage collected.

        _back_axes: Mapping[str, BackAxis]
        if back_axes is None:
            _back_axes = {}
        elif isinstance(back_axes, dict):
            _back_axes = {
                name: BackAxis(name=back) if isinstance(back, str) else back for name, back in back_axes.items()
            }
        else:
            _back_axes = {axis: BackAxis() for axis in back_axes}

        _back_items: Mapping[str, BackData]
        _back_data1d: Mapping[str, BackData]
        _back_data2d: Mapping[str, BackData]
        if back_data is None:
            _back_items = {}
            _back_data1d = {}
            _back_data2d = {}
        elif isinstance(back_data, dict):
            _back_items = {
                name: BackData(name=back) if isinstance(back, str) else back
                for name, back in back_data.items()
                if ";" not in name
            }
            _back_data1d = {
                name: BackData(name=back) if isinstance(back, str) else back
                for name, back in back_data.items()
                if ";" in name and "," not in prefix(name, ";")
            }
            _back_data2d = {
                name: BackData(name=back) if isinstance(back, str) else back
                for name, back in back_data.items()
                if ";" in name and "," in prefix(name, ";")
            }
        else:
            _back_items = {name: BackData() for name in back_data if ";" not in name}
            _back_data1d = {name: BackData() for name in back_data if ";" in name and "," not in prefix(name, ";")}
            _back_data2d = {name: BackData() for name in back_data if ";" in name and "," in prefix(name, ";")}

        try:
            yield adapter
        except:
            self._copy_back(view, adapter, _back_axes, _back_items, _back_data1d, _back_data2d, is_error=True)
            raise

        self._copy_back(view, adapter, _back_axes, _back_items, _back_data1d, _back_data2d, is_error=False)

    def _copy_back(  # pylint: disable=too-many-arguments
        self,
        view: StorageView,
        adapter: DafReader,
        back_axes: Mapping[str, BackAxis],
        back_data: Mapping[str, BackData],
        back_data1d: Mapping[str, BackData],
        back_data2d: Mapping[str, BackData],
        *,
        is_error: bool,
    ) -> None:
        self._copy_back_axes(adapter, back_axes, is_error=is_error)
        self._copy_back_items(view, adapter, back_data, is_error=is_error)
        self._copy_back_data1d(view, adapter, back_axes, back_data1d, is_error=is_error)
        self._copy_back_data2d(view, adapter, back_axes, back_data2d, is_error=is_error)

    def _copy_back_axes(self, adapter: DafReader, back_axes: Mapping[str, BackAxis], *, is_error: bool) -> None:
        for axis, back in back_axes.items():
            if (back.copy_on_error or not is_error) and (not back.optional or adapter.has_axis(axis)):
                self.create_axis(back.name or axis, adapter.axis_entries(axis))

    def _copy_back_items(
        self, view: StorageView, adapter: DafReader, back_data: Mapping[str, BackData], *, is_error: bool
    ) -> None:
        for item, back in back_data.items():
            if (not back.copy_on_error and is_error) or (back.optional and not adapter.has_item(item)):
                continue

            if back.name is not None:
                back_name = back.name
            elif view.has_item(item):
                back_name = view.base_item(item)
            else:
                back_name = item

            self.set_item(back_name, adapter.get_item(item), overwrite=back.overwrite)

    def _copy_back_data1d(
        self,
        view: StorageView,
        adapter: DafReader,
        back_axes: Mapping[str, BackAxis],
        back_data1d: Mapping[str, BackData],
        *,
        is_error: bool,
    ) -> None:
        for data1d, back in back_data1d.items():
            if (not back.copy_on_error and is_error) or (back.optional and not adapter.has_data1d(data1d)):
                continue

            axis = prefix(data1d, ";")
            back_axis = _back_axis_name(view, axis, back_axes)

            partial = adapter.get_vector(data1d)
            slice_indices = view.axis_slice_indices(axis)
            if slice_indices is None:
                complete = partial
            else:
                assert not isinstance(back.default, CompleteData), (
                    f"missing a default value for completing the partial 1D data: {data1d} "
                    f"from the computed data set: {adapter.name} "
                    f"back to the original data set: {self.name}"
                )
                complete = be_vector(np.full(self.axis_size(back_axis), back.default, dtype=partial.dtype))
                complete[slice_indices] = partial

            if back.name is not None:
                back_name = back.name
            elif view.has_data1d(data1d):
                back_name = suffix(view.base_data1d(data1d), ";")
            else:
                back_name = suffix(data1d, ";")

            self.set_data1d(f"{back_axis};{back_name}", complete, overwrite=back.overwrite)

    def _copy_back_data2d(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        view: StorageView,
        adapter: DafReader,
        back_axes: Mapping[str, BackAxis],
        back_data2d: Mapping[str, BackData],
        *,
        is_error: bool,
    ) -> None:
        for data2d, back in back_data2d.items():
            if (not back.copy_on_error and is_error) or (back.optional and not adapter.has_data2d(data2d)):
                continue

            (view_rows_axis, view_columns_axis) = prefix(data2d, ";").split(",")
            back_rows_axis_name = _back_axis_name(view, view_rows_axis, back_axes)
            back_columns_axis_name = _back_axis_name(view, view_columns_axis, back_axes)

            if back.name is not None:
                back_name = back.name
            elif view.has_data2d(data2d):
                back_name = suffix(view.base_data2d(data2d), ";")
            else:
                back_name = suffix(data2d, ";")

            complete_name = f"{back_rows_axis_name},{back_columns_axis_name};{back_name}"

            partial = adapter.get_matrix(data2d)
            partial_row_indices = view.axis_slice_indices(view_rows_axis) if view.has_axis(view_rows_axis) else None
            partial_column_indices = (
                view.axis_slice_indices(view_columns_axis) if view.has_axis(view_columns_axis) else None
            )

            if partial_row_indices is None and partial_column_indices is None:
                self.set_data2d(complete_name, partial, overwrite=back.overwrite)
                continue

            assert not isinstance(back.default, CompleteData), (
                f"missing a default value for completing the partial 2D data: {data2d} "
                f"from the computed data set: {adapter.name} "
                f"back to the original data set: {self.name}"
            )

            complete_rows_count = self.axis_size(back_rows_axis_name)
            complete_columns_count = self.axis_size(back_columns_axis_name)
            complete_shape = (complete_rows_count, complete_columns_count)

            if partial_row_indices is None:
                partial_row_indices = np.arange(complete_rows_count)
            if partial_column_indices is None:
                partial_column_indices = np.arange(complete_columns_count)

            if is_sparse_in_rows(partial):
                if back.default == 0:
                    complete_column_indices = partial_column_indices[partial.indices]
                    complete_indptr = np.zeros_like(partial.indptr, shape=complete_rows_count + 1)
                    complete_indptr[1:][partial_row_indices] = np.diff(partial.indptr)
                    np.cumsum(complete_indptr, out=complete_indptr)
                    complete = sp.csr_matrix(
                        (partial.data, complete_column_indices, complete_indptr),
                        shape=complete_shape,
                    )
                    self.set_data2d(complete_name, complete, overwrite=back.overwrite)
                    continue

                partial = as_dense(partial)

            assert_data(is_dense_in_rows(partial), "row-major matrix", partial)

            partial = be_dense_in_rows(as_dense(partial))
            partial_flat_data = partial.reshape(partial.size)
            partial_flat_indices = (
                (partial_row_indices * complete_columns_count)[:, np.newaxis] + partial_column_indices[np.newaxis, :]
            ).reshape(partial.size)

            if has_dtype(partial, STR_DTYPE):
                complete = np.full(partial.shape, back.default, dtype=STR_DTYPE)
                complete_flat_data = complete.reshape(complete.size)
                complete_flat_data[partial_flat_indices] = partial_flat_data
                self.set_data2d(complete_name, complete, overwrite=back.overwrite)

            else:
                dtype = dtype_of(partial)
                assert dtype is not None
                with self.create_dense_in_rows(complete_name, dtype=dtype, overwrite=back.overwrite) as complete:
                    complete_flat_data = complete.reshape(complete.size)
                    complete_flat_data[:] = back.default
                    complete_flat_data[partial_flat_indices] = partial_flat_data

    @contextmanager
    def computation(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        self,
        required_inputs: Optional[Collection[str]] = None,
        optional_inputs: Optional[Collection[str]] = None,
        assured_outputs: Optional[Collection[str]] = None,
        optional_outputs: Optional[Collection[str]] = None,
        overwrite: bool = False,
        name: str = ".computation#",
        storage: Optional[StorageWriter] = None,
        derived: Optional[StorageWriter] = None,
    ) -> Generator["DafWriter", None, None]:
        """
        Implement some computation on a ``daf`` data set, with explicit input and output data names.

        If the ``name`` starts with ``.``, it is appended to both the `.StorageView` and the `.DafWriter` names. If the
        name ends with ``#``, we append the object id to it to make it unique.

        .. note::

            This is expected to be used by "well behaved" computation tools for ``daf`` data. Typically this is called
            implicitly by `.computation`. In theory, you only need to invoke this manually if the list of inputs and
            outputs depends on the parameters. The description here is still useful for better understanding of the
            behavior of `.computation`.

        This restricts the data available for the computation to just the listed ``required_inputs`` and, if they exist,
        ``optional_inputs``. Once the computation is complete, it ensures the ``required_outputs`` exist and ensures
        that only the ``required_outputs`` and ``optional_outputs`` are copied into the data set. If ``overwrite``, this
        will overwrite any existing data.

        During the computation, any intermediate and derived results are placed in the ``storage`` and ``derived``,
        which by default are simple `.MemoryStorage` objects. Only explicitly listed results, or derived data based on
        pre-existing (or explicitly listed result) data, are copied back into the data set.

        The expected (manual) usage is:

        .. code:: python

            def my_computation(data: daf.DafWriter, ...) -> ...:
                '''
                Describe the computation...

                Required Inputs: ...

                Optional Inputs: ...

                Assured Outputs: ...

                Optional Outputs: ...
                '''

                # Here the `data` may contain more than we think and documented that we need.
                # If the code accesses unlisted data, it would still work when it should fail.
                # If the code fails to create some "assured" data, it is the caller that will fail.
                # Likewise, the code can easily leaks temporary results into the `data`.

                my_computation_implementation(data, ...)  # Unsafe: do not do this!

                # Instead do this:

                with daf.computation(data, name=".my_computation",
                                     required_inputs="...", optional_inputs="...",
                                     assured_outputs="...", optional_outputs="...") as work:

                    # Here the `work` data set contains just the inputs,
                    # so accidently accessing other data will fail.
                    # Also, here we can freely write temporary results to the `work` data set,
                    # without it leaking back to the original `data` set.

                    return my_computation_implementation(work, ...)  # Safe!

                # Here the `data` set is updated with only the outputs.
                # Required outputs are guaranteed to exist here.

        .. todo::

            If both the final and temporary storage are `.FilesWriter`, avoid copying large 2D data files and instead
            directly move them from one directory to another.
        """
        required_inputs = required_inputs or {}
        optional_inputs = optional_inputs or {}
        assured_outputs = assured_outputs or {}
        optional_outputs = optional_outputs or {}

        raw_name = name
        if name.startswith("."):
            name = self.name + name

        self.verify_has(required_inputs, reason=f"required for: {raw_name}")

        view_axis: Dict[str, str] = {}
        for axis in _all_axes(required_inputs):
            view_axis[axis] = axis
        for optional_names in (optional_inputs, assured_outputs, optional_outputs):
            for axis in _all_axes(optional_names):
                if self.has_axis(axis):
                    view_axis[axis] = axis

        view_data: Dict[str, str] = {}
        for required_name in required_inputs:
            view_data[required_name] = suffix(required_name, ";") if ";" in required_name else required_name
        for optional_name in optional_inputs:
            if self.has_data(optional_name):
                view_data[optional_name] = suffix(optional_name, ";") if ";" in optional_name else optional_name

        unique: Optional[List[None]] = None
        if name.endswith("#"):
            unique = []
            name = name + str(id(unique))

        work = DafWriter(
            storage or MemoryStorage(name=name + ".storage"),
            base=StorageView(self.chain, hide_implicit=True, axes=view_axis, data=view_data, name=name + ".base"),
            derived=derived or MemoryStorage(name=name + ".derived"),
            name=name,
        )
        if unique is not None:
            setattr(work, "__daf_unique__", unique)  # Prevent it from being garbage collected.

        yield work

        work.verify_has(assured_outputs, reason=f"assured by: {raw_name}")

        self._copy_axes(work, assured_outputs, optional_outputs)

        for assured_name in assured_outputs:
            self._copy_data(work, assured_name, overwrite)

        for optional_name in optional_outputs:
            if work.has_data(optional_name):
                self._copy_data(work, optional_name, overwrite)

        for rows_axis in self.axis_names():
            if not work.derived.has_axis(rows_axis):
                continue
            for columns_axis in self.axis_names():
                if not work.derived.has_axis(columns_axis):
                    continue
                for derived_name in work.derived.data2d_names((rows_axis, columns_axis)):
                    base_name = prefix(derived_name, "|")
                    if self.has_data2d(base_name):
                        self.derived.set_matrix(derived_name, work.get_matrix(derived_name))

    def _copy_axes(self, work: DafReader, assured_outputs: Collection[str], optional_outputs: Collection[str]) -> None:
        for axis in _all_axes(assured_outputs):
            if not self.has_axis(axis):
                self.create_axis(axis, work.axis_entries(axis))

        for axis in _all_axes(optional_outputs):
            if not self.has_axis(axis) and work.has_axis(axis):
                self.create_axis(axis, work.axis_entries(axis))

    def _copy_data(self, work: DafReader, name: str, overwrite: bool) -> None:
        if name.endswith(";"):
            return
        if ";" not in name:
            self.set_item(name, work.get_item(name), overwrite=overwrite)
        elif "," in prefix(name, ";"):
            self.set_data2d(name, work.get_matrix(name), overwrite=overwrite)
        else:
            self.set_data1d(name, work.get_vector(name), overwrite=overwrite)


def _back_axis_name(view: StorageView, axis: str, back_axes: Mapping[str, BackAxis]) -> str:
    if view.has_axis(axis):
        return view.base_axis(axis)

    back_axis = back_axes.get(axis)
    if back_axis is None or back_axis.name is None:
        return axis

    return back_axis.name


def _all_axes(names: Collection[str]) -> Set[str]:
    axes: Set[str] = set()
    for name in names:
        if ";" in name:
            axes.update(prefix(name, ";").split(","))
    return axes


CALLABLE = TypeVar("CALLABLE")


def computation(  # pylint: disable=too-many-arguments
    required_inputs: Optional[Mapping[str, str]] = None,
    optional_inputs: Optional[Mapping[str, str]] = None,
    assured_outputs: Optional[Mapping[str, str]] = None,
    optional_outputs: Optional[Mapping[str, str]] = None,
    name: Optional[str] = None,
    storage: Callable[[DafWriter], Optional[StorageWriter]] = lambda *_args, **_kwargs: None,
    derived: Callable[[DafWriter], Optional[StorageWriter]] = lambda *_args, **_kwargs: None,
) -> Callable[[CALLABLE], CALLABLE]:
    """
    Wrap a computation on a ``daf`` data set.

    .. note::

        This is the simplest way to write a "well behaved" generic computation tool using ``daf``.

    The wrapped function must take a `.DafWriter` data set as its first argument and ``overwrite`` as a keyword argument
    with a default value of ``False``. The data set is automatically replaced by a restricted view of the original data
    set, by using `.DafWriter.computation`. The wrapped function will therefore only have access to the
    ``required_inputs`` and ``optional_inputs``. It may freely write temporary results into the data, but only results
    listed in ``assured_outputs`` and ``optional_outputs`` will be copied into the original data set. If ``overwrite``,
    this will overwrite existing data.

    .. note::

        If the computation creates a new axis, list it in the outputs as ``axis;``.

    By default, the ``name`` will append the wrapped function's name (with a ``#`` suffix). The ``storage`` and
    ``derived`` used will, by default, be simple `.MemoryStorage` objects. You can overwrite this in an arbitrary way
    using a helper function that takes all the arguments of the wrapped function and returns the `.StorageWriter`.

    In addition, this will embed the documentation of the inputs and outputs into the function's documentation string.
    We also capture the inputs and outputs of the computation in properties ``__daf_required_inputs__``,
    ``__daf_optional_inputs__``, ``__daf_assured_outputs__`` and ``__daf_optional_inputs`` to support additional
    meta-programming.

    That is, given something like:

    .. code:: python

        @daf.computation(
            required_inputs={
                "foo,bar;baz": '''
                    Input baz per foo and bar.
                    '''
            },
            assured_outputs={
                "foo,bar;vaz": '''
                    Output vaz per foo and bar.
                '''
            },
        )
        def compute_vaz(data: DafWriter, ..., *, ..., overwrite: bool = False) -> None:
            '''
            Compute vaz values.

            __DAF__
            '''
            # Directly work on the `data` here, invoking `DafWriter.computation` is automatic.
            # That is, this can write any intermediate results it wants into the `data`.
            # It must create `foo,bar;vaz`, which will be copied into the caller's data.

    Then ``help(compute_vaz)`` will print:

    .. code:: text

        Compute vaz values.

        **Required Inputs**

        ``foo,bar;baz``
            Input baz per foo and bar.

        **Assured Outputs**

        ``foo,bar;vaz``
            Output vaz per foo and bar.

        If ``overwrite``, will overwrite existing data.
    """

    def wrapped(function: Callable) -> Callable:
        assert (
            function.__doc__ is not None
        ), f"missing documentation for the computation: {function.__module__}.{function.__qualname__}"
        function.__doc__ = dedent(function.__doc__)
        assert "\n__DAF__\n" in function.__doc__, (
            "missing a __DAF__ line in the documentation "
            f"of the computation: {function.__module__}.{function.__qualname__}"
        )

        overwrite_parameter = signature(function).parameters.get("overwrite")
        assert (
            overwrite_parameter is not None
        ), f"missing overwrite parameter for the computation: {function.__module__}.{function.__qualname__}"
        assert overwrite_parameter.default is False, (
            f"default value: {overwrite_parameter.default} is not False for the overwrite parameter "
            f"of the computation: {function.__module__}.{function.__qualname__}"
        )
        assert (
            overwrite_parameter.kind == Parameter.KEYWORD_ONLY
        ), f"overwrite parameter is not keyword-only for the computation: {function.__module__}.{function.__qualname__}"

        daf_required_inputs = None if required_inputs is None else required_inputs.keys()
        daf_optional_inputs = None if optional_inputs is None else optional_inputs.keys()
        daf_assured_outputs = None if assured_outputs is None else assured_outputs.keys()
        daf_optional_outputs = None if optional_outputs is None else optional_outputs.keys()

        def wrapper(data: DafWriter, *args: Any, overwrite: bool = False, **kwargs: Any) -> Any:
            with data.computation(
                required_inputs=daf_required_inputs,
                optional_inputs=daf_optional_inputs,
                assured_outputs=daf_assured_outputs,
                optional_outputs=daf_optional_outputs,
                overwrite=overwrite,
                name=name or f".{function.__module__}.{function.__qualname__}#",
                storage=storage(data, *args, **kwargs),
                derived=derived(data, *args, **kwargs),
            ) as work:
                return function(work, *args, **kwargs)

        setattr(wrapper, "__daf_required_inputs__", daf_required_inputs)
        setattr(wrapper, "__daf_optional_inputs__", daf_optional_inputs)
        setattr(wrapper, "__daf_assured_outputs__", daf_assured_outputs)
        setattr(wrapper, "__daf_optional_outputs__", daf_optional_outputs)
        function.__doc__ = wrapper.__doc__ = function.__doc__.replace(
            "\n__DAF__\n",
            _documentation(required_inputs, optional_inputs, assured_outputs, optional_outputs),
        )

        return function if "sphinx" in sys.argv[0] else wrapper  # type: ignore

    return wrapped  # type: ignore


def _documentation(
    required_inputs: Optional[Mapping[str, str]],
    optional_inputs: Optional[Mapping[str, str]],
    assured_outputs: Optional[Mapping[str, str]],
    optional_outputs: Optional[Mapping[str, str]],
) -> str:
    return "".join(
        _data_documentation(required_inputs, "Required Inputs")
        + _data_documentation(optional_inputs, "Optional Inputs")
        + _data_documentation(assured_outputs, "Assured Outputs")
        + _data_documentation(optional_outputs, "Optional Outputs")
        + ["\nIf ``overwrite``, will overwrite existing data.\n"]
    )


def _data_documentation(data: Optional[Mapping[str, str]], title: str) -> List[str]:
    if data is None or len(data) == 0:
        return []
    texts = ["\n**", title, "**\n\n"]
    for name, description in data.items():
        texts.extend(["``", name, "``\n", "    ", dedent(description).strip().replace("\n", "\n    "), "\n"])
    return texts
