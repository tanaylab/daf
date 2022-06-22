"""
Read-only interface for ``daf`` data sets.
"""

# pylint: disable=duplicate-code

from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd  # type: ignore

from ..storage import AxisView
from ..storage import MemoryStorage
from ..storage import StorageChain
from ..storage import StorageReader
from ..storage import StorageView
from ..storage import StorageWriter
from ..storage import extract_1d_axis
from ..storage import extract_2d_axes
from ..storage import parse_2d_axes
from ..typing import ROW_MAJOR
from ..typing import AnyData
from ..typing import FrameInColumns
from ..typing import FrameInRows
from ..typing import MatrixInRows
from ..typing import Series
from ..typing import Vector
from ..typing import as_layout
from ..typing import as_matrix
from ..typing import as_vector
from ..typing import be_dense_in_rows
from ..typing import be_matrix_in_rows
from ..typing import be_sparse_in_rows
from ..typing import be_vector
from ..typing import freeze
from ..typing import is_dense_in_rows
from ..typing import is_matrix_in_rows
from ..typing import is_sparse_in_rows
from ..typing import is_vector
from ..typing import optimize
from . import operations as _operations

# pylint: enable=duplicate-code

__all__ = [
    "DafReader",
    "transpose_name",
]


# pylint: disable=protected-access


class PipelineState:  # pylint: disable=too-few-public-methods
    """
    State while evaluating an operations pipeline.
    """

    def __init__(self, data: Any, ndim: int, axes: Tuple[str, str], source_name: str) -> None:
        #: The data we have computed so far.
        self.data = data

        #: The number of dimensions in the data.
        self.ndim = ndim

        #: The original 2D axes. For 1D data, only the 1st axis is meaningful.
        self.axes = axes

        #: The pipeline so far (for error messages).
        self.pipeline = source_name

        #: The canonical name of the data.
        self.canonical = source_name


class DafReader:  # pylint: disable=too-many-public-methods
    """
    Read-only access to a ``daf`` data set.

    .. note:::

        It is safe to add new data to the ``base`` after wrapping it with a ``DafReader``, but overwriting existing data
        is **not** safe, since any cached ``derived`` data will **not** be updated, causing subtle problems.
    """

    def __init__(self, base: StorageReader, *, derived: Optional[StorageWriter] = None, name: str = ".daf#") -> None:
        """
        If the ``name`` starts with ``.``, it is appended to the ``base`` name. If the name ends with ``#``, we append
        the object id to it to make it unique.
        """
        if name.startswith("."):
            name = base.name + name
        if name.endswith("#"):
            name += str(id(self))

        #: The name of the data set for messages.
        self.name = name

        #: The storage the ``daf`` data set is based on.
        self.base = base.as_reader()

        #: How to store derived data computed from the storage data, for example, an alternate layout of 2D data, of the
        #: result of a pipeline (e.g. ``cell,gene;UMIs|Sum``). By default this is stored in a `.MemoryStorage` so
        #: expensive operations (such as `.as_layout`) will only be computed once in the application's lifetime. You can
        #: explicitly set this to `.NO_STORAGE` to disable the caching, or specify some persistent storage such as
        #: `.FilesWriter` to allow the caching to be reused across multiple application invocations. You can even set
        #: this to be the same as the base storage to have everything (base and derived data) be stored in the same
        #: place.
        self.derived = derived or MemoryStorage(name=self.name + ".derived")

        #: A `.StorageChain` to use to actually access the data. This looks first in ``derived`` and then in the
        #: ``base``.
        self.chain = StorageChain([self.derived, self.base], name=self.name + ".chain")

        for axis in self.base.axis_names():
            if not self.derived.has_axis(axis):
                self.derived.create_axis(axis, freeze(optimize(as_vector(self.chain.axis_entries(axis)))))

    def as_reader(self) -> "DafReader":
        """
        Return the data set as a `.DafReader`.

        This is a no-op (returns self) for "real" read-only data sets, but for writable data sets, it returns a "real"
        read-only wrapper object (that does not implement the writing methods). This ensures that the result can't be
        used to modify the data if passed by mistake to a function that takes a `.DafWriter`.
        """
        return self

    # pylint: disable=duplicate-code

    def description(  # pylint: disable=too-many-branches
        self, *, detail: bool = False, deep: bool = False, description: Optional[Dict] = None
    ) -> Dict:
        """
        Return a dictionary describing the  ``daf`` data set, useful for debugging.

        The result uses the ``name`` field as a key, with a nested dictionary value with the keys ``class``, ``axes``,
        and ``data``.

        If not ``detail``, the ``axes`` will contain a dictionary mapping each axis to a description of its size, and
        the ``data`` will contain just a list of the data names, data, except for `.StorageView` where it will be a
        dictionary mapping each exposed name to the base name.

        If ``detail``, both the ``axes`` and the ``data`` will contain a mapping providing additional
        `.data_description` of the relevant data.

        If ``deep``, there may be additional keys describing the internal storage.

        If ``description`` is provided, collect the result into it. This allows collecting multiple data set
        descriptions into a single overall system state description.

        An example output (without ``detail`` or ``deep``) of a data set with just per-cell-per-gene UMIs:

        .. code:: yaml

            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell,gene;UMIs
        """
        description = description or {}
        if self.name in description:
            return description

        self_description: Dict
        description[self.name] = self_description = {}

        self_description["class"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        self_description["axes"] = self.chain._axes_description(detail=detail)
        self_description["data"] = self.chain._data_description(detail=detail)

        if deep:
            self_description["chain"] = self.chain.name
            self_description["derived"] = self.derived.name
            if hasattr(self, "storage"):
                self_description["storage"] = getattr(self, "storage").name
            self_description["base"] = self.base.name
            self.chain.description(deep=True, detail=detail, description=description)

        return description

    # pylint: enable=duplicate-code

    def verify_has(self, names: Collection[str], *, reason: str = "required") -> None:
        """
        Assert that all the listed data ``names`` exist in the data set, regardless if each is a 0D, 1D or 2D data name.

        To verify an axis exists, list it as ``axis;``.
        """
        for name in names:
            assert self.has_data(name), f"missing the data: {name} which is {reason} in the data set: {self.name}"

    def has_data(self, name: str) -> bool:
        """
        Return whether the data set contains the ``name`` data, regardless of whether it is a 0D, 1D or 2D data.

        To test whether an axis exists, you can use the ``axis;`` name.
        """
        if name.endswith(";"):
            return self.has_axis(name[:-1])
        if ";" not in name:
            return self.has_item(name)
        axes = name.split(";")[0]
        if "," in axes:
            return self.has_data2d(name)
        return self.has_data1d(name)

    def item_names(self) -> List[str]:
        """
        Return the list of names of the 0D data items that exists in the data set, in alphabetical order.
        """
        return sorted(self.chain.item_names())

    def has_item(self, name: str) -> bool:
        """
        Check whether the ``name`` 0D data item exists in the data set.
        """
        return self.chain.has_item(name)

    def get_item(self, name: str) -> Any:
        """
        Access a 0D data item from the data set (which must exist) by its ``name``.

        If the ``name`` contains ``|``, than it should be in the format ``axis;name|operation|operation|...`` or
        ``rows_axis,columns_axis;name|operation|operation|...``, where each ``operation`` should be of the form
        ``Name,param=value,...``. Since we are getting a 0D data item, the pipeline should contain `.Reduction`
        operation(s) that convert the raw input data all the way down to a scalar, e.g. ``axis;name|Sum`` or
        ``rows_axis,columns_axis;name|Sum|Sum``. Any 1D/2D data computed by the pipeline will be cached in the
        ``derived`` storage so it would not have to be re-computed if used in a following ``get_...`` call. You can
        disable this globally by speciying a `.NO_STORAGE` ``derived`` storage in the constructor, or for a specific
        operation by using ``|!Name,...`` instead of ``|Name,...``.

        See `.operations` for the list of built-in operations. Additional operations can be offered by other Python
        packages.
        """
        if "|" in name:
            return self._get_pipeline(name, 0)
        assert self.has_item(name), f"missing item: {name} in the data set: {self.name}"
        return self.chain._get_item(name)

    def axis_names(self) -> List[str]:
        """
        Return the list of names of the axes that exist in the data set, in alphabetical order.
        """
        return sorted(self.chain._axis_names())

    def has_axis(self, axis: str) -> bool:
        """
        Check whether the ``axis`` exists in the data set.
        """
        return self.chain._has_axis(axis)

    def axis_size(self, axis: str) -> int:
        """
        Get the number of entries along some ``axis`` (which must exist).
        """
        assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"
        return self.chain._axis_size(axis)

    def axis_entries(self, axis: str) -> Vector:
        """
        Get the unique name of each entry in the data set along some ``axis`` (which must exist).
        """
        assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"
        return freeze(optimize(as_vector(self.chain._axis_entries(axis))))

    def data1d_names(self, axis: str, *, full: bool = True) -> List[str]:
        """
        Return the names of the 1D data that exists in the data set for a specific ``axis`` (which must exist), in
        alphabetical order.

        The returned names are in the format ``axis;name`` which uniquely identifies the 1D data. If not ``full``, the
        returned names include only the simple ``name`` without the ``axis;`` prefix.
        """
        assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"
        names = sorted(self.chain._data1d_names(axis))
        if not full:
            names = [name.split(";")[1] for name in names]
        return names

    def has_data1d(self, name: str) -> bool:
        """
        Check whether the ``name`` 1D data exists.

        The name must be in the format ``axis;name`` which uniquely identifies the 1D data.
        """
        return self.chain.has_data1d(name)

    def get_vector(self, name: str) -> Vector:
        """
        Get the ``name`` 1D data (which must exist) as a `.Vector`.

        The name must be in the format ``axis;name`` which uniquely identifies the 1D data.

        If the ``name`` contains ``|``, than it should be in the format ``axis;name|operation|operation|...`` or
        ``rows_axis,columns_axis;name|operation|operation|...``, where each ``operation`` should be of the form
        ``Name,param=value,...``. Since we are getting a 1D data item, if the ``name`` starts with a 2D data name, one
        of the operations should be a `.Reduction` converts its input down to a vector of one entry per row of its
        input, e.g. ``rows_axis,columns_axis;name|Sum`` is equivalent to R's ``rowSums``. Any 1D/2D data computed by the
        pipeline will be cached in the ``derived`` storage so it would not have to be re-computed if used in a following
        ``get_...`` call. You can disable this globally by speciying a `.NO_STORAGE` ``derived`` storage in the
        constructor, or for a specific operation by using ``|!Name,...`` instead of ``|Name,...``.

        See `.operations` for the list of built-in operations. Additional operations can be offered by other Python
        packages.
        """
        if "|" in name:
            return be_vector(self._get_pipeline(name, 1))
        axis = extract_1d_axis(name)
        assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"
        assert self.has_data1d(name), f"missing 1D data: {name} in the data set: {self.name}"
        return freeze(optimize(be_vector(as_vector(self.chain._get_data1d(axis, name)), size=self.axis_size(axis))))

    def get_series(self, name: str) -> Series:
        """
        Get the ``name`` 1D data (which must exist) as a ``pandas.Series``.

        The name must be in the format ``axis;name`` which uniquely identifies the 1D data.

        The ``axis`` entries will form the index of the series; if getting a pipeline, starting with 2D data, the index
        of the series will be the entries of the ``rows_axis``.

        If the ``name`` contains ``|``, than it should be in the format ``axis;name|operation|operation|...`` or
        ``rows_axis,columns_axis;name|operation|operation|...``, where each ``operation`` should be of the form
        ``Name,param=value,...``. Since we are getting a 1D data item, if the ``name`` starts with a 2D data name, one
        of the operations should be a `.Reduction` converts its input down to a vector of one entry per row of its
        input, e.g. ``rows_axis,columns_axis;name|Sum`` is equivalent to R's ``rowSums``. Any 1D/2D data computed by the
        pipeline will be cached in the ``derived`` storage so it would not have to be re-computed if used in a following
        ``get_...`` call. You can disable this globally by speciying a `.NO_STORAGE` ``derived`` storage in the
        constructor, or for a specific operation by using ``|!Name,...`` instead of ``|Name,...``.

        See `.operations` for the list of built-in operations. Additional operations can be offered by other Python
        packages.
        """
        vector = self.get_vector(name)
        if "|" in name:
            axis = name.split("|")[0].split(";")[0].split(",")[0]
        else:
            axis = extract_1d_axis(name)
        index = self.axis_entries(axis)
        return freeze(optimize(pd.Series(vector, index=index)))

    def data2d_names(self, axes: Union[str, Tuple[str, str]], *, full: bool = True) -> List[str]:
        """
        Return the names of the 2D data that exists in the data set for a specific pair of ``axes`` (which must exist).

        The returned names are in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data. If
        not ``full``, the returned names include only the simple ``name`` without the ``row_axis,columns_axis;`` prefix.

        .. note::

            Data will be listed in the results even if it is only stored in the other layout (that is, as
            ``columns_axis,rows_axis;name``). Such data can still be fetched (e.g. using `.get_matrix`), in which case
            it will be re-layout internally (and the result will be cached in `.derived`).
        """
        if isinstance(axes, str):
            axes = parse_2d_axes(axes)
        assert self.has_axis(axes[0]), f"missing axis: {axes[0]} in the data set: {self.name}"
        assert self.has_axis(axes[1]), f"missing axis: {axes[1]} in the data set: {self.name}"

        names_set = set(self.chain._data2d_names(axes))
        names_set.update([transpose_name(name) for name in self.chain._data2d_names((axes[1], axes[0]))])
        names = sorted(names_set)
        if not full:
            names = [name.split(";")[1] for name in names]
        return names

    def has_data2d(self, name: str) -> bool:
        """
        Check whether the ``name`` 2D data exists.

        The name must be in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data.

        This will also succeed if only the transposed ``columns_axis,rows_axis;name`` data exists in the data set.
        However, fetching the data in the specified order is likely to be less efficient.
        """
        return self.chain.has_data2d(name) or self.chain.has_data2d(transpose_name(name))

    def get_matrix(self, name: str) -> MatrixInRows:
        """
        Get the ``name`` 2D data (which must exist) as a `.MatrixInRows`.

        The name must be in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data.

        If this required us to re-layout the raw stored data, we cache the result in the ``derived`` storage.

        If the ``name`` contains ``|``, than it should be in the format
        ``rows_axis,columns_axis;name|operation|operation|...``, where each ``operation`` should be of the form
        ``Name,param=value,...``. Since we are getting a 2D data item, all the operations must be `.ElementWise`
        operations. Any 2D data computed by the pipeline will be cached in the ``derived`` storage so it would not have
        to be re-computed if used in a following ``get_...`` call. You can disable this globally by speciying a
        `.NO_STORAGE` ``derived`` storage in the constructor, or for a specific operation by using ``|!Name,...``
        instead of ``|Name,...``.

        The data will always be returned in `.ROW_MAJOR` order, that is, either as a ``numpy`` `.DenseInRows` or as a
        ``scipy.sparse`` `.SparseInRows`, depending on how it is stored. The caller is responsible for distinguishing
        between these two cases (e.g. using `.is_sparse` and/or `.is_dense`) to pick a code path for processing the
        data, as these two types don't really provide the same set of operations.

        See `.operations` for the list of built-in operations. Additional operations can be offered by other Python
        packages.
        """
        if "|" in name:
            return be_matrix_in_rows(self._get_pipeline(name, 2))
        axes = extract_2d_axes(name)
        assert self.has_axis(axes[0]), f"missing axis: {axes[0]} in the data set: {self.name}"
        assert self.has_axis(axes[1]), f"missing axis: {axes[1]} in the data set: {self.name}"
        assert self.has_data2d(name), f"missing 2D data: {name} in the data set: {self.name}"

        transposed_name = transpose_name(name)
        if self.chain.has_data2d(name):
            data2d = self.chain.get_data2d(name)
            matrix = as_matrix(data2d)
        else:
            data2d = self.chain.get_data2d(transposed_name)
            matrix = as_matrix(data2d).transpose()

        if is_matrix_in_rows(matrix):
            matrix_in_rows = matrix
        else:
            if not self.chain.has_data2d(transposed_name):
                transposed_matrix = be_matrix_in_rows(matrix.transpose())
                self.derived.set_matrix(transposed_name, freeze(optimize(transposed_matrix)))
            matrix_in_rows = as_layout(matrix, ROW_MAJOR)

        matrix_in_rows = freeze(optimize(matrix_in_rows))
        if id(matrix_in_rows) != id(data2d):
            self.derived.set_matrix(name, matrix_in_rows)

        return be_matrix_in_rows(matrix_in_rows, shape=(self.axis_size(axes[0]), self.axis_size(axes[1])))

    def get_frame(self, name: str) -> FrameInRows:
        """
        Get the ``name`` 2D data (which must exist) as a ``pandas.DataFrame``.

        The name must be in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data.

        If the ``name`` contains ``|``, than it should be in the format
        ``rows_axis,columns_axis;name|operation|operation|...``, where each ``operation`` should be of the form
        ``Name,param=value,...``. Since we are getting a 2D data item, all the operation must be `.ElementWise`
        operations. Any 2D data computed by the pipeline will be cached in the ``derived`` storage so it would not have
        to be re-computed if used in a following ``get_...`` call. You can disable this globally by speciying a
        `.NO_STORAGE` ``derived`` storage in the constructor, or for a specific operation by using ``|!Name,...``
        instead of ``|Name,...``.

        See `.operations` for the list of built-in operations. Additional operations can be offered by other Python
        packages.

        The data will always be returned in `.ROW_MAJOR` order as a ``numpy`` `.DenseInRows`. Due to ``pandas``
        limitations, if the data is stored as a ``scipy.sparse.spmatrix``, it will be converted to a dense ``numpy`` 2D
        array, which will be cached in ``derived``, as if the ``name`` was suffixed by ``|`` `.Densify`. If you wish to
        disable this caching, explicitly add ``|!Densify`` to the end of the name.

        .. note::

          This (and using `.Densify` in general) should be restricted to cases where the data is known to be "not too
          big". For example, it would be a **bad** idea to ask for a frame of the UMIs of all genes of all cells in a
          data set with ~2M cells and ~30K genes, forcing a dense representation with a size of ~240GB, which is ~40
          times the "mere" ~6GB needed to represent the sparse data.
        """
        name += "|Densify"
        dense = be_dense_in_rows(self.get_matrix(name))
        axes = extract_2d_axes(name)
        index = self.axis_entries(axes[0])
        columns = self.axis_entries(axes[1])
        frame = pd.DataFrame(dense, index=index, columns=columns)
        frame.index.name = axes[0]
        frame.columns.name = axes[1]
        return freeze(optimize(frame))

    def get_columns(self, axis: str, columns: Optional[Sequence[str]] = None) -> FrameInColumns:
        """
        Get an arbitrary collection of 1D data for the same ``axis`` as ``columns`` of a ``pandas.DataFrame``.

        The returned data will always be in `.COLUMN_MAJOR` order.

        If no ``columns`` are specified, returns all the 1D data for the ``axis``, in alphabetical order (that is, as if
        ``columns`` was set to `.data1d_names` with ``full=False`` for the ``axis``).

        The specified ``columns`` names should only be the simple name of each column (possibly followed by
        ``|operation|operation...`` to invoke a pipeline of `.ElementWise` operations). These names will be used as the
        column names of the frame, and the axis entries will be used as the index of the frame.

        See `.operations` for the list of built-in operations. Additional operations can be offered by other Python
        packages.
        """
        index = self.axis_entries(axis)
        columns = columns or self.data1d_names(axis, full=False)
        data = {column: self.get_vector(f"{axis};{column}") for column in columns}
        frame = pd.DataFrame(data, index=index)
        frame.index.name = axis
        return freeze(optimize(frame))

    def view(
        self,
        *,
        axes: Optional[Mapping[str, Union[None, str, AnyData, AxisView]]] = None,
        data: Optional[Mapping[str, Optional[str]]] = None,
        name: str = ".view#",
        cache: Optional[StorageWriter] = None,
        hide_implicit: bool = False,
    ) -> "DafReader":
        """
        Create a read-only view of the data set.

        This can be used to create slices of some axes, rename axes and/or data, and/or hide some data. It is just a
        thin wrapper around the constructor of `.StorageView`; see there for the semantics of the parameters.

        If the ``name`` starts with ``.``, it is appended to both the `.StorageView` and the `.DafReader` names. If the
        name ends with ``#``, we append the object id to it to make it unique.
        """
        # pylint: disable=duplicate-code

        if name.startswith("."):
            name = self.name + name

        unique: Optional[List[None]] = None
        if name.endswith("#"):
            unique = []
            name = name + str(id(unique))

        # pylint: enable=duplicate-code

        for axis in axes or {}:
            assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"

        for data_name in data or {}:
            assert self.has_data(data_name), f"missing data: {data_name} in the data set: {self.name}"

        view = DafReader(
            StorageView(
                self.chain, axes=axes, data=data, cache=cache, hide_implicit=hide_implicit, name=name + ".base"
            ),
            name=name,
        )

        if unique is not None:
            setattr(view, "__daf_unique__", unique)  # Prevent it from being garbage collected.

        return view

    def _get_pipeline(self, pipeline: str, result_ndim: int) -> Any:
        step_texts = pipeline.split("|")
        source_name = step_texts[0]
        operation_texts = step_texts[1:]

        assert (
            ";" in source_name
        ), f"0D source: {source_name} for the pipeline: {pipeline} for the data set: {self.name}"

        pipeline_state: PipelineState

        if "," in source_name:
            axes = extract_2d_axes(source_name)
            pipeline_state = PipelineState(
                data=self.get_matrix(source_name), ndim=2, axes=axes, source_name=source_name
            )
        else:
            axis = extract_1d_axis(source_name)
            axes = (axis, axis)
            pipeline_state = PipelineState(
                data=self.get_vector(source_name), ndim=1, axes=axes, source_name=source_name
            )

        for operation_text in operation_texts:
            self._pipeline_step(pipeline_state, operation_text)

        method = ["get_item", "get_vector", "get_matrix"][result_ndim]
        assert pipeline_state.ndim == result_ndim, (
            f"got: {pipeline_state.ndim}D data instead of the expected: {result_ndim}D data for the method: {method} "
            f"from the pipeline: {pipeline} for the data set: {self.name}"
        )

        return pipeline_state.data

    def _pipeline_step(self, pipeline_state: PipelineState, operation_text: str) -> None:
        pipeline_state.pipeline += "|" + operation_text

        cache = not operation_text.startswith("!")
        if not cache:
            operation_text = operation_text[1:]

        operation_object = self._operation_object(pipeline_state, operation_text)
        canonical_before = pipeline_state.canonical
        pipeline_state.canonical += "|" + operation_object.canonical

        if isinstance(operation_object, _operations.Reduction):
            pipeline_state.canonical = pipeline_state.canonical.replace(";", ",", 1)
            if pipeline_state.ndim == 2:
                pipeline_state.canonical = pipeline_state.canonical.replace(",", ";", 1)
            pipeline_state.ndim -= 1

        if self._fetch_derived(pipeline_state):
            cache = False
        elif isinstance(operation_object, _operations.Reduction):
            DafReader._compute_reduction(pipeline_state, operation_object)
        else:
            nop, cache = self._compute_element_wise(pipeline_state, operation_object, cache)
            if nop:
                assert not cache
                pipeline_state.canonical = canonical_before

        if pipeline_state.ndim > 0:
            pipeline_state.data = freeze(optimize(pipeline_state.data))

        if cache:
            if pipeline_state.ndim == 2:
                self.derived.set_matrix(pipeline_state.canonical, be_matrix_in_rows(pipeline_state.data))
            elif pipeline_state.ndim == 1:
                self.derived.set_vector(pipeline_state.canonical, be_vector(pipeline_state.data))
            else:
                self.derived.set_item(pipeline_state.canonical, pipeline_state.data)

    def _operation_object(
        self, pipeline_state: PipelineState, operation_text: str
    ) -> Union[_operations.ElementWise, _operations.Reduction]:
        operation_parts = operation_text.split(",")
        operation_name = operation_parts[0]
        parameters = operation_parts[1:]

        assert pipeline_state.ndim != 0, (
            f"0D input for the operation: {operation_name} "
            f"in the pipeline: {pipeline_state.pipeline} for the data set: {self.name}"
        )

        operation_class = _operations.Operation._registry.get(operation_name)
        assert operation_class is not None, (
            f"missing operation: {operation_name} "
            f"in the pipeline: {pipeline_state.pipeline} for the data set: {self.name}"
        )

        kwargs: Dict[str, str] = {}
        for parameter in parameters:
            parameter_parts = parameter.split("=")
            assert len(parameter_parts) == 2, (
                f"invalid operation parameter: {parameter} for the operation: {operation_name} "
                f"in the pipeline: {pipeline_state.pipeline} for the data set: {self.name}"
            )
            kwargs[parameter_parts[0]] = parameter_parts[1]

        operation_object = operation_class(_input_dtype=str(pipeline_state.data.dtype), **kwargs)
        assert isinstance(operation_object, (_operations.ElementWise, _operations.Reduction))
        return operation_object

    def _fetch_derived(self, pipeline_state: PipelineState) -> bool:
        if pipeline_state.ndim == 2 and self.derived.has_data2d(pipeline_state.canonical):
            pipeline_state.data = as_matrix(self.derived.get_data2d(pipeline_state.canonical))
            return True

        if pipeline_state.ndim == 1 and self.derived.has_data1d(pipeline_state.canonical):
            pipeline_state.data = as_vector(self.derived.get_data1d(pipeline_state.canonical))
            return True

        if pipeline_state.ndim == 0 and self.derived.has_item(pipeline_state.canonical):
            pipeline_state.data = self.derived.get_item(pipeline_state.canonical)
            return True

        return False

    @staticmethod
    def _compute_reduction(pipeline_state: PipelineState, operation_object: _operations.Reduction) -> None:
        dtype = operation_object.dtype
        shape = pipeline_state.data.shape

        if is_dense_in_rows(pipeline_state.data):
            pipeline_state.data = be_vector(
                operation_object.dense_to_vector(pipeline_state.data), dtype=dtype, size=shape[0]
            )

        elif is_sparse_in_rows(pipeline_state.data):
            pipeline_state.data = be_vector(
                operation_object.sparse_to_vector(pipeline_state.data),
                dtype=dtype,
                size=shape[0],
            )

        else:
            pipeline_state.data = operation_object.vector_to_scalar(be_vector(pipeline_state.data))

    def _compute_element_wise(  # pylint: disable=too-many-return-statements
        self, pipeline_state: PipelineState, operation_object: _operations.ElementWise, cache: bool
    ) -> Tuple[bool, bool]:
        dtype = operation_object.dtype
        shape = pipeline_state.data.shape

        if is_dense_in_rows(pipeline_state.data):

            if operation_object.sparsifies:
                pipeline_state.data = be_sparse_in_rows(
                    operation_object.dense_to_sparse(pipeline_state.data), dtype=dtype, shape=shape
                )
                return False, cache

            if operation_object.nop:
                return True, False

            if cache:
                with self.derived._create_dense_in_rows(
                    pipeline_state.canonical, axes=pipeline_state.axes, dtype=dtype, shape=shape
                ) as output_data:
                    operation_object.dense_to_dense(pipeline_state.data, output_data)
                    pipeline_state.data = output_data
                return False, False

            output_data = be_dense_in_rows(np.empty(pipeline_state.data.shape, dtype=dtype))
            operation_object.dense_to_dense(pipeline_state.data, output_data)
            pipeline_state.data = output_data
            return False, False

        if is_sparse_in_rows(pipeline_state.data):

            if operation_object.densifies:
                if cache:
                    with self.derived._create_dense_in_rows(
                        pipeline_state.canonical, axes=pipeline_state.axes, dtype=dtype, shape=shape
                    ) as output_data:
                        operation_object.sparse_to_dense(pipeline_state.data, output_data)
                        pipeline_state.data = output_data
                    return False, False

                output_data = be_dense_in_rows(np.empty(pipeline_state.data.shape, dtype=dtype))
                operation_object.sparse_to_dense(pipeline_state.data, output_data)
                pipeline_state.data = output_data
                return False, False

            if operation_object.nop:
                return True, False

            pipeline_state.data = be_sparse_in_rows(
                operation_object.sparse_to_sparse(pipeline_state.data), dtype=dtype, shape=shape
            )
            return False, cache

        assert is_vector(pipeline_state.data)

        if operation_object.nop:
            return True, False

        pipeline_state.data = be_vector(
            operation_object.vector_to_vector(be_vector(pipeline_state.data)), dtype=dtype, size=shape[0]
        )
        return False, cache


def transpose_name(name: str) -> str:
    """
    Given a 2D data name ``rows_axis,columns_axis;name`` return the transposed data name
    ``columns_axis,rows_axis;name``.

    .. note::

        This will refuse to transpose pipelined names ``rows_axis,columns_axis;name|operation|...`` as doing so would
        change the meaning of the name. For example, ``cell,gene;UMIs|Sum`` gives the sum of the UMIs of all the genes
        in each cell, while ``gene,cell;UMIs|Sum`` gives the sum of the UMIs for all the cells each gene.
    """
    assert "|" not in name, f"transposing the pipelined name: {name}"

    name_parts = name.split(";")
    assert len(name_parts) == 2, f"invalid 2D data name: {name}"

    axes = name_parts[0].split(",")
    assert len(axes) == 2, f"invalid 2D data name: {name}"

    name_parts[0] = ",".join(reversed(axes))
    return ";".join(name_parts)
