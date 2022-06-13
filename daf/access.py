"""
High-level API for accessing data in a ``daf`` data set.

This interface is intended for ``daf`` users (that is, applications built on top of ``daf``). It allows placing any
"reasonable" type of data into a `.DafWriter`, while ensuring that accessing data in a  `.DafReader` will always return
"clean" data. For example, 2D data returned by `.DafReader` is always `.is_optimal` `.is_frozen` `.MatrixInRows`
regardless of whatever was put into it.

This is in contrast to the low-level `.StorageReader` and `.StorageWriter` interface which, to simplify writing storage
adapters, requires storing "clean" data as above, but does not guarantee anything when accessing the stored data. That
is, `.DafReader` and `.DafWriter` satisfy the `robustness principle
<https://en.wikipedia.org/wiki/Robustness_principle>`_ both "up" towards the application and "down" towards the storage
format adapters.

Accessing data in ``daf`` is based on string names in the following format(s):

* 0D data is identified by a simple ``name``, e.g. ``description`` might be a string describing the overall data set.
  There is no restriction on the data type of 0D data except that it should be reasonably de/serializable to allow
  storing it in a disk file.

* 1D/2D data is specified along some axes, where each ``axis`` has a simple name and a string name for each entry along
  the axis.

* 1D data along some axis is identified by ``axis;name``, e.g. ``cell;age`` might assign an age to every cell in the
  data set. Such data is returned as a ``numpy`` 1D array (that is, `.Vector`) or as a ``pandas.Series``.

* 2D data along two axes is identifies by ``rows_axis,columns_axis;name``, e.g. ``cell,gene;UMIs`` would give the number
  of unique molecular identifiers (that is, the count of mRNA molecules) for each gene in each cell.

  All such data is provided in `.ROW_MAJOR` order; that is, in the above example, each row will describe a cell, and
  will contain (consecutively in memory) the UMIs of each gene. Requesting ``gene,cell;UMIs`` will return data where
  each row describes a cell, and will contain (consecutively in memory) its UMIs in each cell.

  .. note::

    Calling ``.transpose()`` on 2D data does **not** modify the memory layout; this is why it is an extremely fast
    operation. That is, the transpose of ``cell,gene;UMIs`` data contains the same rows, columns, and values as
    ``gene,cell;UMIs`` data, but the former will be in `.COLUMN_MAJOR` layout and the latter will be in `.ROW_MAJOR`
    layout. The two may be "equal" but will **not** be identical when it comes to performance (for non-trivial data
    sizes). For example, summing the UMIs of each cell would be **much** slower for the ``gene,cell;UMIs`` data. It is
    therefore **important** to keep track of the memory order of any non-trivial 2D data, and ensure operations are
    applied to the right layout. Otherwise the code will experience **extreme** slowdowns.

  2D data can be stored in either dense (``numpy`` 2D array) or sparse (``scipy.sparse.csr_matrix`` and
  ``scipy.sparse.csc_matrix``) formats. Which one you'll get when accessing the data will depend on what was stored.
  This allows for efficient storage and processing of large sparse matrices, at the cost of requiring the users to
  examine the fetched data (e.g. using `.is_sparse` or `.is_dense`) to pick the right code path to process it (since
  ``numpy`` arrays and ``scipy.sparse`` matrices don't really support the same operations).

  You can also request the data as a ``pandas.DataFrame`` (that is, `.Frame`), in which case, due to ``pandas``
  limitations, the data will always be returned in the dense (``numpy``) format. The index and columns of the frame
  will be the relevant axis entries.

.. note::

    To avoid ambiguities and to ensure that storing ``daf`` data in files works as expected, the axis and simple data
    names should be restricted to alphanumeric, ``_``, ``-``, and/or ``+`` characters. Other characters may cause
    various problems; in particular, do **not** use ``,``, ``;``, ``=`` or ``|`` characters in simple names.
"""

# pylint: disable=too-many-lines,protected-access

# pylint: disable=duplicate-code

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
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from .storage import AxisView
from .storage import MemoryStorage
from .storage import StorageChain
from .storage import StorageReader
from .storage import StorageView
from .storage import StorageWriter
from .storage import extract_1d_axis
from .storage import extract_2d_axes
from .storage import extract_name
from .storage import parse_2d_axes
from .typing import ROW_MAJOR
from .typing import STR_DTYPE
from .typing import AnyData
from .typing import DenseInRows
from .typing import DType
from .typing import FrameInColumns
from .typing import FrameInRows
from .typing import MatrixInRows
from .typing import Series
from .typing import Vector
from .typing import as_dense
from .typing import as_layout
from .typing import as_matrix
from .typing import as_vector
from .typing import assert_data
from .typing import be_dense_in_rows
from .typing import be_matrix_in_rows
from .typing import be_vector
from .typing import dtype_of
from .typing import freeze
from .typing import has_dtype
from .typing import is_dense_in_rows
from .typing import is_matrix_in_rows
from .typing import is_sparse_in_rows
from .typing import optimize

# pylint: enable=duplicate-code

__all__ = [
    "DafReader",
    "DafWriter",
    "BackAxis",
    "BackData",
    "COMPLETE_DATA",
    "computation",
    "transpose_name",
]


class DafReader:  # pylint: disable=too-many-public-methods
    """
    Read-only access to a ``daf`` data set.

    .. note::

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

        #: How to store derived data computed from the storage data, for example, a different layout of 2D data. By
        #: default this is stored in a `.MemoryStorage` so expensive operations (such as `.as_layout`) will only be
        #: computed once in the application's lifetime. You can explicitly set this to `.NO_STORAGE` to disable the
        #: caching, or specify some persistent storage such as `.FilesWriter` to allow the caching to be reused across
        #: multiple application invocations. You can even set this to be the same as the base storage to have everything
        #: (base and derived data) be stored in the same place.
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

    def item_names(self) -> Collection[str]:
        """
        Return a collection of the names of the 0D data items that exists in the data set.
        """
        return self.chain.item_names()

    def has_item(self, name: str) -> bool:
        """
        Check whether the ``name`` 0D data item exists in the data set.
        """
        return self.chain.has_item(name)

    def get_item(self, name: str) -> Any:
        """
        Access a 0D data item from the data set (which must exist) by its ``name``.
        """
        assert self.has_item(name), f"missing item: {name} in the data set: {self.name}"
        return self.chain._get_item(name)

    def axis_names(self) -> Collection[str]:
        """
        Return a collection of the names of the axes that exist in the data set.
        """
        return self.chain._axis_names()

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

    def data1d_names(self, axis: str) -> Collection[str]:
        """
        Return the names of the 1D data that exists in the data set for a specific ``axis`` (which must exist).

        The returned names are in the format ``axis;name`` which uniquely identifies the 1D data.
        """
        assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"
        return self.chain._data1d_names(axis)

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
        """
        axis = extract_1d_axis(name)
        assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"
        assert self.has_data1d(name), f"missing 1D data: {name} in the data set: {self.name}"
        return freeze(optimize(as_vector(self.chain._get_data1d(axis, name))))

    def get_series(self, name: str) -> Series:
        """
        Get the ``name`` 1D data (which must exist) as a ``pandas.Series``.

        The name must be in the format ``axis;name`` which uniquely identifies the 1D data.

        The ``axis`` entries will form the index of the series.
        """
        axis = extract_1d_axis(name)
        assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"
        assert self.has_data1d(name), f"missing 1D data: {name} in the data set: {self.name}"
        return freeze(
            optimize(pd.Series(self.chain._get_data1d(axis, name), index=self.axis_entries(extract_1d_axis(name))))
        )

    def data2d_names(self, axes: Union[str, Tuple[str, str]]) -> Collection[str]:
        """
        Return the names of the 2D data that exists in the data set for a specific pair of ``axes`` (which must exist).

        The returned names are in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data.

        .. note::

            If two copies of the data exist in transposed axes order, then two different names will be returned. This
            can serve as a *hint* that it is efficient to access the data in both layouts; we can't guarantee this for
            data not created by ``daf`` (e.g., wrapped ``AnnData`` objects).
        """
        if isinstance(axes, str):
            axes = parse_2d_axes(axes)
        assert self.has_axis(axes[0]), f"missing axis: {axes[0]} in the data set: {self.name}"
        assert self.has_axis(axes[1]), f"missing axis: {axes[1]} in the data set: {self.name}"
        return self.chain._data2d_names(axes)

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

        The data will always be returned in `.ROW_MAJOR` order, that is, either as a ``numpy`` `.DenseInRows` or as a
        ``scipy.sparse`` `.SparseInRows`, depending on how it is stored. The caller is responsible for distinguishing
        between these two cases (e.g. using `.is_sparse` and/or `.is_dense`) to pick a code path for processing the
        data, as these two types don't really provide the same set of operations.

        If this required us to re-layout the raw stored data, we cache the result in the ``derived`` storage.
        """
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

        return matrix_in_rows

    def get_frame(self, name: str) -> FrameInRows:
        """
        Get the ``name`` 2D data (which must exist) as a ``pandas.DataFrame``.

        The name must be in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data.

        The data will always be returned in `.ROW_MAJOR` order as a ``numpy`` `.DenseInRows`. Due to ``pandas``
        limitations, if the data is stored as a ``scipy.sparse.spmatrix``, it will be converted to a dense ``numpy`` 2D
        array.

        .. note::

          This should be restricted to cases where the data is known to be "not too big". For example, it would be a
          **bad** idea to ask for a frame of the UMIs of all genes of all cells in a data set with ~2M cells and ~30K
          genes, forcing a dense representation with a size of ~240GB, which is ~40 times the "mere" ~6GB needed to
          represent the sparse data.

        .. todo::

            Cache the dense version of sparse data.
        """
        axes = extract_2d_axes(name)
        assert self.has_axis(axes[0]), f"missing axis: {axes[0]} in the data set: {self.name}"
        assert self.has_axis(axes[1]), f"missing axis: {axes[1]} in the data set: {self.name}"
        assert self.has_data2d(name), f"missing 2D data: {name} in the data set: {self.name}"

        frame = pd.DataFrame(
            as_dense(self.get_matrix(name)), index=self.axis_entries(axes[0]), columns=self.axis_entries(axes[1])
        )
        frame.index.name = axes[0]
        frame.columns.name = axes[1]
        return freeze(optimize(frame))

    def get_columns(self, axis: str, columns: Sequence[str]) -> FrameInColumns:
        """
        Get an arbitrary collection of 1D data for the same ``axis`` as ``columns`` of a ``pandas.DataFrame``.

        The specified ``columns`` names should only be the simple name of each column. These will be used as the column
        names of the frame, and the axis entries will be used as the index of the frame.

        The returned data will always be in `.COLUMN_MAJOR` order.
        """
        assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"
        for name in columns:
            assert self.has_data1d(f"{axis};{name}"), f"missing 1D data: {axis};{name} in the data set: {self.name}"

        frame = pd.DataFrame(
            {column: self.get_vector(f"{axis};{column}") for column in columns}, index=self.axis_entries(axis)
        )
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
        if name.startswith("."):
            name = self.name + name

        unique: Optional[List[None]] = None
        if name.endswith("#"):
            unique = []
            name = name + str(id(unique))

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
        """
        assert not self.has_axis(axis), f"refuse to recreate the axis: {axis} in the data set: {self.name}"
        entries = freeze(optimize(as_vector(entries)))
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
        back_axes: Union[None, Collection[str], Mapping[str, BackAxis]] = None,
        back_data: Union[None, Collection[str], Mapping[str, BackData]] = None,
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
          these will be copied into the original data set. If ``back_axes`` is a ``dict``, it provides a `.BackAxis`
          specifying exactly how to copy each axis back. Otherwise it is just a collection of the new axes to copy
          on success, preserving their name.

        * If ``back_data`` is specified, it should list (some of) the new data created (or modified) by the processing
          code. Each of these will be copied back into the original data set. If ``back_data`` is a ``dict``, it
          provides a `.BackData` specifying exactly how to copy each data back. Otherwise it is a just a collection
          of the data to copy on success, preserving the names and requiring that such data will not use any sliced
          axes.

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
        if name.startswith("."):
            name = self.name + name

        unique: Optional[List[None]] = None
        if name.endswith("#"):
            unique = []
            name = name + str(id(unique))

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
            _back_axes = back_axes
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
            _back_items = {name: back for name, back in back_data.items() if ";" not in name}
            _back_data1d = {
                name: back for name, back in back_data.items() if ";" in name and "," not in name.split(";")[0]
            }
            _back_data2d = {name: back for name, back in back_data.items() if ";" in name and "," in name.split(";")[0]}
        else:
            _back_items = {name: BackData() for name in back_data if ";" not in name}
            _back_data1d = {name: BackData() for name in back_data if ";" in name and "," not in name.split(";")[0]}
            _back_data2d = {name: BackData() for name in back_data if ";" in name and "," in name.split(";")[0]}

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

            axis = extract_1d_axis(data1d)
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
                back_name = extract_name(view.base_data1d(data1d))
            else:
                back_name = extract_name(data1d)

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

            (view_rows_axis, view_columns_axis) = extract_2d_axes(data2d)
            back_rows_axis_name = _back_axis_name(view, view_rows_axis, back_axes)
            back_columns_axis_name = _back_axis_name(view, view_columns_axis, back_axes)

            if back.name is not None:
                back_name = back.name
            elif view.has_data2d(data2d):
                back_name = extract_name(view.base_data2d(data2d))
            else:
                back_name = extract_name(data2d)

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
            view_data[required_name] = extract_name(required_name)
        for optional_name in optional_inputs:
            if self.has_data(optional_name):
                view_data[optional_name] = extract_name(optional_name)

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
            for columns_axis in self.axis_names():
                for derived_name in work.derived.data2d_names((rows_axis, columns_axis)):
                    base_name = derived_name.split("|")[0]
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
        elif "," in name.split(";")[0]:
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
            axes.update(name.split(";")[0].split(","))
    return axes


def transpose_name(name: str) -> str:
    """
    Given a 2D data name ``rows_axis,columns_axis;name`` return the transposed data name
    ``columns_axis,rows_axis;name``.
    """
    parts = name.split(";")
    assert len(parts) == 2, f"invalid 2D data name: {name}"

    axes = parts[0].split(",")
    assert len(axes) == 2, f"invalid 2D data name: {name}"

    parts[0] = ",".join(reversed(axes))
    return ";".join(parts)


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

    ..note::

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
        wrapper.__doc__ = function.__doc__.replace(
            "\n__DAF__\n",
            _documentation(required_inputs, optional_inputs, assured_outputs, optional_outputs),
        )

        return wrapper

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
