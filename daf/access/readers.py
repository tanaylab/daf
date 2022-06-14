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
from ..typing import as_dense
from ..typing import as_layout
from ..typing import as_matrix
from ..typing import as_vector
from ..typing import be_matrix_in_rows
from ..typing import freeze
from ..typing import is_matrix_in_rows
from ..typing import optimize

# pylint: enable=duplicate-code

__all__ = [
    "DafReader",
    "transpose_name",
]


# pylint: disable=protected-access


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
