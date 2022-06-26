"""
The types here define the abstract interface implemented by all ``daf`` storage format adapters. This interface focuses
on simplicity to to make it easier to **implement** new adapters for specific formats, which makes it inconvenient to
actually **use**. For a more usable interface, see the `.DafReader` and `.DafWriter` classes.

For example, we only require storage objects to accept `.is_optimal` `.is_frozen` `.MatrixInRows` 2D data, but we allow
storage objects to return almost anything they happen to contain (as long as it is `.Known2D` data). This simplifies
writing storage format adapters that access arbitrary data.

In general ``daf`` users would not be interested in the abstract storage interface defined here, other than to construct
storage objects (using the concrete implementation constructors) and possibly accessing ``.name``, ``.description``, and
``.as_reader``.

It is the higher level `.DafReader` and `.DafWriter` classes which will actually use the API exposed here. It is still
documented as it is exported, and it gives deeper insight into how ``daf`` works.

Implementing a new storage format adapter requires implementing the abstract methods. These are essentially simplified
versions of the above and are omitted from the documentation to reduce clutter. If you wish to implement an adapter to a
new storage format, you are advised to look at the sources and consider the existing implementations (in particular,
`.MemoryStorage`) as a starting point.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any
from typing import Collection
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from ..typing import FIXED_DTYPES
from ..typing import ROW_MAJOR
from ..typing import STR_DTYPE
from ..typing import DenseInRows
from ..typing import DType
from ..typing import Known1D
from ..typing import Known2D
from ..typing import MatrixInRows
from ..typing import Vector
from ..typing import as_layout
from ..typing import as_matrix
from ..typing import as_vector
from ..typing import assert_data
from ..typing import be_dense_in_rows
from ..typing import data_description
from ..typing import freeze
from ..typing import is_dtype
from ..typing import is_frozen
from ..typing import is_matrix_in_rows
from ..typing import is_optimal
from ..typing import is_vector
from ..typing import optimize
from . import chains as _chains

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "StorageReader",
    "StorageWriter",
    "prefix",
    "suffix",
]


class StorageReader(ABC):
    """
    Low-level read-only storage of data in axes in some format.

    This is an abstract base class defining an interface which is implemented by the concrete storage format adapters.

    .. note::

        The abstract methods are not public; if you want to implement a storage adapter yourself, look at the source
        code. You can use the simple `.MemoryStorage` class as a starting point.
    """

    def __init__(self, *, name: str) -> None:
        #: The name of the storage for messages.
        self.name = name

    def as_reader(self) -> "StorageReader":
        """
        Return the storage as a `.StorageReader`.

        This is a no-op (returns self) for "real" read-only storage, but for writable storage, it returns a "real"
        read-only wrapper object (that does not implement the writing methods). This ensures that the result can't be
        used to modify the data if passed by mistake to a function that takes a `.StorageWriter`.
        """
        return self

    # pylint: disable=duplicate-code

    def description(self, *, detail: bool = False, deep: bool = False, description: Optional[Dict] = None) -> Dict:
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

        .. todo::

            Make the `.StorageReader.description` of a `.StorageView` list the mapping from the exposed and the base
            names, and some indication of axis slicing.
        """
        description = description or {}
        if self.name in description:
            return description

        self_description: Dict
        description[self.name] = self_description = {}

        self_description["class"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"

        self._self_description(self_description, detail=detail)

        self_description["axes"] = self._axes_description(detail=detail)

        self_description["data"] = self._data_description(detail=detail)

        if deep:
            self._deep_description(description, self_description, detail=detail)

        return description

    # pylint: enable=duplicate-code

    def _self_description(self, self_description: Dict, *, detail: bool) -> None:
        pass

    def _axes_description(self, *, detail: bool) -> Dict:
        axes: Dict = {}
        for axis in sorted(self._axis_names()):
            if detail:
                axes[axis] = data_description(self._axis_entries(axis))
            else:
                axes[axis] = str(self._axis_size(axis)) + " entries"
        return axes

    def _data_description(self, *, detail: bool) -> Union[Dict, List[str]]:
        axes = sorted(self._axis_names())

        if not detail:
            data = sorted(self._item_names())
            for axis in axes:
                data.extend(sorted(self._data1d_names(axis)))
            for rows_axis in axes:
                for columns_axis in axes:
                    data.extend(sorted(self._data2d_names((rows_axis, columns_axis))))
            return data

        details: Dict = {}

        for name in sorted(self._item_names()):
            details[name] = data_description(self._get_item(name))

        for axis in axes:
            for name in sorted(self._data1d_names(axis)):
                details[name] = data_description(self._get_data1d(axis, name))

        for rows_axis in axes:
            for columns_axis in axes:
                for name in sorted(self._data2d_names((rows_axis, columns_axis))):
                    details[name] = data_description(self._get_data2d((rows_axis, columns_axis), name))

        return details

    def _deep_description(self, description: Dict, self_description: Dict, *, detail: bool) -> None:
        pass

    def item_names(self) -> Collection[str]:
        """
        Return a collection of the names of the 0D data items that exists in the storage.
        """
        return self._item_names()

    @abstractmethod
    def _item_names(self) -> Collection[str]:
        ...

    def has_item(self, name: str) -> bool:
        """
        Check whether the ``name`` 0D data item exists in the storage.
        """
        return self._has_item(name)

    @abstractmethod
    def _has_item(self, name: str) -> bool:
        ...

    def get_item(self, name: str) -> Any:
        """
        Access a 0D data item from the storage (which must exist) by its ``name``.
        """
        assert self._has_item(name), f"missing item: {name} in the storage: {self.name}"
        return self._get_item(name)

    @abstractmethod
    def _get_item(self, name: str) -> Any:
        ...

    def axis_names(self) -> Collection[str]:
        """
        Return a collection of the names of the axes that exist in the storage.
        """
        return self._axis_names()

    @abstractmethod
    def _axis_names(self) -> Collection[str]:
        ...

    def has_axis(self, axis: str) -> bool:
        """
        Check whether the ``axis`` exists in the storage.
        """
        return self._has_axis(axis)

    @abstractmethod
    def _has_axis(self, axis: str) -> bool:
        ...

    def axis_size(self, axis: str) -> int:
        """
        Get the number of entries along some ``axis`` (which must exist).
        """
        assert self._has_axis(axis), f"missing axis: {axis} in the storage: {self.name}"
        return self._axis_size(axis)

    @abstractmethod
    def _axis_size(self, axis: str) -> int:
        ...

    def axis_entries(self, axis: str) -> Known1D:
        """
        Get the unique name of each entry in the storage along some ``axis`` (which must exist).
        """
        assert self._has_axis(axis), f"missing axis: {axis} in the storage: {self.name}"
        return self._axis_entries(axis)

    @abstractmethod
    def _axis_entries(self, axis: str) -> Known1D:
        ...

    def data1d_names(self, axis: str) -> Collection[str]:
        """
        Return the names of the 1D data that exists in the storage for a specific ``axis`` (which must exist).

        The returned names are in the format ``axis;name`` which uniquely identifies the 1D data.
        """
        assert self._has_axis(axis), f"missing axis: {axis} in the storage: {self.name}"
        return self._data1d_names(axis)

    @abstractmethod
    def _data1d_names(self, axis: str) -> Collection[str]:
        ...

    def has_data1d(self, name: str) -> bool:
        """
        Check whether the ``name`` 1D data exists.

        The name must be in the format ``axis;name`` which uniquely identifies the 1D data.
        """
        assert ";" in name, f"0D name: {name} for: has_data1d for the storage: {self.name}"
        axis = prefix(name, ";")
        axes = axis.split(",")
        assert len(axes) == 1, f"{len(axes)}D name: {name} for: has_data1d for the storage: {self.name}"
        return self._has_axis(axis) and self._has_data1d(axis, name)

    @abstractmethod
    def _has_data1d(self, axis: str, name: str) -> bool:
        ...

    def get_data1d(self, name: str) -> Known1D:
        """
        Get the ``name`` 1D data (which must exist) as an `.Known1D`.

        The name must be in the format ``axis;name`` which uniquely identifies the 1D data.
        """
        assert ";" in name, f"0D name: {name} for: get_data1d for the storage: {self.name}"
        axis = prefix(name, ";")
        axes = axis.split(",")
        assert len(axes) == 1, f"{len(axes)}D name: {name} for: has_data1d for the storage: {self.name}"
        assert self._has_axis(axis), f"missing axis: {axis} in the storage: {self.name}"
        assert self._has_data1d(axis, name), f"missing 1D data: {name} in the storage: {self.name}"
        return self._get_data1d(axis, name)

    @abstractmethod
    def _get_data1d(self, axis: str, name: str) -> Known1D:
        ...

    def data2d_names(self, axes: Union[str, Tuple[str, str]]) -> Collection[str]:
        """
        Return the names of the 2D data that exists in the storage for a specific pair of ``axes`` (which must exist).

        The returned names are in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data.

        If two copies of the data exist in transposed axes order, then two different names will be returned.
        """
        if isinstance(axes, str):
            parts = axes.split(",")
            assert len(parts) == 2, f"{len(axes)}D axes: {axes} for: data2d_names for the storage: {self.name}"
            axes = (parts[0], parts[1])
        assert self._has_axis(axes[0]), f"missing axis: {axes[0]} in the storage: {self.name}"
        assert self._has_axis(axes[1]), f"missing axis: {axes[1]} in the storage: {self.name}"
        return self._data2d_names(axes)

    @abstractmethod
    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:
        ...

    def has_data2d(self, name: str) -> bool:
        """
        Check whether the ``name`` 2D data exists.

        The name must be in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data.
        """
        assert ";" in name, f"0D name: {name} for: has_data2d for the storage: {self.name}"
        axes = prefix(name, ";").split(",")
        assert len(axes) == 2, f"{len(axes)}D name: {name} for: has_data2d for the storage: {self.name}"
        return self._has_axis(axes[0]) and self._has_axis(axes[1]) and self._has_data2d((axes[0], axes[1]), name)

    @abstractmethod
    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:
        ...

    def get_data2d(self, name: str) -> Known2D:
        """
        Get the ``name`` 2D data (which must exist).

        The name must be in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data.
        """
        assert ";" in name, f"0D name: {name} for: get_data2d for the storage: {self.name}"
        axes = prefix(name, ";").split(",")
        assert len(axes) == 2, f"{len(axes)}D name: {name} for: has_data2d for the storage: {self.name}"
        assert self._has_axis(axes[0]), f"missing axis: {axes[0]} in the storage: {self.name}"
        assert self._has_axis(axes[1]), f"missing axis: {axes[1]} in the storage: {self.name}"
        assert self._has_data2d((axes[0], axes[1]), name), f"missing 2D data: {name} in the storage: {self.name}"
        return self._get_data2d((axes[0], axes[1]), name)

    @abstractmethod
    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Known2D:
        ...


class StorageWriter(StorageReader):
    """
    Low-level read-write storage of data in axes in formats.

    This is an abstract base class defining an interface which is implemented by the concrete storage formats.

    .. note::

        The abstract methods are not public; if you want to implement a storage adapter yourself, look at the source
        code. You can use the simple `.MemoryStorage` class as a starting point.

    .. todo::

        The `.StorageWriter` interface needs to be extended to allow for deleting data (dangerous as this may be).
        Currently the only supported way is to create a `.StorageView` that hides some data, saving that into a new
        storage, and removing the old storage, which is "unreasonable" even though this is a very rare operation.
    """

    def as_reader(self) -> StorageReader:
        """
        Return the storage as a `.StorageReader`.

        This is a no-op (returns self) for "real" read-only storage, but for writable storage, it returns a "real"
        read-only wrapper object (that does not implement the writing methods). This ensures that the result can't be
        used to modify the data if passed by mistake to a function that takes a `.StorageWriter`.
        """
        return _chains.StorageChain([self], name=self.name + ".as_reader")

    def update(self, storage: StorageReader, *, overwrite: bool = False) -> None:
        """
        Update the storage with a copy of all the data from another ``storage``.

        If ``overwrite``, this will silently overwrite any existing data.

        Any axes that already exist must have exactly the same entries as in the copied storage.

        This can be used to copy data between different storage objects. A common idiom is creating a new empty storage
        and then calling ``update`` to fill it with the data from some other storage (often a `.StorageView` and/or a
        `.StorageChain` to control exactly what is being copied). A notable exception is `.AnnDataWriter` which, due to
        `AnnData <https://pypi.org/project/anndata>`_ limitations, must be given the copied storage in its constructor.

        .. note::

            This will convert any non-`.Matrix` 2D data in the ``storage`` into a `.Matrix` to satisfy our promise that
            we only put `.Matrix` data into a storage (even though we allow it to return any `.Known2D` data).
        """
        self._update_axes(storage)
        self._update_data(storage, overwrite)
        self._update_vector(storage, overwrite)
        self._update_matrices(storage, overwrite)

    def _update_axes(self, storage: StorageReader) -> None:
        for axis in storage.axis_names():
            new_entries = freeze(optimize(as_vector(storage.axis_entries(axis))))
            if self._has_axis(axis):
                old_entries = as_vector(self._axis_entries(axis))
                assert np.all(old_entries == new_entries), (
                    f"inconsistent entries for the axis: {axis} "
                    f"between the storage: {self.name} "
                    f"and the storage: {storage.name}"
                )
            else:
                self._create_axis(axis, new_entries)

    def _update_data(self, storage: StorageReader, overwrite: bool) -> None:
        for name in storage.item_names():
            self.set_item(name, storage.get_item(name), overwrite=overwrite)

    def _update_vector(self, storage: StorageReader, overwrite: bool) -> None:
        for axis in storage.axis_names():
            for name in storage.data1d_names(axis):
                self.set_vector(name, freeze(optimize(as_vector(storage.get_data1d(name)))), overwrite=overwrite)

    def _update_matrices(self, storage: StorageReader, overwrite: bool) -> None:
        for rows_axis in storage.axis_names():
            for columns_axis in storage.axis_names():
                for name in storage.data2d_names((rows_axis, columns_axis)):
                    self.set_matrix(
                        name,
                        freeze(optimize(as_layout(as_matrix(storage.get_data2d(name)), ROW_MAJOR))),
                        overwrite=overwrite,
                    )

    def set_item(self, name: str, item: Any, *, overwrite: bool = False) -> None:
        """
        Set a ``name`` 0D data ``item``.

        If ``overwrite``, will silently overwrite an existing item of the same name, otherwise overwriting will fail.
        """
        assert overwrite or not self._has_item(
            name
        ), f"refuse to overwrite the item: {name} in the storage: {self.name}"

        self._set_item(name, item)

    @abstractmethod
    def _set_item(self, name: str, item: Any) -> None:
        ...

    def create_axis(self, axis: str, entries: Vector) -> None:
        """
        Create a new ``axis`` and the unique ``entries`` identifying each entry along the axis.

        The ``entries`` must be `.is_optimal` `.is_frozen` `.Vector` and contain string data.

        It is always an error to overwrite an existing axis.
        """
        assert_data(condition=is_vector(entries, dtype=STR_DTYPE), kind="1D np.ndarray", data=entries, dtype=STR_DTYPE)
        assert_data(is_optimal(entries), "optimal 1D np.ndarray", entries, dtype=STR_DTYPE)
        assert_data(is_frozen(entries), "frozen 1D np.ndarray", entries, dtype=STR_DTYPE)

        assert not self._has_axis(axis), f"refuse to recreate the axis: {axis} in the storage: {self.name}"

        self._create_axis(axis, entries)

    @abstractmethod
    def _create_axis(self, axis: str, entries: Vector) -> None:
        ...

    def set_vector(self, name: str, vector: Vector, *, overwrite: bool = False) -> None:
        """
        Set a ``name`` `.Vector` data.

        The name must be in the format ``axis;name`` which uniquely identifies the 1D data. The data must be
        `.is_frozen` and `.is_optimal`.

        If ``overwrite``, will silently overwrite an existing 1D data of the same name, otherwise overwriting will fail.
        """
        assert_data(is_vector(vector), "1D numpy.ndarray", vector)
        assert_data(is_frozen(vector), "frozen 1D numpy.ndarray", vector)
        assert_data(is_optimal(vector), "optimal 1D numpy.ndarray", vector)

        assert ";" in name, f"0D name: {name} for: set_vector for the storage: {self.name}"
        axis = prefix(name, ";")
        axes = axis.split(",")
        assert len(axes) == 1, f"{len(axes)}D name: {name} for: set_vector for the storage: {self.name}"
        assert self._has_axis(axis), f"missing axis: {axis} in the storage: {self.name}"

        assert overwrite or not self._has_data1d(
            axis, name
        ), f"refuse to overwrite the 1D data: {name} in the storage: {self.name}"

        assert len(vector) == self._axis_size(axis), (
            f"1D data: {name} size: {len(vector)} is different from axis size: {self._axis_size(axis)} "
            f"in the storage: {self.name}"
        )

        self._set_vector(axis, name, vector)

    @abstractmethod
    def _set_vector(self, axis: str, name: str, vector: Vector) -> None:
        ...

    def set_matrix(self, name: str, matrix: MatrixInRows, *, overwrite: bool = False) -> None:
        """
        Set a ``name`` ``matrix``.

        The name must be in the format ``rows_axis,columns_axis;name`` which uniquely identifies the 2D data. The data
        must be an `.is_frozen` `.is_optimal` `.MatrixInRows`.

        If ``overwrite``, will silently overwrite an existing 2D data of the same name, otherwise overwriting will fail.
        """
        assert_data(is_matrix_in_rows(matrix), "row-major matrix", matrix)
        assert_data(is_optimal(matrix), "optimal matrix", matrix)
        assert_data(is_frozen(matrix), "frozen matrix", matrix)

        assert ";" in name, f"0D name: {name} for: set_matrix for the storage: {self.name}"
        axes = prefix(name, ";").split(",")
        assert len(axes) == 2, f"{len(axes)}D name: {name} for: set_matrix for the storage: {self.name}"
        assert self._has_axis(axes[0]), f"missing axis: {axes[0]} in the storage: {self.name}"
        assert self._has_axis(axes[1]), f"missing axis: {axes[1]} in the storage: {self.name}"

        assert overwrite or not self._has_data2d(
            (axes[0], axes[1]), name
        ), f"refuse to overwrite the 2D data: {name} in the storage: {self.name}"

        assert matrix.shape[0] == self._axis_size(axes[0]), (
            f"2D data: {name} rows: {matrix.shape[0]} is different from axis size: {self._axis_size(axes[0])} "
            f"in the storage: {self.name}"
        )
        assert matrix.shape[1] == self._axis_size(axes[1]), (
            f"2D data: {name} columns: {matrix.shape[1]} is different from axis size: {self._axis_size(axes[1])} "
            f"in the storage: {self.name}"
        )

        self._set_matrix((axes[0], axes[1]), name, matrix)

    @abstractmethod
    def _set_matrix(self, axes: Tuple[str, str], name: str, matrix: MatrixInRows) -> None:
        ...

    @contextmanager
    def create_dense_in_rows(
        self, name: str, *, dtype: DType, overwrite: bool = False
    ) -> Generator[DenseInRows, None, None]:
        """
        Create an uninitialized `.ROW_MAJOR` .`DenseInRows` of some ``dtype`` to be set by the ``name`` in the storage,
        expecting the code to initialize it.
        """
        assert ";" in name, f"0D name: {name} for: create_dense_in_rows for the storage: {self.name}"
        axes = prefix(name, ";").split(",")
        assert len(axes) == 2, f"{len(axes)}D name: {name} for: create_dense_in_rows for the storage: {self.name}"
        assert self._has_axis(axes[0]), f"missing axis: {axes[0]} in the storage: {self.name}"
        assert self._has_axis(axes[1]), f"missing axis: {axes[1]} in the storage: {self.name}"

        assert overwrite or not self._has_data2d(
            (axes[0], axes[1]), name
        ), f"refuse to overwrite the 2D data: {name} in the storage: {self.name}"

        assert is_dtype(dtype, FIXED_DTYPES), f"unsupported dtype: {dtype}"

        shape = (self._axis_size(axes[0]), self._axis_size(axes[1]))
        with self._create_dense_in_rows(name, axes=(axes[0], axes[1]), shape=shape, dtype=dtype) as dense:
            yield dense

    @contextmanager
    def _create_dense_in_rows(
        self, name: str, *, axes: Tuple[str, str], shape: Tuple[int, int], dtype: DType
    ) -> Generator[DenseInRows, None, None]:
        dense = be_dense_in_rows(np.empty(shape, dtype=dtype), dtype=dtype)
        yield dense
        self._set_matrix(axes, name, freeze(optimize(dense)))


def prefix(text: str, char: str) -> str:
    """
    Return the characters in the ``text`` before the separator ``char`` (which need not exist).

    It would have been much nicer if this was a method of ``str``.
    """
    return text.split(char, 1)[0]


def suffix(text: str, char: str) -> str:
    """
    Return the characters in the ``text`` after the separator ``char`` (which must exist).

    It would have been much nicer if this was a method of ``str``.
    """
    return text.split(char, 1)[1]
