"""
Low-level interface for storage objects.

The types here define the abstract interface implemented by all ``daf`` storage format adapters. This interface focuses
on simplicity to to make it easier to **implement** new adapters for specific formats, which makes it inconvenient to
actually **use**. For a more usable interface, see the ``TODOL-DafReader`` and ``TODOL-DafReader`` classes.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any
from typing import Collection
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd  # type: ignore

from ..typing import STR_DTYPE
from ..typing import Array1D
from ..typing import Array2D
from ..typing import Grid
from ..typing import Matrix
from ..typing import Series
from ..typing import Table
from ..typing import Vector
from ..typing import as_array1d
from ..typing import as_array2d
from ..typing import as_grid
from ..typing import assert_data
from ..typing import be_array_in_rows
from ..typing import freeze
from ..typing import is_array1d
from ..typing import is_matrix_in_rows
from ..typing import is_optimal
from ..typing import is_series
from ..typing import is_table
from ..typing import is_vector
from . import chains as _chains

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "StorageReader",
    "StorageWriter",
    "extract_1d_axis",
    "extract_2d_axes",
    "parse_2d_axes",
]


class StorageReader(ABC):
    """
    Low-level read-only storage of data in axes in some format.

    This is an abstract base class defining an interface which is implemented by the concrete storage format adapters.

    .. note::

        Not all the abstract methods are public; if you want to implement a storage adapter yourself, look at the source
        code. You can use the simple `.MemoryStorage` class as a starting point.
    """

    def __init__(self, *, name: Optional[str] = None) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """
        The optional name of the storage for messages.

        By default we report the ``id`` of the object, which isn't very useful, but is better than nothing.
        """
        return self._name or str(id(self))

    def as_reader(self) -> "StorageReader":
        """
        Return the storage as a `.StorageReader`.

        This is a no-op (returns self) for "real" read-only objects, but for writable objects, it returns a "real"
        read-only wrapper object (that does not implement the writing methods). This ensures that the result can't be
        used to modify the data if passed by mistake to a function that takes a `.StorageWriter`.
        """
        return self

    @abstractmethod
    def datum_names(self) -> Collection[str]:
        """
        Return a collection of the names of the 0D ("blobs") data that exists in the storage.
        """

    @abstractmethod
    def has_datum(self, name: str) -> bool:
        """
        Check whether the ``name`` 0D ("blob") datum exists in the storage.
        """

    def get_datum(self, name: str) -> Any:
        """
        Access a 0D ("blob") datum from the storage (which must exist) by its ``name``.
        """
        assert self.has_datum(name), f"missing datum: {name} in the storage: {self.name}"
        return self._get_datum(name)

    @abstractmethod
    def _get_datum(self, name: str) -> Any:
        ...

    @abstractmethod
    def axis_names(self) -> Collection[str]:
        """
        Return a collection of the names of the axes that exist in the storage.
        """

    @abstractmethod
    def has_axis(self, axis: str) -> bool:
        """
        Check whether the ``axis`` exists in the storage.
        """

    def axis_size(self, axis: str) -> int:
        """
        Get the number of entries along some ``axis`` (which must exist).
        """
        assert self.has_axis(axis), f"missing axis: {axis} in the storage: {self.name}"
        return self._axis_size(axis)

    @abstractmethod
    def _axis_size(self, axis: str) -> int:
        ...

    def axis_entries(self, axis: str) -> Array1D:
        """
        Get the unique name of each entry in the storage along some ``axis`` (which must exist).
        """
        assert self.has_axis(axis), f"missing axis: {axis} in the storage: {self.name}"
        return self._axis_entries(axis)

    @abstractmethod
    def _axis_entries(self, axis: str) -> Array1D:
        ...

    def vector_names(self, axis: str) -> Collection[str]:
        """
        Return the names of the 1D data that exists in the storage for a specific ``axis`` (which must exist).

        The returned names are in the format ``axis:name`` which uniquely identifies the 1D data.
        """
        assert self.has_axis(axis), f"missing axis: {axis} in the storage: {self.name}"
        return self._vector_names(axis)

    @abstractmethod
    def _vector_names(self, axis: str) -> Collection[str]:
        ...

    def has_vector(self, name: str) -> bool:
        """
        Check whether the ``name`` 1D data exists.

        The name must be in the format ``axis:name`` which uniquely identifies the 1D data.
        """
        axis = extract_1d_axis(name)
        return self.has_axis(axis) and self._has_vector(axis, name)

    @abstractmethod
    def _has_vector(self, axis: str, name: str) -> bool:
        ...

    def get_array1d(self, name: str) -> Array1D:
        """
        Get the ``name`` 1D data (which must exist) as an `.Array1D`.

        The name must be in the format ``axis:name`` which uniquely identifies the 1D data.
        """
        axis = extract_1d_axis(name)
        assert self.has_axis(axis), f"missing axis: {axis} in the storage: {self.name}"
        assert self.has_vector(name), f"missing vector: {name} in the storage: {self.name}"
        return self._get_array1d(axis, name)

    @abstractmethod
    def _get_array1d(self, axis: str, name: str) -> Array1D:
        ...

    def get_series(self, name: str) -> Series:
        """
        Get the ``name`` 1D data (which must exist) as a ``pandas.Series``.

        The name must be in the format ``axis:name`` which uniquely identifies the 1D data.

        The ``index`` of the series will be the names of the entries of the axis.
        """
        axis = extract_1d_axis(name)
        assert self.has_axis(axis), f"missing axis: {axis} in the storage: {self.name}"
        assert self.has_vector(name), f"missing vector: {name} in the storage: {self.name}"
        index = self.axis_entries(axis)
        array1d = self._get_array1d(axis, name)
        return freeze(pd.Series(array1d, index=index))

    def matrix_names(self, axes: Union[str, Tuple[str, str]]) -> Collection[str]:
        """
        Return the names of the 2D data that exists in the storage for a specific pair of ``axes`` (which must exist).

        The returned names are in the format ``rows_axis,columns_axis:name`` which uniquely identifies the 2D data.

        If two copies of the data exist in transposed axes order, then two different names will be returned.
        """
        if isinstance(axes, str):
            axes = parse_2d_axes(axes)
        assert self.has_axis(axes[0]), f"missing rows axis: {axes[0]} in the storage: {self.name}"
        assert self.has_axis(axes[1]), f"missing columns axis: {axes[1]} in the storage: {self.name}"
        return self._matrix_names(axes)

    @abstractmethod
    def _matrix_names(self, axes: Tuple[str, str]) -> Collection[str]:
        ...

    def has_matrix(self, name: str) -> bool:
        """
        Check whether the ``name`` 2D data exists.

        The name must be in the format ``rows_axis,columns_axis:name`` which uniquely identifies the 2D data.
        """
        axes = extract_2d_axes(name)
        return self.has_axis(axes[0]) and self.has_axis(axes[1]) and self._has_matrix(axes, name)

    @abstractmethod
    def _has_matrix(self, axes: Tuple[str, str], name: str) -> bool:
        ...

    def get_grid(self, name: str) -> Grid:
        """
        Get the ``name`` 2D data (which must exist) as a `.Grid`.

        The name must be in the format ``rows_axis,columns_axis:name`` which uniquely identifies the 2D data.
        """
        axes = extract_2d_axes(name)
        assert self.has_axis(axes[0]), f"missing rows axis: {axes[0]} in the storage: {self.name}"
        assert self.has_axis(axes[1]), f"missing columns axis: {axes[1]} in the storage: {self.name}"
        assert self.has_matrix(name), f"missing matrix: {name} in the storage: {self.name}"
        return self._get_grid(axes, name)

    @abstractmethod
    def _get_grid(self, axes: Tuple[str, str], name: str) -> Grid:
        ...

    def get_table(self, name: str) -> Table:
        """
        Get the ``name`` 2D data (which must exist) as a `.Table` (that is, a ``pandas.DataFrame`` with homogeneous data
        elements).

        The name must be in the format ``rows_axis,columns_axis:name`` which uniquely identifies the 2D data.

        The ``index`` and ``columns`` of the table will be the names of the entries of the rows and column axes.
        """
        axes = extract_2d_axes(name)
        assert self.has_axis(axes[0]), f"missing rows axis: {axes[0]} in the storage: {self.name}"
        assert self.has_axis(axes[1]), f"missing columns axis: {axes[1]} in the storage: {self.name}"
        assert self.has_matrix(name), f"missing matrix: {name} in the storage: {self.name}"
        index = self.axis_entries(axes[0])
        columns = self.axis_entries(axes[1])
        array2d = as_array2d(self._get_grid(axes, name))
        return pd.DataFrame(array2d, index=index, columns=columns)


class StorageWriter(StorageReader):
    """
    Low-level read-write storage of data in axes in formats.

    This is an abstract base class defining an interface which is implemented by the concrete storage formats.

    .. note::

        Not all the abstract methods are public; if you want to implement a storage adapter yourself, look at the source
        code. You can use the simple `.MemoryStorage` class as a starting point.
    """

    def as_reader(self) -> StorageReader:
        """
        Return the storage as a `.StorageReader`.

        This is a no-op (returns self) for "real" read-only objects, but for writable objects, it returns a "real"
        read-only wrapper object (that does not implement the writing methods). This ensures that the result can't be
        used to modify the data if passed by mistake to a function that takes a `.StorageWriter`.
        """
        return _chains.StorageChain([self], name=self._name)

    def update(self, storage: StorageReader, *, overwrite: bool = False) -> None:
        """
        Update the storage with a copy of all the data from another ``storage``.

        If ``overwrite``, this will silently overwrite any existing data.

        Any axes that already exist must have exactly the same entries as in the copied storage.
        """
        self._update_axes(storage)
        self._update_data(storage, overwrite)
        self._update_vectors(storage, overwrite)
        self._update_matrices(storage, overwrite)

    def _update_axes(self, storage: StorageReader) -> None:
        for axis in storage.axis_names():
            new_entries = storage.axis_entries(axis)
            if self.has_axis(axis):
                old_entries = self.axis_entries(axis)
                assert np.all(old_entries == new_entries), (
                    f"inconsistent entries for the axis: {axis} "
                    f"between the storage: {self.name} "
                    f"and the storage: {storage.name}"
                )
            else:
                self.create_axis(axis, new_entries)

    def _update_data(self, storage: StorageReader, overwrite: bool) -> None:
        for datum in storage.datum_names():
            self.set_datum(datum, storage.get_datum(datum), overwrite=overwrite)

    def _update_vectors(self, storage: StorageReader, overwrite: bool) -> None:
        for axis in storage.axis_names():
            for vector in storage.vector_names(axis):
                self.set_vector(vector, storage.get_array1d(vector), overwrite=overwrite)

    def _update_matrices(self, storage: StorageReader, overwrite: bool) -> None:
        for rows_axis in storage.axis_names():
            for columns_axis in storage.axis_names():
                for matrix in storage.matrix_names((rows_axis, columns_axis)):
                    self.set_matrix(matrix, storage.get_grid(matrix), overwrite=overwrite)

    def set_datum(self, name: str, datum: Any, *, overwrite: bool = False) -> None:
        """
        Set a ``name`` 0D ("blob") ``datum``.

        If ``overwrite``, will silently overwrite an existing datum of the same name, otherwise overwriting will fail.
        """
        assert overwrite or not self.has_datum(
            name
        ), f"refuse to overwrite the datum: {name} in the storage: {self.name}"

        self._set_datum(name, datum)

    @abstractmethod
    def _set_datum(self, name: str, datum: Any) -> None:
        ...

    def create_axis(self, axis: str, entries: Array1D) -> None:
        """
        Create a new ``axis`` and the unique ``entries`` identifying each entry along the axis.

        It is always an error to overwrite an existing axis.
        """
        assert_data(is_array1d(entries, dtype=STR_DTYPE), "1D np.ndarray", entries, STR_DTYPE)

        assert not self.has_axis(axis), f"refuse to recreate the axis: {axis} in the storage: {self.name}"

        self._create_axis(axis, freeze(entries))

    @abstractmethod
    def _create_axis(self, axis: str, entries: Array1D) -> None:
        ...

    def set_vector(self, name: str, vector: Vector, *, overwrite: bool = False) -> None:
        """
        Set a ``name`` 1D data.

        The name must be in the format ``axis:name`` which uniquely identifies the 1D data.

        If ``overwrite``, will silently overwrite an existing vector of the same name, otherwise overwriting will fail.
        """
        assert_data(is_vector(vector), "vector", vector, None)

        assert overwrite or not self.has_vector(
            name
        ), f"refuse to overwrite the vector: {name} in the storage: {self.name}"

        axis = extract_1d_axis(name)

        assert len(vector) == self.axis_size(axis), (
            f"vector: {name} size: {len(vector)} is different from axis size: {self.axis_size(axis)} "
            f"in the storage: {self.name}"
        )

        if is_series(vector):
            assert np.all(
                vector.index == self.axis_entries(axis)
            ), f"series index entries for vector: {name} are different from axis in the storage: {self.name}"

        array1d = as_array1d(vector)
        self._set_array1d(axis, name, freeze(array1d))

    @abstractmethod
    def _set_array1d(self, axis: str, name: str, array1d: Array1D) -> None:
        ...

    def set_matrix(self, name: str, matrix: Matrix, *, overwrite: bool = False) -> None:
        """
        Set a ``name`` 2D data ``matrix``.

        The name must be in the format ``rows_axis,columns_axis:name`` which uniquely identifies the 2D data. The data
        must be in row-major order, and optimized.

        If ``overwrite``, will silently overwrite an existing matrix of the same name, otherwise overwriting will fail.
        """
        assert_data(is_matrix_in_rows(matrix), "row-major matrix", matrix, None)
        assert_data(is_optimal(matrix), "optimal matrix", matrix, None)

        assert overwrite or not self.has_matrix(
            name
        ), f"refuse to overwrite the matrix: {name} in the storage: {self.name}"

        axes = extract_2d_axes(name)

        assert matrix.shape[0] == self.axis_size(axes[0]), (
            f"matrix: {name} rows: {matrix.shape[0]} is different from axis size: {self.axis_size(axes[0])} "
            f"in the storage: {self.name}"
        )
        assert matrix.shape[1] == self.axis_size(axes[1]), (
            f"matrix: {name} columns: {matrix.shape[1]} is different from axis size: {self.axis_size(axes[1])} "
            f"in the storage: {self.name}"
        )

        if is_table(matrix):
            assert np.all(
                matrix.index == self.axis_entries(axes[0])
            ), f"table rows index entries for matrix: {name} are different from axis in the storage: {self.name}"
            assert np.all(
                matrix.columns == self.axis_entries(axes[1])
            ), f"table columns index entries for matrix: {name} are different from axis in the storage: {self.name}"

        self._set_grid(axes, name, freeze(as_grid(matrix)))

    @abstractmethod
    def _set_grid(self, axes: Tuple[str, str], name: str, grid: Grid) -> None:
        ...

    @contextmanager
    def create_array2d(self, name: str, *, dtype: str, overwrite: bool = False) -> Generator[Array2D, None, None]:
        """
        Create an uninitialized 2D dense array of some ``dtype`` to be set by the ``name`` in the storage, expecting the
        code to initialize it.

        The name must be in the format ``rows_axis,columns_axis:name`` which uniquely identifies the 2D data. The
        returned array is in row-major order.

        Expected usage is:

        .. code::

            with storage.create_array2d(name="rows_axis,columns_axis:name", dtype="...") as array2d:
                # Here the array is still not necessarily set inside the storage,
                # that is, one can't assume ``get_grid`` will access it.
                # It is only available for filling in the values:
                array2d[..., ...] = ...

            # Here the array IS set inside the storage,
            # that is, one can use ``get_grid`` to access it.

        This allows ``TODOL-Files`` to create the array on disk, without first having to create an in-memory copy. By
        default (for other adapters), this just creates and returns an uninitialized in-memory 2D dense array, then
        calls `.StorageWriter.set_matrix` with the initialized result.
        """
        assert overwrite or not self.has_matrix(
            name
        ), f"refuse to overwrite the matrix: {name} in the storage: {self.name}"

        axes = extract_2d_axes(name)

        shape = (self.axis_size(axes[0]), self.axis_size(axes[1]))
        with self._create_array2d(shape, name, dtype) as array2d:
            yield array2d

    @contextmanager
    def _create_array2d(self, shape: Tuple[int, int], name: str, dtype: str) -> Generator[Array2D, None, None]:
        array2d = be_array_in_rows(np.empty(shape, dtype=dtype), dtype=dtype)
        yield array2d
        self.set_matrix(name, array2d, overwrite=True)


def extract_1d_axis(name: str) -> str:
    """
    Extract the axis out of a ``axis:name`` 1D data name.
    """
    parts = name.split(":")
    assert len(parts) == 2, f"invalid 1D data name: {name}"
    return parts[0]


def extract_2d_axes(name: str) -> Tuple[str, str]:
    """
    Extract the axes out of ``rows_axis,column_axis:name`` 2D data name.
    """
    parts = name.split(":")
    assert len(parts) == 2, f"invalid 2D data name: {name}"
    return parse_2d_axes(parts[0])


def parse_2d_axes(name: str) -> Tuple[str, str]:
    """
    Parse the axes in a ``rows_axis,column_axis`` 2D data axes.
    """
    axes = name.split(",")
    assert len(axes) == 2, f"invalid 2D data name: {name}"
    return (axes[0], axes[1])
