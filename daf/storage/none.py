"""
The :py:obs:`NO_STORAGE` is used to disable caching (e.g., caching sliced data in a
:py:obj:`daf.storage.views.StorageView`).

Creating or setting data inside ``NO_STORAGE`` has no effect. Unlike a real storage, it allows querying for names of
axes that don't exist (since in it, no axes exist). It also
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from typing import Collection
from typing import Generator
from typing import Tuple

from ..typing import Array1D
from ..typing import Array2D
from ..typing import Grid
from ..typing import Matrix
from ..typing import Series
from ..typing import Table
from ..typing import Vector
from . import interface as _interface

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "NO_STORAGE",
]


class NoStorage(_interface.StorageWriter):  # pylint: disable=too-many-public-methods,unused-argument
    """
    A dummy storage which doesn't actually store anything.
    """

    def __init__(self) -> None:
        super().__init__(name="NoStorage")

    def as_reader(self) -> _interface.StorageReader:
        return self

    def datum_names(self) -> Collection[str]:
        return []

    def has_datum(self, name: str) -> bool:
        return False

    def get_datum(self, name: str) -> Any:
        assert False, "never happens"

    def _get_datum(self, name: str) -> Any:
        assert False, "never happens"

    def axis_names(self) -> Collection[str]:
        return []

    def has_axis(self, axis: str) -> bool:
        return False

    def axis_size(self, axis: str) -> int:
        assert False, "never happens"

    def _axis_size(self, axis: str) -> int:
        assert False, "never happens"

    def axis_entries(self, axis: str) -> Array1D:
        assert False, "never happens"

    def _axis_entries(self, axis: str) -> Array1D:
        assert False, "never happens"

    def vector_names(self, axis: str) -> Collection[str]:
        return []

    def _vector_names(self, axis: str) -> Collection[str]:
        assert False, "never happens"

    def has_vector(self, name: str) -> bool:
        return False

    def _has_vector(self, axis: str, name: str) -> bool:
        assert False, "never happens"

    def get_array1d(self, name: str) -> Array1D:
        assert False, "never happens"

    def _get_array1d(self, axis: str, name: str) -> Array1D:
        assert False, "never happens"

    def get_series(self, name: str) -> Series:
        assert False, "never happens"

    def matrix_names(self, axes: Tuple[str, str]) -> Collection[str]:
        return []

    def _matrix_names(self, axes: Tuple[str, str]) -> Collection[str]:
        assert False, "never happens"

    def has_matrix(self, name: str) -> bool:
        return False

    def _has_matrix(self, axes: Tuple[str, str], name: str) -> bool:
        assert False, "never happens"

    def get_grid(self, name: str) -> Grid:
        assert False, "never happens"

    def _get_grid(self, axes: Tuple[str, str], name: str) -> Grid:
        assert False, "never happens"

    def get_table(self, name: str) -> Table:
        assert False, "never happens"

    def update(self, storage: _interface.StorageReader, *, overwrite: bool = False) -> None:
        pass

    def set_datum(self, name: str, datum: Any, *, overwrite: bool = False) -> None:
        pass

    def _set_datum(self, name: str, datum: Any) -> None:
        assert False, "never happens"

    def create_axis(self, axis: str, entries: Array1D) -> None:
        pass

    def _create_axis(self, axis: str, entries: Array1D) -> None:
        assert False, "never happens"

    def set_vector(self, name: str, vector: Vector, *, overwrite: bool = False) -> None:
        pass

    def _set_array1d(self, axis: str, name: str, array1d: Array1D) -> None:
        assert False, "never happens"

    def set_matrix(self, name: str, matrix: Matrix, *, overwrite: bool = False) -> None:
        pass

    def _set_grid(self, axes: Tuple[str, str], name: str, grid: Grid) -> None:
        assert False, "never happens"

    @contextmanager
    def create_array2d(self, name: str, *, dtype: str, overwrite: bool = False) -> Generator[Array2D, None, None]:
        assert False, "never happens"

    @contextmanager
    def _create_array2d(self, shape: Tuple[int, int], name: str, dtype: str) -> Generator[Array2D, None, None]:
        assert False, "never happens"


#: Storage used to specify not caching sliced or derived data.
NO_STORAGE = NoStorage()
