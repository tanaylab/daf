"""
The `.NO_STORAGE` is used to disable caching (e.g., caching sliced data in a ``TODOL-StorageView``).

Creating or setting data inside `.NO_STORAGE` has no effect. Unlike a real storage, it allows querying for names of axes
that don't exist (since in it, no axes exist). It also
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from typing import Collection
from typing import Generator
from typing import Tuple
from typing import Union

from ..typing import Array1D
from ..typing import ArrayInRows
from ..typing import Data2D
from ..typing import GridInRows
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

    def array1d_names(self, axis: str) -> Collection[str]:
        return []

    def _array1d_names(self, axis: str) -> Collection[str]:
        assert False, "never happens"

    def has_array1d(self, name: str) -> bool:
        return False

    def _has_array1d(self, axis: str, name: str) -> bool:
        assert False, "never happens"

    def get_array1d(self, name: str) -> Array1D:
        assert False, "never happens"

    def _get_array1d(self, axis: str, name: str) -> Array1D:
        assert False, "never happens"

    def data2d_names(self, axes: Union[str, Tuple[str, str]]) -> Collection[str]:
        return []

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:
        assert False, "never happens"

    def has_data2d(self, name: str) -> bool:
        return False

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:
        assert False, "never happens"

    def get_data2d(self, name: str) -> Data2D:
        assert False, "never happens"

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Data2D:
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

    def set_array1d(self, name: str, array1d: Array1D, *, overwrite: bool = False) -> None:
        pass

    def _set_array1d(self, axis: str, name: str, array1d: Array1D) -> None:
        assert False, "never happens"

    def set_grid(self, name: str, grid: GridInRows, *, overwrite: bool = False) -> None:
        pass

    def _set_grid(self, axes: Tuple[str, str], name: str, grid: GridInRows) -> None:
        assert False, "never happens"

    @contextmanager
    def create_array_in_rows(
        self, name: str, *, dtype: str, overwrite: bool = False
    ) -> Generator[ArrayInRows, None, None]:
        assert False, "never happens"

    @contextmanager
    def _create_array_in_rows(
        self, shape: Tuple[int, int], name: str, dtype: str
    ) -> Generator[ArrayInRows, None, None]:
        assert False, "never happens"


#: Storage used to specify not caching sliced or derived data.
NO_STORAGE = NoStorage()
