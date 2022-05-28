"""
Simple in-memory storage.

This just keeps everything in-memory, similarly to the way an ``AnnData`` object works; that is, this is a lightweight
object that just keeps references to the data it is given. Unlike ``AnnData`` it allows for efficient storage of
multiple axes.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Dict
from typing import Optional
from typing import Tuple

from ..typing import Array1D
from ..typing import Data2D
from ..typing import GridInRows
from . import interface as _interface

# pylint: enable=duplicate-code,cyclic-import


__all__ = [
    "MemoryStorage",
]


class MemoryReader(_interface.StorageReader):
    """
    Implement the `.StorageReader` interface for in-memory storage.
    """

    def __init__(self, *, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        # The 0D ("blob") data.
        self._data: Dict[str, Any] = {}

        # The entries for each known axis.
        self._axes: Dict[str, Array1D] = {}

        # For each axis, the 1D data indexed by this axis.
        self._arrays: Dict[str, Dict[str, Array1D]] = {}

        # For each pair of axis, the 2D `.is_optimal` `.ROW_MAJOR` `.GridInRows` indexed by that pair.
        self._grids: Dict[Tuple[str, str], Dict[str, GridInRows]] = {}

    def datum_names(self) -> Collection[str]:
        """
        See :`.StorageReader.datum_names`.
        """
        return self._data.keys()

    def has_datum(self, name: str) -> bool:
        """
        See `.StorageReader.has_datum`.
        """
        return name in self._data

    def _get_datum(self, name: str) -> Any:
        return self._data[name]

    def axis_names(self) -> Collection[str]:
        """
        See `.StorageReader.axis_names`.
        """
        return self._axes.keys()

    # pylint: disable=duplicate-code

    def has_axis(self, axis: str) -> bool:
        """
        See `.StorageReader.has_axis`.
        """
        return axis in self._axes

    def _axis_size(self, axis: str) -> int:
        return len(self._axes[axis])

    def _axis_entries(self, axis: str) -> Array1D:
        return self._axes[axis]

    # pylint: enable=duplicate-code

    def _array1d_names(self, axis: str) -> Collection[str]:
        return self._arrays[axis].keys()

    def _has_array1d(self, axis: str, name: str) -> bool:
        return name in self._arrays[axis]

    def _get_array1d(self, axis: str, name: str) -> Array1D:
        return self._arrays[axis][name]

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:
        return self._grids[axes].keys()

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:
        return name in self._grids[axes]

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Data2D:
        return self._grids[axes][name]


class MemoryStorage(MemoryReader, _interface.StorageWriter):
    """
    Adapter for simple read-write in-memory storage.

    .. note::

        This just keeps a reference to the data it is given, so care must be taken not to mess it up after it has been
        put into the storage. It does `.freeze` it to prevent accidental modifications.
    """

    def _set_datum(self, name: str, datum: Any) -> None:
        self._data[name] = datum

    def _create_axis(self, axis: str, entries: Array1D) -> None:
        self._arrays[axis] = {}
        for other_axis in self._axes:
            self._grids[(axis, other_axis)] = {}
            self._grids[(other_axis, axis)] = {}
        self._grids[(axis, axis)] = {}
        self._axes[axis] = entries

    def _set_array1d(self, axis: str, name: str, array1d: Array1D) -> None:
        self._arrays[axis][name] = array1d

    def _set_grid(self, axes: Tuple[str, str], name: str, grid: GridInRows) -> None:
        self._grids[axes][name] = grid
