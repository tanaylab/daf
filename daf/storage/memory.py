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
from ..typing import Grid
from ..typing import as_grid
from . import interface as _interface

# pylint: enable=duplicate-code,cyclic-import


__all__ = [
    "MemoryStorage",
]


class MemoryReader(_interface.StorageReader):
    """
    Implement the :py:obj:`~daf.storage.interface.StorageReader` interface for in-memory storage.

    We restrict the stored data to be in plain format (that is, :py:obj:`~daf.typing.array1d.Array1D` and
    :py:obj:`~daf.typing.grid.Grid` data), storing the axes indices separately as
    :py:obj:`~daf.typing.array1d.Array1D` of strings and only constructing ``pandas.Series`` and/or
    ``pandas.DataFrame`` on demand, which is acceptable as these are just lightweight objects referring to the actual
    data arrays.
    """

    def __init__(self, *, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        #: The 0D ("blob") data.
        self.data: Dict[str, Any] = {}

        #: The entries for each known axis.
        self.axes: Dict[str, Array1D] = {}

        #: For each axis, the 1D data indexed by this axis.
        self.arrays: Dict[str, Dict[str, Array1D]] = {}

        #: For each pair of axis, the 2D row-major data indexed by that pair.
        self.grids: Dict[Tuple[str, str], Dict[str, Grid]] = {}

    def datum_names(self) -> Collection[str]:
        """
        See :py:obj:`~daf.storage.interface.StorageReader.datum_names`.
        """
        return self.data.keys()

    def has_datum(self, name: str) -> bool:
        """
        See :py:obj:`~daf.storage.interface.StorageReader.has_datum`.
        """
        return name in self.data

    def _get_datum(self, name: str) -> Any:
        return self.data[name]

    def axis_names(self) -> Collection[str]:
        """
        See :py:obj:`~daf.storage.interface.StorageReader.axis_names`.
        """
        return self.axes.keys()

    # pylint: disable=duplicate-code

    def has_axis(self, axis: str) -> bool:
        """
        See :py:obj:`~daf.storage.interface.StorageReader.has_axis`.
        """
        return axis in self.axes

    def _axis_size(self, axis: str) -> int:
        return len(self.axes[axis])

    def _axis_entries(self, axis: str) -> Array1D:
        return self.axes[axis]

    # pylint: enable=duplicate-code

    def _vector_names(self, axis: str) -> Collection[str]:
        return self.arrays[axis].keys()

    def _has_vector(self, axis: str, name: str) -> bool:
        return name in self.arrays[axis]

    def _get_array1d(self, axis: str, name: str) -> Array1D:
        return self.arrays[axis][name]

    def _matrix_names(self, axes: Tuple[str, str]) -> Collection[str]:
        return self.grids[axes].keys()

    def _has_matrix(self, axes: Tuple[str, str], name: str) -> bool:
        return name in self.grids[axes]

    def _get_grid(self, axes: Tuple[str, str], name: str) -> Grid:
        return as_grid(self.grids[axes][name])


class MemoryStorage(MemoryReader, _interface.StorageWriter):
    """
    Adapter for simple read-write in-memory storage.

    .. note::

        This just keeps a reference to the data it is given, so care must be taken not to mess it up after it has been
        put into the storage. It does :py:obj:`~daf.typing.freeze` it to prevent accidental modifications.
    """

    def _set_datum(self, name: str, datum: Any) -> None:
        self.data[name] = datum

    def _create_axis(self, axis: str, entries: Array1D) -> None:
        self.arrays[axis] = {}
        for other_axis in self.axes:
            self.grids[(axis, other_axis)] = {}
            self.grids[(other_axis, axis)] = {}
        self.grids[(axis, axis)] = {}
        self.axes[axis] = entries

    def _set_array1d(self, axis: str, name: str, array1d: Array1D) -> None:
        self.arrays[axis][name] = array1d

    def _set_grid(self, axes: Tuple[str, str], name: str, grid: Grid) -> None:
        self.grids[axes][name] = grid
