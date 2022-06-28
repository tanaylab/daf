"""
Simple in-memory storage.

This just keeps everything in-memory, similarly to the way an ``AnnData`` object works; that is, this is a lightweight
object that just keeps references to the data it is given. Unlike ``AnnData`` it allows for efficient storage of
multiple axes.

This is the "default" storage type you should use unless you need something specific another storage format provides
(typically, disk storage).
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Dict
from typing import Optional
from typing import Tuple

from ..typing import Known1D
from ..typing import Known2D
from ..typing import MatrixInRows
from ..typing import Vector
from . import interface as _interface

# pylint: enable=duplicate-code,cyclic-import


__all__ = [
    "MemoryStorage",
]


class _MemoryReader(_interface.StorageReader):
    """
    Implement the `.StorageReader` interface for in-memory storage.
    """

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name)

        # The 0D data.
        self._items: Dict[str, Any] = {}

        # The entries for each known axis.
        self._axes: Dict[str, Vector] = {}

        # For each axis, the 1D data indexed by this axis.
        self._arrays: Dict[str, Dict[str, Vector]] = {}

        # For each pair of axis, the 2D `.is_optimal` `.ROW_MAJOR` `.MatrixInRows` indexed by that pair.
        self._matrices: Dict[Tuple[str, str], Dict[str, MatrixInRows]] = {}

    # pylint: disable=duplicate-code

    def _item_names(self) -> Collection[str]:
        return self._items.keys()

    def _has_item(self, name: str) -> bool:
        return name in self._items

    # pylint: enable=duplicate-code

    def _get_item(self, name: str) -> Any:
        return self._items[name]

    def _axis_names(self) -> Collection[str]:
        return self._axes.keys()

    # pylint: disable=duplicate-code

    def _has_axis(self, axis: str) -> bool:
        return axis in self._axes

    def _axis_size(self, axis: str) -> int:
        return len(self._axes[axis])

    def _axis_entries(self, axis: str) -> Known1D:
        return self._axes[axis]

    def _data1d_names(self, axis: str) -> Collection[str]:
        return self._arrays[axis].keys()

    def _has_data1d(self, axis: str, name: str) -> bool:
        return name in self._arrays[axis]

    def _get_data1d(self, axis: str, name: str) -> Known1D:
        return self._arrays[axis][name]

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:
        return self._matrices[axes].keys()

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:
        return name in self._matrices[axes]

    # pylint: enable=duplicate-code

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Known2D:
        return self._matrices[axes][name]


class MemoryStorage(_MemoryReader, _interface.StorageWriter):
    """
    Implement the `.StorageWriter` interface for in-memory storage.

    If the ``name`` ends with ``#``, we append the object id to it to make it unique.

    If ``copy`` is specified, it is copied into the directory, using the ``overwrite``.

    .. note::

        This just keeps a reference to the data it is given, so care must be taken not to mess it up after it has been
        put into the storage. It does `.freeze` it to prevent accidental modifications.
    """

    def __init__(
        self, *, name: str = "memory#", copy: Optional[_interface.StorageReader] = None, overwrite: bool = False
    ) -> None:
        if name.endswith("#"):
            name += str(id(self))
        # pylint: disable=duplicate-code
        super().__init__(name=name)
        if copy is not None:
            self.update(copy, overwrite=overwrite)
        # pylint: enable=duplicate-code

    def _set_item(self, name: str, item: Any) -> None:
        self._items[name] = item

    # pylint: disable=duplicate-code

    def _create_axis(self, axis: str, entries: Vector) -> None:
        self._arrays[axis] = {}
        for other_axis in self._axes:
            self._matrices[(axis, other_axis)] = {}
            self._matrices[(other_axis, axis)] = {}
        self._matrices[(axis, axis)] = {}
        self._axes[axis] = entries

    # pylint: enable=duplicate-code

    def _set_vector(self, axis: str, name: str, vector: Vector) -> None:
        self._arrays[axis][name] = vector

    def _set_matrix(self, axes: Tuple[str, str], name: str, matrix: MatrixInRows) -> None:
        self._matrices[axes][name] = matrix
