"""
A chain of storage objects (first one wins).

This serves two purposes. First, it is only a `.StorageReader`, so it protects the wrapped storage objects against
accidental modification, even if they implement `.StorageWriter` as well. Second, it allows delta-encoding, where data
in early storage objects enhances or even overrides the data in later objects.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple

import numpy as np

from ..typing import Array1D
from ..typing import Grid
from . import interface as _interface
from . import none as _none

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "StorageChain",
]


# pylint: disable=protected-access


class StorageChain(_interface.StorageReader):
    """
    Low-level read-only access to a ``chain`` of storage objects (first one wins).
    """

    def __init__(self, chain: Sequence[_interface.StorageReader], *, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        unique_readers: List[_interface.StorageReader] = []
        unique_ids: Set[int] = set()

        for reader in chain:
            StorageChain._add_reader(unique_readers, unique_ids, reader)

        self.chain: Tuple[_interface.StorageReader, ...] = tuple(unique_readers)

        self._verify()

    @staticmethod
    def _add_reader(
        unique_readers: List[_interface.StorageReader], unique_ids: Set[int], reader: _interface.StorageReader
    ) -> None:
        if isinstance(reader, _none.NoStorage):
            return

        if isinstance(reader, StorageChain):
            for chained_reader in reader.chain:
                StorageChain._add_reader(unique_readers, unique_ids, chained_reader)
            return

        reader_id = id(reader)
        if reader_id in unique_ids:
            return

        unique_readers.append(reader)
        unique_ids.add(reader_id)

    def _verify(self) -> None:
        axes_entries: Dict[str, Tuple[str, Array1D]] = {}
        for storage in self.chain:
            new_name = storage.name
            for axis in storage.axis_names():
                new_entries = storage.axis_entries(axis)
                old_data = axes_entries.get(axis)
                if old_data is None:
                    axes_entries[axis] = (new_name, new_entries)
                else:
                    old_name, old_entries = old_data
                    assert np.all(old_entries == new_entries), (
                        f"inconsistent entries for the axis: {axis} "
                        f"between the storage: {old_name} "
                        f"and the storage: {new_name} "
                        f"in the storage chain: {self.name}"
                    )

    def datum_names(self) -> Collection[str]:
        names: Set[str] = set()
        for storage in self.chain:
            names.update(storage.datum_names())
        return names

    def has_datum(self, name: str) -> bool:
        for storage in self.chain:
            if storage.has_datum(name):
                return True
        return False

    def _get_datum(self, name: str) -> Any:
        for storage in self.chain:
            if storage.has_datum(name):
                return storage._get_datum(name)
        assert False, "never happens"

    def axis_names(self) -> Collection[str]:
        names: Set[str] = set()
        for storage in self.chain:
            names.update(storage.axis_names())
        return names

    def has_axis(self, axis: str) -> bool:
        for storage in self.chain:
            if storage.has_axis(axis):
                return True
        return False

    def _axis_size(self, axis: str) -> int:
        for storage in self.chain:
            if storage.has_axis(axis):
                return storage._axis_size(axis)
        assert False, "never happens"

    def _axis_entries(self, axis: str) -> Array1D:
        for storage in self.chain:
            if storage.has_axis(axis):
                return storage._axis_entries(axis)
        assert False, "never happens"

    def _vector_names(self, axis: str) -> Collection[str]:
        names: Set[str] = set()
        for storage in self.chain:
            if storage.has_axis(axis):
                names.update(storage._vector_names(axis))
        return names

    def _has_vector(self, axis: str, name: str) -> bool:
        for storage in self.chain:
            if storage.has_axis(axis) and storage._has_vector(axis, name):
                return True
        return False

    def _get_array1d(self, axis: str, name: str) -> Array1D:
        for storage in self.chain:
            if storage.has_axis(axis) and storage.has_vector(name):
                return storage._get_array1d(axis, name)
        assert False, "never happens"

    def _matrix_names(self, axes: Tuple[str, str]) -> Collection[str]:
        names: Set[str] = set()
        for storage in self.chain:
            if storage.has_axis(axes[0]) and storage.has_axis(axes[1]):
                names.update(storage._matrix_names(axes))
        return names

    def _has_matrix(self, axes: Tuple[str, str], name: str) -> bool:
        for storage in self.chain:
            if storage.has_axis(axes[0]) and storage.has_axis(axes[1]) and storage._has_matrix(axes, name):
                return True
        return False

    def _get_grid(self, axes: Tuple[str, str], name: str) -> Grid:
        for storage in self.chain:
            if storage.has_axis(axes[0]) and storage.has_axis(axes[1]) and storage._has_matrix(axes, name):
                return storage._get_grid(axes, name)
        assert False, "never happens"
