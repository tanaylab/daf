"""
A chain of storage objects (first one wins).

This serves two purposes. First, it is only a `.StorageReader`, so it protects the wrapped storage objects against
accidental modification, even if they implement `.StorageWriter` as well. Second, it allows delta-encoding, where data
in early storage objects enhances or even overrides the data in later objects. For example, this is used in `.DafReader`
to present a unified view of the data in the derived cache and the base storage.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Sequence
from typing import Set
from typing import Tuple

import numpy as np

from ..typing import Known1D
from ..typing import Known2D
from ..typing import Vector
from ..typing import as_vector
from . import interface as _interface
from . import none as _none

# pylint: enable=duplicate-code,cyclic-import

__all__ = ["StorageChain"]


# pylint: disable=protected-access


class StorageChain(_interface.StorageReader):
    """
    Low-level read-only access to a ``chain`` of storage objects (first one wins).

    If the ``name`` ends with ``#``, we append the object id to it to make it unique.

    This is different from other storage implementations in that modifications to the wrapped storage objects are
    guaranteed to be immediately visible in the chain (in contrast to, for example, `.StorageView` where this is **not**
    the case). The implementation of `.DafWriter` relies on this fact.
    """

    def __init__(self, chain: Sequence[_interface.StorageReader], *, name: str = "chain#") -> None:
        if name.endswith("#"):
            name += str(id(self))
        super().__init__(name=name)

        unique_readers: List[_interface.StorageReader] = []
        unique_ids: Set[int] = set()

        for reader in chain:
            StorageChain._add_reader(unique_readers, unique_ids, reader)

        #: The unique chained `.StorageReader` objects (first one wins).
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
        axes_entries: Dict[str, Tuple[str, Vector]] = {}
        for storage in self.chain:
            new_name = storage.name
            for axis in storage._axis_names():
                new_entries = as_vector(storage._axis_entries(axis))
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

    def _self_description(self, self_description: Dict, *, detail: bool) -> None:
        self_description["chain"] = [storage.name for storage in self.chain]

    def _deep_description(self, description: Dict, self_description: Dict, *, detail: bool) -> None:
        for storage in self.chain:
            storage.description(description=description, detail=detail)

    def _item_names(self) -> Collection[str]:
        names: Set[str] = set()
        for storage in self.chain:
            names.update(storage._item_names())
        return names

    def _has_item(self, name: str) -> bool:
        for storage in self.chain:
            if storage._has_item(name):
                return True
        return False

    def _get_item(self, name: str) -> Any:
        for storage in self.chain:
            if storage._has_item(name):
                return storage._get_item(name)
        assert False, "never happens"

    def _axis_names(self) -> Collection[str]:
        names: Set[str] = set()
        for storage in self.chain:
            names.update(storage._axis_names())
        return names

    def _has_axis(self, axis: str) -> bool:
        for storage in self.chain:
            if storage._has_axis(axis):
                return True
        return False

    def _axis_size(self, axis: str) -> int:
        for storage in self.chain:
            if storage._has_axis(axis):
                return storage._axis_size(axis)
        assert False, "never happens"

    def _axis_entries(self, axis: str) -> Known1D:
        for storage in self.chain:
            if storage._has_axis(axis):
                return storage._axis_entries(axis)
        assert False, "never happens"

    def _data1d_names(self, axis: str) -> Collection[str]:
        names: Set[str] = set()
        for storage in self.chain:
            if storage._has_axis(axis):
                names.update(storage._data1d_names(axis))
        return names

    def _has_data1d(self, axis: str, name: str) -> bool:
        for storage in self.chain:
            if storage._has_axis(axis) and storage._has_data1d(axis, name):
                return True
        return False

    def _get_data1d(self, axis: str, name: str) -> Known1D:
        for storage in self.chain:
            if storage._has_axis(axis) and storage.has_data1d(name):
                return storage._get_data1d(axis, name)
        assert False, "never happens"

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:
        names: Set[str] = set()
        for storage in self.chain:
            if storage._has_axis(axes[0]) and storage.has_axis(axes[1]):
                names.update(storage._data2d_names(axes))
        return names

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:
        for storage in self.chain:
            if storage._has_axis(axes[0]) and storage._has_axis(axes[1]) and storage._has_data2d(axes, name):
                return True
        return False

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Known2D:
        for storage in self.chain:
            if storage._has_axis(axes[0]) and storage._has_axis(axes[1]) and storage._has_data2d(axes, name):
                return storage._get_data2d(axes, name)
        assert False, "never happens"
