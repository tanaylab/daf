"""
Storage views allow slicing the data, and/or renaming and/or hiding specific axes or data.

A view is just a light-weight read-only adapter of some underlying storage; a common idiom (e.g. for exporting a subset
of the data) is to create a view, then copy its contents into an empty new persistent storage (such as `.FilesWriter`)
to save just this data to the disk. This is crucial when converting ``daf`` data to ``AnnData``, as ``AnnData`` requires
specific axes names, and is not capable of dealing with too many axes.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd  # type: ignore

from ..typing import ENTRIES_DTYPES
from ..typing import ROW_MAJOR
from ..typing import STR_DTYPE
from ..typing import Array1D
from ..typing import Data2D
from ..typing import Grid
from ..typing import as_array1d
from ..typing import as_grid
from ..typing import as_layout
from ..typing import assert_data
from ..typing import freeze
from ..typing import is_array1d
from ..typing import is_dtype
from ..typing import optimize
from . import interface as _interface
from . import memory as _memory

# pylint: enable=duplicate-code

__all__ = ["DataView", "AxisView", "StorageView"]

#: Describe how to expose an axis of some wrapped `.StorageReader` from a `.StorageView`:
#:
#: * By default, axes are passed as-is. But this can be changed by setting ``hide_implicit`` to ``True``, in which
#:   case, by default axes are hidden.
#:
#: * If the ``AxisView`` is ``None``, the axis is explicitly hidden, together with all data that is based on it.
#:
#: * If the ``AxisView`` is a string, the axis is exposed under this new name. For example, when viewing data for
#:   copying into ``AnnData``, one axis needs to be exposed as ``obs`` and another as ``vars``.
#:
#: * If the ``AxisView`` is an `.Array1D`, then the axis is sliced:
#:
#:   * An array of strings contains the names of the entries to expose.
#:   * An array of integers contains the indices of the entries to expose.
#:   * An array of Booleans contains a mask of the entries to expose.
#:
#:   .. note::
#:
#:      The order of the string or integer entries is significant as it controls the order of the entries in the exposed
#:      axis. Slicing using a Boolean mask keeps the original entries order.
#:
#: * If the ``AxisView`` is a tuple of a string and an array, the axis is both renamed and sliced as above.
AxisView = Union[None, str, Array1D, Tuple[str, Array1D]]

#: Describe how to expose data of some wrapped `.StorageReader` from a `.StorageView`:
#:
#:  * By default, all data is passed as-is, as long as the axes it is based on (if any) are exposed. If the axes were
#:    renamed, the data is exposed using the new axes names. Setting ``hide_implicit`` to ``True`` changes the
#:    default to hide data by default.
#:
#:  * If the ``DataView`` is ``None``, the data is explicitly hidden, even if the axes it is based on are exposed.
#:
#:  * If the ``DataView`` is a string, it is exposed under this new name (which should not contain the axes names(s)).
#:
#: .. note::
#:
#:    For 2D data, specify only one of the two ``foo,bar;baz`` and ``bar,foo;baz`` names, and it will automatically also
#:    apply to its transpose.
DataView = Optional[str]


class StorageView(_interface.StorageReader):  # pylint: disable=too-many-instance-attributes
    """
    A read-only view of some other `.StorageReader`.

    A view is defined by describing how to expose each *existing* axis or data, that is, the keys to the ``axes`` and
    ``data`` dictionaries are the names in the wrapped storage, **not** the exposed names of the view.

    By default, sliced data is stored in a ``cache`` using `.MemoryStorage`. To disable caching, specify a `.NO_STORAGE`
    value for it.

    .. note::

        Do **not** modify the wrapped storage after creating a view. Modifications may or may not be visible in the
        view, causing subtle problems.
    """

    def __init__(
        self,
        storage: _interface.StorageReader,
        *,
        axes: Optional[Dict[str, AxisView]] = None,
        data: Optional[Dict[str, DataView]] = None,
        name: Optional[str] = None,
        cache: Optional[_interface.StorageWriter] = None,
        hide_implicit: bool = False,
    ) -> None:
        super().__init__(name=name)

        #: The wrapped storage.
        self.storage = storage

        #: Cached storage for sliced data.
        self.cache = cache or _memory.MemoryStorage(name=f"{self.name}.cache")

        # How to expose each of the axes of the wrapped storage.
        self._wrapped_axis_views: Dict[str, Optional[str]] = {}

        # How to expose each of the data of the wrapped storage.
        self._wrapped_data_views: Dict[str, Optional[str]] = {}

        # How to expose each of the vectors of the wrapped storage.
        self._wrapped_array1d_views: Dict[str, Optional[str]] = {}

        #: How to expose each of the 2D data of the wrapped storage.
        self._wrapped_data2d_views: Dict[str, Optional[str]] = {}

        # Map each exposed axis back to a wrapped storage axis.
        self._exposed_axes: Dict[str, Tuple[str, Optional[Array1D]]] = {}

        # Map each exposed datum back to a wrapped storage datum.
        self._exposed_data: Dict[str, str] = {}

        # Map each exposed vector back to a wrapped storage vector.
        self._exposed_array1d: Dict[str, Dict[str, str]] = {}

        # Map each exposed 2D data back to a wrapped storage vector.
        self._exposed_data2d: Dict[Tuple[str, str], Dict[str, str]] = {}

        axes = axes or {}
        self._verify_axes(axes)
        self._collect_axes(axes, hide_implicit)
        self._collect_cache_axes()

        self._collect_views(data or {})

        self._collect_data(hide_implicit)

        self._init_array1d()
        self._collect_array1d(hide_implicit)

        self._init_data2d()
        self._collect_data2d(hide_implicit)

    def _verify_axes(self, axes: Dict[str, AxisView]) -> None:
        for axis in axes:
            assert self.storage.has_axis(
                axis
            ), f"missing the axis: {axis} in the storage: {self.storage.name} for the view: {self.name}"

    def _collect_axes(self, axes: Dict[str, AxisView], hide_implicit: bool) -> None:
        for wrapped_axis in self.storage.axis_names():
            entries: Optional[Array1D] = None
            axis_view = axes.get(wrapped_axis, None if hide_implicit else wrapped_axis)
            if axis_view is None:
                exposed_axis = None
            elif isinstance(axis_view, str):
                exposed_axis = axis_view
            elif isinstance(axis_view, tuple):
                exposed_axis, entries = axis_view
                assert isinstance(exposed_axis, str), (
                    f"invalid view for the axis: {wrapped_axis} "
                    f"of the storage: {self.storage.name} "
                    f"for the view: {self.name}"
                )
            else:
                exposed_axis = wrapped_axis
                entries = axis_view

            if entries is not None:
                assert_data(is_array1d(entries, dtype=ENTRIES_DTYPES), "1D numpy array", entries, None)

                if is_dtype(entries.dtype, "bool"):
                    entries = np.where(entries)[0]
                    assert entries is not None
                elif is_dtype(entries.dtype, STR_DTYPE):
                    entries = as_array1d(
                        pd.Series(
                            np.arange(self.storage.axis_size(wrapped_axis)),
                            index=self.storage.axis_entries(wrapped_axis),
                        )[entries]
                    )
                else:
                    entries = entries.copy()

            self._wrapped_axis_views[wrapped_axis] = exposed_axis
            if exposed_axis is not None:
                assert exposed_axis not in self._exposed_axes, (
                    f"both the axis: {self._exposed_axes[exposed_axis][0]} "
                    f"and the axis: {wrapped_axis} "
                    f"of the storage: {self.storage.name} "
                    f"are exposed as the same axis: {exposed_axis} "
                    f"of the view: {self.name}"
                )
                self._exposed_axes[exposed_axis] = (wrapped_axis, entries)

    def _collect_cache_axes(self) -> None:
        for axis, (wrapped_name, wrapped_indices) in self._exposed_axes.items():
            entries = self.storage.axis_entries(wrapped_name)
            if wrapped_indices is not None:
                entries = entries[wrapped_indices]
            self.cache.create_axis(axis, freeze(entries))

    def _collect_views(self, data: Dict[str, DataView]) -> None:
        for wrapped_data, data_view in data.items():
            assert self._collect_view(
                wrapped_data, data_view
            ), f"missing the data: {wrapped_data} in the storage: {self.storage.name} for the view: {self.name}"

    def _collect_view(self, wrapped_data: str, data_view: DataView) -> bool:
        if ";" not in wrapped_data:
            self._wrapped_data_views[wrapped_data] = data_view
            return self.storage.has_datum(wrapped_data)

        axes, name = wrapped_data.split(";")

        if "," not in axes:
            self._wrapped_array1d_views[wrapped_data] = data_view
            return self.storage.has_array1d(wrapped_data)

        self._wrapped_data2d_views[wrapped_data] = data_view
        rows_axis, columns_axis = axes.split(",")
        transposed_data = f"{columns_axis},{rows_axis};{name}"
        if transposed_data not in self._wrapped_data2d_views:
            self._wrapped_data2d_views[transposed_data] = data_view
        else:
            transposed_view = self._wrapped_data2d_views[transposed_data]
            assert transposed_view == data_view, (
                f"conflicting 2D data views: {wrapped_data} => {data_view} "
                f"and: {transposed_data} => {transposed_view} "
                f"for the storage: {self.storage.name} "
                f"for the view: {self.name}"
            )
        return self.storage.has_data2d(wrapped_data)

    def _collect_data(self, hide_implicit: bool) -> None:
        for wrapped_data in self.storage.datum_names():
            exposed_data = self._wrapped_data_views.get(wrapped_data, None if hide_implicit else wrapped_data)
            self._wrapped_data_views[wrapped_data] = exposed_data
            if exposed_data is not None:
                assert exposed_data not in self._exposed_data, (
                    f"both the datum: {self._exposed_data[exposed_data]} "
                    f"and the datum: {wrapped_data} "
                    f"of the storage: {self.storage.name} "
                    f"are exposed as the same datum: {exposed_data} "
                    f"of the view: {self.name}"
                )
                self._exposed_data[exposed_data] = wrapped_data

    def _init_array1d(self) -> None:
        for exposed_axis in self._exposed_axes:
            self._exposed_array1d[exposed_axis] = {}

    def _collect_array1d(self, hide_implicit: bool) -> None:
        for wrapped_axis in self.storage.axis_names():
            exposed_axis = self.exposed_axis(wrapped_axis)
            if exposed_axis is None:
                continue

            _exposed_array1d = self._exposed_array1d[exposed_axis]

            for wrapped_data in self.storage.array1d_names(wrapped_axis):
                exposed_data: Optional[str]

                name = wrapped_data.split(";")[1]
                data_view = self._wrapped_array1d_views.get(wrapped_data, None if hide_implicit else name)
                if data_view is None:
                    exposed_data = None
                else:
                    exposed_data = f"{exposed_axis};{data_view}"

                self._wrapped_array1d_views[wrapped_data] = exposed_data
                if exposed_data is not None:
                    assert exposed_data not in _exposed_array1d, (
                        f"both the 1D data: {_exposed_array1d[exposed_data]} "
                        f"and the 1D data: {wrapped_data} "
                        f"of the storage: {self.storage.name} "
                        f"are exposed as the same 1D data: {exposed_data} "
                        f"of the view: {self.name}"
                    )
                    _exposed_array1d[exposed_data] = wrapped_data

    def _init_data2d(self) -> None:
        for exposed_rows_axis in self._exposed_axes:
            for exposed_columns_axis in self._exposed_axes:
                self._exposed_data2d[(exposed_rows_axis, exposed_columns_axis)] = {}

    def _collect_data2d(self, hide_implicit: bool) -> None:
        for wrapped_rows_axis in self.storage.axis_names():
            exposed_rows_axis = self.exposed_axis(wrapped_rows_axis)
            if exposed_rows_axis is None:
                continue

            for wrapped_columns_axis in self.storage.axis_names():
                exposed_columns_axis = self.exposed_axis(wrapped_columns_axis)
                if exposed_columns_axis is None:
                    continue

                exposed_data2d = self._exposed_data2d[(exposed_rows_axis, exposed_columns_axis)]

                for wrapped_data in self.storage.data2d_names((wrapped_rows_axis, wrapped_columns_axis)):
                    exposed_data: Optional[str]

                    name = wrapped_data.split(";")[1]
                    data_view = self._wrapped_data2d_views.get(wrapped_data, None if hide_implicit else name)
                    if data_view is None:
                        exposed_data = None
                    else:
                        exposed_data = f"{exposed_rows_axis},{exposed_columns_axis};{data_view}"

                    self._wrapped_data2d_views[wrapped_data] = exposed_data
                    if exposed_data is not None:
                        assert exposed_data not in exposed_data2d, (
                            f"both the 2D data: {self._exposed_data[exposed_data]} "
                            f"and the 2D data: {wrapped_data} "
                            f"of the storage: {self.storage.name} "
                            f"are exposed as the same 2D data: {exposed_data} "
                            f"of the view: {self.name}"
                        )
                        exposed_data2d[exposed_data] = wrapped_data

    def exposed_axis(self, wrapped_axis: str) -> Optional[str]:
        """
        Given the name of an ``wrapped_axis`` in the wrapped `.StorageReader`, return the
        name it is exposed as in the view, or ``None`` if it is hidden.
        """
        assert self.storage.has_axis(wrapped_axis), f"missing axis: {wrapped_axis} in the storage: {self.storage.name}"
        return self._wrapped_axis_views[wrapped_axis]

    def exposed_datum(self, wrapped_datum: str) -> Optional[str]:
        """
        Given the name of an ``wrapped_datum`` in the wrapped `.StorageReader`, return the name it is exposed as in the
        view, or ``None`` if it is hidden.
        """
        assert self.storage.has_datum(
            wrapped_datum
        ), f"missing datum: {wrapped_datum} in the storage: {self.storage.name}"
        return self._wrapped_data_views[wrapped_datum]

    def exposed_array1d(self, wrapped_array1d: str) -> Optional[str]:
        """
        Given the name of an ``wrapped_array1d`` in the wrapped `.StorageReader`, return the name it is exposed as in
        the view, or ``None`` if it is hidden.
        """
        assert self.storage.has_array1d(
            wrapped_array1d
        ), f"missing 1D data: {wrapped_array1d} in the storage: {self.storage.name}"
        return self._wrapped_array1d_views[wrapped_array1d]

    def exposed_data2d(self, wrapped_data2d: str) -> Optional[str]:
        """
        Given the name of an ``wrapped_data2d`` in the wrapped `.StorageReader`, return the name it is exposed as in the
        view, or ``None`` if it is hidden.
        """
        assert self.storage.has_data2d(
            wrapped_data2d
        ), f"missing 2D data: {wrapped_data2d} in the storage: {self.storage.name}"
        return self._wrapped_data2d_views[wrapped_data2d]

    def wrapped_axis(self, exposed_axis: str) -> str:
        """
        Given the name of an ``exposed_axis`` (which must exist), return its name in the wrapped `.StorageReader`.
        """
        assert self.has_axis(exposed_axis), f"missing axis: {exposed_axis} in the storage: {self.name}"
        return self._exposed_axes[exposed_axis][0]

    def wrapped_datum(self, exposed_datum: str) -> str:
        """
        Given the name of an ``exposed_datum`` (which must exist), return its name in the wrapped `.StorageReader`.
        """
        assert self.has_datum(exposed_datum), f"missing datum: {exposed_datum} in the storage: {self.name}"
        return self._exposed_data[exposed_datum]

    def wrapped_array1d(self, exposed_array1d: str) -> str:
        """
        Given the name of an ``exposed_array1d`` (which must exist), return its name in the wrapped `.StorageReader`.
        """
        assert self.has_array1d(exposed_array1d), f"missing 1D data: {exposed_array1d} in the storage: {self.name}"
        axis = _interface.extract_1d_axis(exposed_array1d)
        return self._exposed_array1d[axis][exposed_array1d]

    def wrapped_data2d(self, exposed_data2d: str) -> str:
        """
        Given the name of an ``exposed_data2d`` (which must exist), return its name in the wrapped `.StorageReader`.
        """
        assert self.has_data2d(exposed_data2d), f"missing 2D data: {exposed_data2d} in the storage: {self.name}"
        axes = _interface.extract_2d_axes(exposed_data2d)
        return self._exposed_data2d[axes][exposed_data2d]

    def _datum_names(self) -> Collection[str]:
        return self._exposed_data.keys()

    def _has_datum(self, name: str) -> bool:
        return name in self._exposed_data

    def _get_datum(self, name: str) -> Any:
        return self.storage.get_datum(self._exposed_data[name])

    def _axis_names(self) -> Collection[str]:
        return self._exposed_axes.keys()

    def _has_axis(self, axis: str) -> bool:
        return axis in self._exposed_axes

    def _axis_size(self, axis: str) -> int:
        return self.cache.axis_size(axis)

    def _axis_entries(self, axis: str) -> Array1D:
        return self.cache.axis_entries(axis)

    def _array1d_names(self, axis: str) -> Collection[str]:
        return self._exposed_array1d[axis].keys()

    def _has_array1d(self, axis: str, name: str) -> bool:
        return name in self._exposed_array1d[axis]

    def _get_array1d(self, axis: str, name: str) -> Array1D:
        if self.cache.has_array1d(name):
            return self.cache.get_array1d(name)

        array1d = self.storage.get_array1d(self._exposed_array1d[axis][name])

        wrapped_entries = self._exposed_axes[axis][1]
        if wrapped_entries is None:
            return array1d

        array1d = array1d[wrapped_entries]
        self.cache.set_array1d(name, freeze(array1d))
        return array1d

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:
        return self._exposed_data2d[axes].keys()

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:
        return name in self._exposed_data2d[axes].keys()

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Data2D:
        if self.cache.has_data2d(name):
            return self.cache.get_data2d(name)

        data2d = self.storage.get_data2d(self._exposed_data2d[axes][name])
        grid: Optional[Grid] = None

        wrapped_column_entries = self._exposed_axes[axes[1]][1]
        if wrapped_column_entries is not None:
            grid = as_grid(data2d)
            grid = grid[:, wrapped_column_entries]

        wrapped_row_entries = self._exposed_axes[axes[0]][1]
        if wrapped_row_entries is not None:
            grid = grid or as_grid(data2d)
            grid = grid[wrapped_row_entries, :]

        if grid is None:
            return data2d

        grid = optimize(as_layout(grid, ROW_MAJOR))
        self.cache.set_grid(name, freeze(grid))
        return grid
