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
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd  # type: ignore

from ..typing import ENTRIES_DTYPES
from ..typing import ROW_MAJOR
from ..typing import STR_DTYPE
from ..typing import AnyData
from ..typing import Known1D
from ..typing import Known2D
from ..typing import Matrix
from ..typing import Vector
from ..typing import as_layout
from ..typing import as_matrix
from ..typing import as_vector
from ..typing import assert_data
from ..typing import data_description
from ..typing import freeze
from ..typing import has_dtype
from ..typing import optimize
from . import interface as _interface
from . import memory as _memory

# pylint: enable=duplicate-code

# pylint: disable=protected-access

__all__ = ["AxisView", "StorageView"]


class AxisView(NamedTuple):
    """
    Describe how to expose an axis of some base `.StorageReader` from a `.StorageView`:
    """

    #: The name to expose the axis as, or ``None`` to keep the same name.
    name: Optional[str] = None

    #: Which entries of the axis to expose (how to slice the axis).
    #:
    #: * If ``None``, expose all the entries (no slicing).
    #: * A vector of strings contains the names of the entries to expose.
    #: * A vector of integers contains the indices of the entries to expose.
    #: * A vector of Booleans contains a mask of the entries to expose.
    entries: Optional[AnyData] = None

    #: Whether to create new data for the axis which tracks the index of each entry in the base data. If ``None``, no
    #: such data is created. Otherwise, ``axis;name`` will be created and will contain the integer index of each exposed
    #: axis entry in the original base data.
    track: Optional[str] = None

    #: If creating new data to track the axis entry indices, whether to overwrite existing data of the same name.
    overwrite: bool = False


class AxisFullView(NamedTuple):
    """
    Internal full representation of how an axis is exposed.
    """

    #: The exposed name (which may be the same as the base axis name).
    exposed_name: str

    #: The base name (which may be the same as the exposed axis name).
    base_name: str

    #: If slicing, the indices of the exposed entries.
    entry_indices: Optional[Vector]

    #: If not ``None``, the simple name of the data exposing the axis entry indices.
    track: Optional[str]

    #: If ``track`` is not ``None``, whether to overwrite existing data of the same name.
    overwrite: bool


class StorageView(_interface.StorageReader):  # pylint: disable=too-many-instance-attributes
    """
    A read-only view of some ``base`` `.StorageReader`.

    If the ``name`` starts with ``.``, it is appended to the base name. If it ends with ``#`` we append the object id to
    it to make it unique.

    A view is defined by describing how to expose each *existing* axis or data, that is, the keys to the ``axes`` and
    ``data`` dictionaries are the names in the base storage, **not** the exposed names of the storage view.

    For 2D data, specify only one of the two ``foo,bar;baz`` and ``bar,foo;baz`` names, and it will automatically also
    apply to its transpose.

    If the explicit of some axis or data is ``None``, than that axis or data are hidden. If an axis or some data is not
    listed, then by default it is exposed as is; if ``hide_implicit``, it is hidden. Hiding an axis hides all the data
    based on that axis.

    If the value of some axis or data is a string, it is the simple name to use to expose the data or axis.

    If the value of an axis is the entries of the axis to expose, it will be sliced (using the same name).

    If the value of an axis is an `.AxisView`, it describes in detail how to expose the axis.

    By default, sliced data is stored in a ``cache`` using `.MemoryStorage`. To disable caching, specify a `.NO_STORAGE`
    value for it.

    .. note::

        Do **not** modify the base storage after creating a view. Modifications may or may not be visible in the view,
        causing subtle problems.
    """

    def __init__(
        self,
        base: _interface.StorageReader,
        *,
        axes: Optional[Mapping[str, Union[None, str, AnyData, AxisView]]] = None,
        data: Optional[Mapping[str, Optional[str]]] = None,
        name: str = ".view#",
        cache: Optional[_interface.StorageWriter] = None,
        hide_implicit: bool = False,
    ) -> None:
        if name.endswith("#"):
            name += str(id(self))
        if name.startswith("."):
            name = base.name + name

        super().__init__(name=name)

        #: The base storage.
        self.base = base

        #: Cached storage for sliced data.
        self.cache = cache or _memory.MemoryStorage(name=self.name + ".cache")

        # How to expose each of the axes of the base storage.
        self._base_axis_views: Dict[str, Optional[AxisFullView]] = {}

        # How to expose each of the items of the base storage.
        self._base_item_views: Dict[str, Optional[str]] = {}

        # How to expose each of the vectors of the base storage.
        self._base_data1d_views: Dict[str, Optional[str]] = {}

        #: How to expose each of the 2D data of the base storage.
        self._base_data2d_views: Dict[str, Optional[str]] = {}

        # Map each exposed axis back to a base storage axis.
        self._exposed_axis_views: Dict[str, AxisFullView] = {}

        # Map each exposed item back to a base storage item.
        self._exposed_items: Dict[str, str] = {}

        # Map each exposed vector back to a base storage vector.
        self._exposed_data1d: Dict[str, Dict[str, Union[str, Vector]]] = {}

        # Map each exposed 2D data back to a base storage vector.
        self._exposed_data2d: Dict[Tuple[str, str], Dict[str, str]] = {}

        axes = axes or {}
        self._verify_axes(axes)
        self._collect_axes(axes, hide_implicit)
        self._collect_cache_axes()

        self._collect_views(data or {})

        self._collect_data(hide_implicit)

        self._init_data1d()
        self._collect_data1d(hide_implicit)
        self._collect_track()

        self._init_data2d()
        self._collect_data2d(hide_implicit)

    def _verify_axes(self, axes: Mapping[str, Union[None, str, AnyData, AxisView]]) -> None:
        for axis in axes:
            assert self.base._has_axis(axis), (
                f"missing the axis: {axis} "
                f"in the base storage: {self.base.name} "
                f"for the storage view: {self.name}"
            )

    def _collect_axes(self, axes: Mapping[str, Union[None, str, AnyData, AxisView]], hide_implicit: bool) -> None:
        for base_axis in self.base._axis_names():
            axis_view_data = axes.get(base_axis, None if hide_implicit else base_axis)
            axis_view: Optional[AxisView]
            if isinstance(axis_view_data, (AxisView, type(None))):
                axis_view = axis_view_data
            elif isinstance(axis_view_data, str):
                axis_view = AxisView(name=axis_view_data)
            else:
                axis_view = AxisView(entries=axis_view_data)

            if axis_view is None:
                self._base_axis_views[base_axis] = None
                continue

            exposed_axis = axis_view.name or base_axis

            entry_indices: Optional[Vector] = None
            if axis_view is not None and axis_view.entries is not None:
                entries_array = as_vector(axis_view.entries)
                assert_data(has_dtype(entries_array, ENTRIES_DTYPES), "any 1D data", entries_array)

                if has_dtype(entries_array, "bool"):
                    entry_indices = np.where(entries_array)[0]
                    assert entry_indices is not None
                elif has_dtype(entries_array, STR_DTYPE):
                    entry_indices = as_vector(
                        pd.Series(
                            np.arange(self.base._axis_size(base_axis)),
                            index=self.base._axis_entries(base_axis),
                        )[entries_array]
                    )
                else:
                    entry_indices = entries_array.copy()
                entry_indices = freeze(entry_indices)

            axis_full_view = AxisFullView(
                exposed_name=exposed_axis,
                base_name=base_axis,
                entry_indices=entry_indices,
                track=axis_view.track,
                overwrite=axis_view.overwrite,
            )
            self._exposed_axis_views[exposed_axis] = axis_full_view
            self._base_axis_views[base_axis] = axis_full_view

    def _collect_cache_axes(self) -> None:
        for exposed_axis, axis_full_view in self._exposed_axis_views.items():
            entries = as_vector(self.base._axis_entries(axis_full_view.base_name))
            if axis_full_view.entry_indices is not None:
                entries = entries[axis_full_view.entry_indices]
            self.cache.create_axis(exposed_axis, freeze(optimize(entries)))

    def _collect_views(self, data: Mapping[str, Optional[str]]) -> None:
        for base_data, data_view in data.items():
            assert self._collect_view(base_data, data_view), (
                f"missing the data: {base_data} "
                f"in the base storage: {self.base.name} "
                f"for the storage view: {self.name} "
            )

    def _collect_view(self, base_data: str, data_view: Optional[str]) -> bool:
        if ";" not in base_data:
            self._base_item_views[base_data] = data_view
            return self.base._has_item(base_data)

        axes, name = base_data.split(";")

        if "," not in axes:
            self._base_data1d_views[base_data] = data_view
            return self.base._has_data1d(axes, base_data)

        self._base_data2d_views[base_data] = data_view
        rows_axis, columns_axis = axes.split(",")
        transposed_data = f"{columns_axis},{rows_axis};{name}"
        if transposed_data not in self._base_data2d_views:
            self._base_data2d_views[transposed_data] = data_view
        else:
            transposed_view = self._base_data2d_views[transposed_data]
            assert transposed_view == data_view, (
                f"conflicting 2D data views: {base_data} => {data_view} "
                f"and: {transposed_data} => {transposed_view} "
                f"for the base storage: {self.base.name} "
                f"for the storage view: {self.name} "
            )
        return self.base._has_data2d((rows_axis, columns_axis), base_data)

    def _collect_data(self, hide_implicit: bool) -> None:
        for base_data in self.base._item_names():
            exposed_data = self._base_item_views.get(base_data, None if hide_implicit else base_data)
            self._base_item_views[base_data] = exposed_data
            if exposed_data is not None:
                assert exposed_data not in self._exposed_items, (
                    f"both the item: {self._exposed_items[exposed_data]} "
                    f"and the item: {base_data} "
                    f"of the base storage: {self.base.name} "
                    f"are exposed as the same item: {exposed_data} "
                    f"of the storage view: {self.name}"
                )
                self._exposed_items[exposed_data] = base_data

    def _init_data1d(self) -> None:
        for exposed_axis in self._exposed_axis_views:
            self._exposed_data1d[exposed_axis] = {}

    def _collect_data1d(self, hide_implicit: bool) -> None:
        for base_axis in self.base._axis_names():
            exposed_axis = self.exposed_axis(base_axis)
            if exposed_axis is None:
                continue

            _exposed_data1d = self._exposed_data1d[exposed_axis]

            for base_data in self.base._data1d_names(base_axis):
                exposed_data: Optional[str]

                name = _interface.extract_name(base_data)
                data_view = self._base_data1d_views.get(base_data, None if hide_implicit else name)
                if data_view is None:
                    exposed_data = None
                else:
                    exposed_data = f"{exposed_axis};{data_view}"

                self._base_data1d_views[base_data] = exposed_data
                if exposed_data is not None:
                    assert exposed_data not in _exposed_data1d, (
                        f"both the 1D data: {_exposed_data1d[exposed_data]} "
                        f"and the 1D data: {base_data} "
                        f"of the base storage: {self.base.name} "
                        f"are exposed as the same 1D data: {exposed_data} "
                        f"of the storage view: {self.name}"
                    )
                    _exposed_data1d[exposed_data] = base_data

    def _collect_track(self) -> None:
        for base_axis, axis_full_view in self._base_axis_views.items():
            if axis_full_view is None or axis_full_view.track is None:
                continue

            exposed_axis = axis_full_view.exposed_name
            exposed_name = f"{exposed_axis};{axis_full_view.track}"
            exposed_data1d = self._exposed_data1d[exposed_axis]

            assert axis_full_view.overwrite or exposed_name not in exposed_data1d, (
                f"both the 1D data: {exposed_data1d[exposed_name]} "
                f"and the tracked entry indices of the axis: {base_axis} "
                f"are exposed as the same 1D data: {exposed_name} "
                f"of the storage view: {self.name}"
                f"of the base view: {self.base.name}"
            )

            self._base_data1d_views[base_axis + ";"] = exposed_name
            entry_indices = axis_full_view.entry_indices
            exposed_data1d[exposed_name] = (
                np.arange(self.base.axis_size(base_axis)) if entry_indices is None else entry_indices
            )

    def _init_data2d(self) -> None:
        for exposed_rows_axis in self._exposed_axis_views:
            for exposed_columns_axis in self._exposed_axis_views:
                self._exposed_data2d[(exposed_rows_axis, exposed_columns_axis)] = {}

    def _collect_data2d(self, hide_implicit: bool) -> None:
        for base_rows_axis in self.base._axis_names():
            exposed_rows_axis = self.exposed_axis(base_rows_axis)
            if exposed_rows_axis is None:
                continue

            for base_columns_axis in self.base._axis_names():
                exposed_columns_axis = self.exposed_axis(base_columns_axis)
                if exposed_columns_axis is None:
                    continue

                exposed_data2d = self._exposed_data2d[(exposed_rows_axis, exposed_columns_axis)]

                for base_data in self.base._data2d_names((base_rows_axis, base_columns_axis)):
                    exposed_data: Optional[str]

                    name = _interface.extract_name(base_data)
                    data_view = self._base_data2d_views.get(base_data, None if hide_implicit else name)
                    if data_view is None:
                        exposed_data = None
                    else:
                        exposed_data = f"{exposed_rows_axis},{exposed_columns_axis};{data_view}"

                    self._base_data2d_views[base_data] = exposed_data
                    if exposed_data is not None:
                        assert exposed_data not in exposed_data2d, (
                            f"both the 2D data: {self._exposed_items[exposed_data]} "
                            f"and the 2D data: {base_data} "
                            f"of the base storage: {self.base.name} "
                            f"are exposed as the same 2D data: {exposed_data} "
                            f"of the storage view: {self.name}"
                        )
                        exposed_data2d[exposed_data] = base_data

    def _self_description(self, self_description: Dict, *, detail: bool) -> None:
        self_description["cache"] = self.cache.name
        self_description["base"] = self.base.name

    def _axes_description(self, *, detail: bool) -> Dict:
        axes: Dict = {}

        for axis in sorted(self._axis_names()):
            axis_full_view = self._exposed_axis_views[axis]
            if axis_full_view.entry_indices is None:
                description = "all " + str(self._axis_size(axis)) + " entries of " + self.base_axis(axis)
            else:
                exposed_size = len(axis_full_view.entry_indices)
                base_size = self.base._axis_size(axis_full_view.base_name)
                percent = exposed_size * 100 / base_size
                description = (
                    str(exposed_size)
                    + " out of "
                    + str(base_size)
                    + " entries of "
                    + str(self.base_axis(axis))
                    + f" ({percent:.2f}%)"
                )

            if detail:
                description += " in " + data_description(self._axis_entries(axis))
            axes[axis] = description

        return axes

    def _data_description(self, *, detail: bool) -> Union[Dict, List[str]]:
        data: Dict = {}

        for name in sorted(self._item_names()):
            description = "from " + self.base_item(name)
            if detail:
                description += " in " + data_description(self._get_item(name))
            data[name] = description

        axes = sorted(self._axis_names())

        for axis in axes:
            for name in sorted(self._data1d_names(axis)):
                description = "from " + self.base_data1d(name)
                if detail:
                    description += " in " + data_description(self._get_data1d(axis, name))
                data[name] = description

        for rows_axis in axes:
            for columns_axis in axes:
                for name in sorted(self._data2d_names((rows_axis, columns_axis))):
                    description = "from " + self.base_data2d(name)
                    if detail:
                        description += " in " + data_description(self._get_data2d((rows_axis, columns_axis), name))
                    data[name] = description

        return data

    def _deep_description(self, description: Dict, self_description: Dict, *, detail: bool) -> None:
        self.cache.description(deep=True, detail=detail, description=description)
        self.base.description(deep=True, detail=detail, description=description)

    def axis_slice_indices(self, exposed_axis: str) -> Optional[Vector]:
        """
        Return the original indices of the entries of the ``exposed_axis``, or ``None`` if the base axis was not
        sliced.
        """
        assert self._has_axis(exposed_axis), f"missing axis: {exposed_axis} in the base storage: {self.name}"
        return self._exposed_axis_views[exposed_axis].entry_indices

    def exposed_axis(self, base_axis: str) -> Optional[str]:
        """
        Given the ``base_axis`` in the base `.StorageReader`, return the name it is exposed as in the view, or ``None``
        if it is hidden.
        """
        assert self.base._has_axis(base_axis), f"missing axis: {base_axis} in the base storage: {self.base.name}"
        axis_full_view = self._base_axis_views[base_axis]
        if axis_full_view is None:
            return None
        return axis_full_view.exposed_name

    def exposed_item(self, base_item: str) -> Optional[str]:
        """
        Given the name of a ``base_item`` in the base `.StorageReader`, return the name it is exposed as in the view,
        or ``None`` if it is hidden.
        """
        assert self.base._has_item(base_item), f"missing item: {base_item} in the base storage: {self.base.name}"
        return self._base_item_views[base_item]

    def exposed_data1d(self, base_data1d: str) -> Optional[str]:
        """
        Given the name of an ``base_data1d`` in the base `.StorageReader`, return the name it is exposed as in the view,
        or ``None`` if it is hidden.

        The exposed name of tracked entry indices of an axis, if any, are available using ``axis;`` as the
        ``base_data1d`` name.
        """
        exposed_data1d = self._base_data1d_views.get(base_data1d)
        assert exposed_data1d is not None or self.base.has_data1d(
            base_data1d
        ), f"missing 1D data: {base_data1d} in the base storage: {self.base.name}"
        return exposed_data1d

    def exposed_data2d(self, base_data2d: str) -> Optional[str]:
        """
        Given the name of an ``base_data2d`` in the base `.StorageReader`, return the name it is exposed as in the view,
        or ``None`` if it is hidden.
        """
        assert self.base.has_data2d(
            base_data2d
        ), f"missing 2D data: {base_data2d} in the base storage: {self.base.name}"
        return self._base_data2d_views[base_data2d]

    def base_axis(self, exposed_axis: str) -> str:
        """
        Given the name of an ``exposed_axis`` (which must exist), return its name in the base `.StorageReader`.
        """
        assert self._has_axis(exposed_axis), f"missing axis: {exposed_axis} in the base storage: {self.name}"
        return self._exposed_axis_views[exposed_axis].base_name

    def base_item(self, exposed_item: str) -> str:
        """
        Given the name of an ``exposed_item`` (which must exist), return its name in the base `.StorageReader`.
        """
        assert self._has_item(exposed_item), f"missing item: {exposed_item} in the base storage: {self.name}"
        return self._exposed_items[exposed_item]

    def base_data1d(self, exposed_data1d: str) -> str:
        """
        Given the name of an ``exposed_data1d`` (which must exist), return its name in the base `.StorageReader`.

        The base name of tracked entry indices of an axis is reported as ``axis;``.
        """
        axis = _interface.extract_1d_axis(exposed_data1d)
        assert self._has_data1d(
            axis, exposed_data1d
        ), f"missing 1D data: {exposed_data1d} in the base storage: {self.name}"
        base_data1d = self._exposed_data1d[axis][exposed_data1d]
        if isinstance(base_data1d, str):
            return base_data1d
        return self._exposed_axis_views[axis].base_name + ";"

    def base_data2d(self, exposed_data2d: str) -> str:
        """
        Given the name of an ``exposed_data2d`` (which must exist), return its name in the base `.StorageReader`.
        """
        axes = _interface.extract_2d_axes(exposed_data2d)
        assert self._has_data2d(
            axes, exposed_data2d
        ), f"missing 2D data: {exposed_data2d} in the base storage: {self.name}"
        return self._exposed_data2d[axes][exposed_data2d]

    def _item_names(self) -> Collection[str]:
        return self._exposed_items.keys()

    def _has_item(self, name: str) -> bool:
        return name in self._exposed_items

    def _get_item(self, name: str) -> Any:
        return self.base.get_item(self._exposed_items[name])

    def _axis_names(self) -> Collection[str]:
        return self._exposed_axis_views.keys()

    def _has_axis(self, axis: str) -> bool:
        return axis in self._exposed_axis_views

    def _axis_size(self, axis: str) -> int:
        return self.cache._axis_size(axis)

    def _axis_entries(self, axis: str) -> Known1D:
        return self.cache._axis_entries(axis)

    def _data1d_names(self, axis: str) -> Collection[str]:
        return self._exposed_data1d[axis].keys()

    def _has_data1d(self, axis: str, name: str) -> bool:
        return name in self._exposed_data1d[axis]

    def _get_data1d(self, axis: str, name: str) -> Known1D:
        if self.cache._has_data1d(axis, name):
            return self.cache._get_data1d(axis, name)

        base_data1d = self._exposed_data1d[axis][name]
        if not isinstance(base_data1d, str):
            return base_data1d

        data1d = self.base.get_data1d(base_data1d)

        base_entries = self._exposed_axis_views[axis].entry_indices
        if base_entries is None:
            return data1d

        vector = freeze(optimize(as_vector(data1d)[base_entries]))
        self.cache._set_vector(axis, name, vector)
        return vector

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:
        return self._exposed_data2d[axes].keys()

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:
        return name in self._exposed_data2d[axes].keys()

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Known2D:
        if self.cache._has_data2d(axes, name):
            return self.cache._get_data2d(axes, name)

        data2d = self.base.get_data2d(self._exposed_data2d[axes][name])
        matrix: Optional[Matrix] = None

        base_column_entries = self._exposed_axis_views[axes[1]].entry_indices
        if base_column_entries is not None:
            matrix = as_matrix(data2d)
            matrix = matrix[:, base_column_entries]

        base_row_entries = self._exposed_axis_views[axes[0]].entry_indices
        if base_row_entries is not None:
            matrix = matrix or as_matrix(data2d)
            matrix = matrix[base_row_entries, :]

        if matrix is None:
            return data2d

        matrix = freeze(optimize(as_layout(matrix, ROW_MAJOR)))
        self.cache._set_matrix(axes, name, matrix)
        return matrix
