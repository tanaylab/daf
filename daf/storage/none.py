"""
The `.NO_STORAGE` is used to disable caching (e.g., caching sliced data in a `.StorageView`).

Creating or setting data inside `.NO_STORAGE` has no effect. Unlike a real storage, it allows querying for names of axes
that don't exist (since in it, no axes exist).
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from ..typing import DenseInRows
from ..typing import DType
from ..typing import Known1D
from ..typing import Known2D
from ..typing import MatrixInRows
from ..typing import Vector
from ..typing import be_dense_in_rows
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

    def description(self, *, detail: bool = False, deep: bool = False, description: Optional[Dict] = None) -> Dict:
        description = description or {}
        description[self.name] = "NO_STORAGE"
        return description

    def _item_names(self) -> Collection[str]:
        return []

    def _has_item(self, name: str) -> bool:
        return False

    def get_item(self, name: str) -> Any:
        assert False, "never happens"

    def _get_item(self, name: str) -> Any:
        assert False, "never happens"

    def _axis_names(self) -> Collection[str]:
        return []

    def _has_axis(self, axis: str) -> bool:
        return False

    def axis_size(self, axis: str) -> int:
        assert False, "never happens"

    def _axis_size(self, axis: str) -> int:
        assert False, "never happens"

    def axis_entries(self, axis: str) -> Known1D:
        assert False, "never happens"

    def _axis_entries(self, axis: str) -> Known1D:
        assert False, "never happens"

    def data1d_names(self, axis: str) -> Collection[str]:
        return []

    def _data1d_names(self, axis: str) -> Collection[str]:
        assert False, "never happens"

    def has_data1d(self, name: str) -> bool:
        return False

    def _has_data1d(self, axis: str, name: str) -> bool:
        assert False, "never happens"

    def get_data1d(self, name: str) -> Known1D:
        assert False, "never happens"

    def _get_data1d(self, axis: str, name: str) -> Known1D:
        assert False, "never happens"

    def data2d_names(self, axes: Union[str, Tuple[str, str]]) -> Collection[str]:
        return []

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:
        assert False, "never happens"

    def has_data2d(self, name: str) -> bool:
        return False

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:
        assert False, "never happens"

    def get_data2d(self, name: str) -> Known2D:
        assert False, "never happens"

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Known2D:
        assert False, "never happens"

    def update(self, storage: _interface.StorageReader, *, overwrite: bool = False) -> None:
        pass

    def set_item(self, name: str, item: Any, *, overwrite: bool = False) -> None:
        pass

    def _set_item(self, name: str, item: Any) -> None:
        assert False, "never happens"

    def create_axis(self, axis: str, entries: Vector) -> None:
        pass

    def _create_axis(self, axis: str, entries: Vector) -> None:
        assert False, "never happens"

    def set_vector(self, name: str, vector: Vector, *, overwrite: bool = False) -> None:
        pass

    def _set_vector(self, axis: str, name: str, vector: Vector) -> None:
        assert False, "never happens"

    def set_matrix(self, name: str, matrix: MatrixInRows, *, overwrite: bool = False) -> None:
        pass

    def _set_matrix(self, axes: Tuple[str, str], name: str, matrix: MatrixInRows) -> None:
        assert False, "never happens"

    @contextmanager
    def create_dense_in_rows(
        self, name: str, *, dtype: DType, overwrite: bool = False
    ) -> Generator[DenseInRows, None, None]:
        assert False, "never happens"

    @contextmanager
    def _create_dense_in_rows(
        self, name: str, *, axes: Tuple[str, str], shape: Tuple[int, int], dtype: DType
    ) -> Generator[DenseInRows, None, None]:
        yield be_dense_in_rows(np.empty(shape, dtype=dtype))


#: Storage used to specify not caching sliced or derived data.
NO_STORAGE = NoStorage()
