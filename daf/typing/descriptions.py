"""
Support functions for describing data in messages.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Union

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import dtypes as _dtypes
from . import freezing as _freezing
from . import layouts as _layouts
from . import optimization as _optimization

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "data_description",
    "assert_data",
]


def data_description(data: Any) -> str:  # pylint: disable=too-many-return-statements,too-many-branches
    """
    Return a short description of some hopefully 1D/2D data for error messages and logging.
    """
    if isinstance(data, np.ndarray):
        frozen = "frozen " if _freezing.is_frozen(data) else ""  # type: ignore
        if data.ndim == 1:
            return f"{frozen}1D numpy.ndarray of {data.shape[0]} of {data.dtype}"

        if isinstance(data, np.matrix):
            return f"{frozen}numpy.matrix of {data.shape[0]}x{data.shape[1]} of {data.dtype}"

        if data.ndim == 2:
            is_column_major = _layouts.COLUMN_MAJOR.is_layout_of(data)  # type: ignore
            is_row_major = _layouts.ROW_MAJOR.is_layout_of(data)  # type: ignore

            optimal = ""
            if (is_column_major or is_row_major) and not _optimization.is_optimal(data):
                optimal = "non-optimal "

            if is_column_major and is_row_major:
                return f"{frozen}{optimal}both-major numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}"
            if is_column_major:
                return f"{frozen}{optimal}column-major numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}"
            if is_row_major:
                return f"{frozen}{optimal}row-major numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}"
            return f"{frozen}none-major numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}"

        return f"{frozen}{data.ndim}D numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}"

    if isinstance(data, pd.Series):
        if not isinstance(data.values, np.ndarray):
            return (
                "pandas.Series containing an instance of "
                f"{data.values.__class__.__module__}.{data.values.__class__.__qualname__} "
                f"of {data.shape[0]} of {data.dtype}"
            )
        assert data.values.ndim == 1
        frozen = "frozen " if _freezing.is_frozen(data) else ""
        return f"{frozen}pandas.Series of {len(data)} of {data.dtype}"

    if isinstance(data, pd.DataFrame):
        if not isinstance(data.values, np.ndarray):
            return (
                "pandas.DataFrame containing an instance of "
                f"{data.values.__class__.__module__}.{data.values.__class__.__qualname__} "
                f"of {data.shape[0]}x{data.shape[1]} of {data.dtype}"
            )
        assert data.values.ndim == 2
        frozen = "frozen " if _freezing.is_frozen(data) else ""
        is_column_major = _layouts.COLUMN_MAJOR.is_layout_of(data.values)  # type: ignore
        is_row_major = _layouts.ROW_MAJOR.is_layout_of(data.values)  # type: ignore

        optimal = ""
        if (is_column_major or is_row_major) and not _optimization.is_optimal(data):
            optimal = "non-optimal "

        dtypes = np.unique(data.dtypes)
        if len(dtypes) > 1:
            return f"pandas.DataFrame of {data.shape[0]}x{data.shape[1]} of mixed types"
        dtype = str(dtypes[0])

        if is_column_major and is_row_major:
            return f"{frozen}{optimal}both-major pandas.DataFrame of {data.shape[0]}x{data.shape[1]} of {dtype}"
        if is_column_major:
            return f"{frozen}{optimal}column-major pandas.DataFrame of {data.shape[0]}x{data.shape[1]} of {dtype}"
        if is_row_major:
            return f"{frozen}{optimal}row-major pandas.DataFrame of {data.shape[0]}x{data.shape[1]} of {dtype}"
        return f"none-major pandas.DataFrame of {data.shape[0]}x{data.shape[1]} of {dtype}"

    if isinstance(data, sp.csr_matrix):
        percent = data.nnz * 100 / (data.shape[0] * data.shape[1])
        frozen = "frozen " if _freezing.is_frozen(data) else ""
        optimal = "" if _optimization.is_optimal(data) else "non-optimal "
        return (
            f"{frozen}{optimal}scipy.sparse.csr_matrix "
            f"of {data.shape[0]}x{data.shape[1]} of {data.dtype} with {percent:.2f}% nnz"
        )

    if isinstance(data, sp.csc_matrix):
        percent = data.nnz * 100 / (data.shape[0] * data.shape[1])
        frozen = "frozen " if _freezing.is_frozen(data) else ""
        optimal = "" if _optimization.is_optimal(data) else "non-optimal "
        return (
            f"{frozen}{optimal}scipy.sparse.csc_matrix "
            f"of {data.shape[0]}x{data.shape[1]} of {data.dtype} with {percent:.2f}% nnz"
        )

    if isinstance(data, sp.spmatrix):
        percent = data.nnz * 100 / (data.shape[0] * data.shape[1])
        return (
            f"{data.__class__.__module__}.{data.__class__.__qualname__} "
            f"of {data.shape[0]}x{data.shape[1]} of {data.dtype} with {percent:.2f}% nnz"
        )

    return f"{data.__class__.__module__}.{data.__class__.__qualname__}"


def assert_data(condition: bool, kind: str, data: Any, dtype: Union[None, str, Collection[str]]) -> None:
    """
    Assert that the ``data`` satisfies some ``condition`` tesing it is of some ``kind`` and ``dtype``, with a friendly
    message if it fails.
    """
    if condition:
        return

    if kind == "pandas.DataFrame":
        assert False, f"expected {kind}, got {data_description(data)}"

    if dtype is None or dtype == _dtypes.ALL_DTYPES:
        dtype = "any reasonable type"
    elif dtype == _dtypes.INT_DTYPES:
        dtype = "int"
    elif dtype == _dtypes.FLOAT_DTYPES:
        dtype = "float"
    elif dtype == _dtypes.NUM_DTYPES:
        dtype = "number"
    elif dtype == _dtypes.FIXED_DTYPES:
        dtype = "fixed"
    elif dtype == _dtypes.ENTRIES_DTYPES:
        dtype = "entries (bool or int or str)"
    elif not isinstance(dtype, str):
        dtype = " or ".join(dtype)

    assert False, f"expected {kind} of {dtype}, got {data_description(data)}"
