"""
Support functions for describing data in messages.
"""

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Union

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import layouts as _layouts  # pylint: disable=cyclic-import

__all__ = [
    "data_description",
    "assert_data",
]


def data_description(data: Any) -> str:  # pylint: disable=too-many-return-statements,too-many-branches
    """
    Return a short description of some hopefully 1D/2D data for error messages and logging.
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return f"1D numpy.ndarray of {data.shape[0]} of {data.dtype}"

        if isinstance(data, np.matrix):
            return f"numpy.matrix of {data.shape[0]}x{data.shape[1]} of {data.dtype}"

        if data.ndim == 2:
            is_column_major = _layouts.COLUMN_MAJOR.is_layout_of(data)  # type: ignore
            is_row_major = _layouts.ROW_MAJOR.is_layout_of(data)  # type: ignore
            if is_column_major and is_row_major:
                return f"both-major numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}"
            if is_column_major:
                return f"column-major numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}"
            if is_row_major:
                return f"row-major numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}"
            return f"none-major numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}"

        return f"{data.ndim}D numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}"

    if isinstance(data, pd.Series):
        if not isinstance(data.values, np.ndarray):
            return (
                "pandas.Series containing an instance of "
                f"{data.values.__class__.__module__}.{data.values.__class__.__qualname__} "
                f"of {data.shape[0]} of {data.dtype}"
            )
        assert data.values.ndim == 1
        return f"pandas.Series of {len(data)} of {data.dtype}"

    if isinstance(data, pd.DataFrame):
        if not isinstance(data.values, np.ndarray):
            return (
                "pandas.DataFrame containing an instance of "
                f"{data.values.__class__.__module__}.{data.values.__class__.__qualname__} "
                f"of {data.shape[0]}x{data.shape[1]} of {data.dtype}"
            )
        assert data.values.ndim == 2
        is_column_major = _layouts.COLUMN_MAJOR.is_layout_of(data.values)  # type: ignore
        is_row_major = _layouts.ROW_MAJOR.is_layout_of(data.values)  # type: ignore
        dtypes = np.unique(data.dtypes)
        if len(dtypes) > 1:
            return f"pandas.DataFrame of {data.shape[0]}x{data.shape[1]} of mixed types"
        dtype = str(dtypes[0])
        if is_column_major and is_row_major:
            return f"both-major pandas Table of {data.shape[0]}x{data.shape[1]} of {dtype}"
        if is_column_major:
            return f"column-major pandas Table of {data.shape[0]}x{data.shape[1]} of {dtype}"
        if is_row_major:
            return f"row-major pandas Table of {data.shape[0]}x{data.shape[1]} of {dtype}"
        return f"none-major pandas Table of {data.shape[0]}x{data.shape[1]} of {dtype}"

    if isinstance(data, sp.csr_matrix):
        percent = data.nnz * 100 / (data.shape[0] * data.shape[1])
        return f"scipy.sparse.csr_matrix of {data.shape[0]}x{data.shape[1]} of {data.dtype} with {percent:.2f}% nnz"

    if isinstance(data, sp.csc_matrix):
        percent = data.nnz * 100 / (data.shape[0] * data.shape[1])
        return f"scipy.sparse.csc_matrix of {data.shape[0]}x{data.shape[1]} of {data.dtype} with {percent:.2f}% nnz"

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
    if dtype is None:
        assert False, f"expected {kind} of any reasonable type, got {data_description(data)}"
    if isinstance(dtype, str):
        dtype = [dtype]
    assert False, f"expected {kind} of {' or '.join(dtype)}, got {data_description(data)}"
