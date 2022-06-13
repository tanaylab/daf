"""
Support functions for describing data in messages.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional
from typing import Set

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import dtypes as _dtypes
from . import fake_pandas as _fake_pandas
from . import fake_sparse as _fake_sparse
from . import freezing as _freezing
from . import layouts as _layouts
from . import optimization as _optimization

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "data_description",
    "assert_data",
]


def data_description(data: Any) -> str:
    """
    Return a not-too-long description of some (hopefully 1D/2D) data for error messages and logging.
    """
    if isinstance(data, np.ndarray):
        return _ndarray_description(data)

    if isinstance(data, pd.Series):
        return _series_description(data)

    if isinstance(data, pd.DataFrame):
        return _frame_description(data)

    if isinstance(data, (sp.csr_matrix, sp.csc_matrix)):
        return _compressed_description(data)

    if isinstance(data, sp.spmatrix):
        return _sparse_description(data)

    suffix = ""
    if isinstance(data, (int, float, bool)):
        suffix = f" = {data}"
    elif isinstance(data, str) and "\n" not in data and len(data) < 100:
        suffix = " = " + data
    return f"{data.__class__.__module__}.{data.__class__.__qualname__}{suffix}"


def _ndarray_description(data: np.ndarray) -> str:  # pylint: disable=too-many-return-statements
    if str(data.dtype) == "bool":
        true = np.sum(data)
        if true == data.size:
            suffix = " (all true)"
        elif true == 0:
            suffix = " (all false)"
        else:
            percent = true * 100 / data.size
            suffix = f" ({true} true, {percent:.2f}%)"
    else:
        suffix = ""

    frozen = "frozen " if _freezing.is_frozen(data) else ""
    if data.ndim == 1:
        return f"{frozen}1D numpy.ndarray of {data.shape[0]} of {data.dtype}{suffix}"

    if isinstance(data, np.matrix):
        return f"{frozen}numpy.matrix of {data.shape[0]}x{data.shape[1]} of {data.dtype}{suffix}"

    if data.ndim == 2:
        is_column_major = _layouts.has_layout(data, _layouts.COLUMN_MAJOR)
        is_row_major = _layouts.has_layout(data, _layouts.ROW_MAJOR)

        optimal = ""
        if (is_column_major or is_row_major) and not _optimization.is_optimal(data):
            optimal = "non-optimal "

        if is_column_major and is_row_major:
            return (
                f"{frozen}{optimal}both-major numpy.ndarray of "
                f"{data.shape[0]}x{data.shape[1]} of {data.dtype}{suffix}"
            )
        if is_column_major:
            return (
                f"{frozen}{optimal}column-major numpy.ndarray of "
                f"{data.shape[0]}x{data.shape[1]} of {data.dtype}{suffix}"
            )
        if is_row_major:
            return (
                f"{frozen}{optimal}row-major numpy.ndarray of "
                f"{data.shape[0]}x{data.shape[1]} of {data.dtype}{suffix}"
            )
        return f"{frozen}none-major numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}{suffix}"

    return f"{frozen}{data.ndim}D numpy.ndarray of {data.shape[0]}x{data.shape[1]} of {data.dtype}{suffix}"


def _series_description(data: _fake_pandas.Series) -> str:
    if not isinstance(data.values, np.ndarray):
        return (
            "pandas.Series containing an instance of "
            f"{data.values.__class__.__module__}.{data.values.__class__.__qualname__} "
            f"of {data.shape[0]} of {data.dtype}"
        )
    assert data.values.ndim == 1
    frozen = "frozen " if _freezing.is_frozen(data) else ""
    return f"{frozen}pandas.Series of {len(data)} of {data.dtype}"


def _frame_description(data: _fake_pandas.DataFrame) -> str:
    if not isinstance(data.values, np.ndarray):
        return (
            "pandas.DataFrame containing an instance of "
            f"{data.values.__class__.__module__}.{data.values.__class__.__qualname__} "
            f"of {data.shape[0]}x{data.shape[1]} of {data.dtype}"
        )
    assert data.values.ndim == 2
    frozen = "frozen " if _freezing.is_frozen(data.values) else ""
    is_column_major = _layouts.has_layout(data.values, _layouts.COLUMN_MAJOR)
    is_row_major = _layouts.has_layout(data.values, _layouts.ROW_MAJOR)

    optimal = ""
    if (is_column_major or is_row_major) and not _optimization.is_optimal(data.values):
        optimal = "non-optimal "

    dtypes = set(data.dtypes)
    if len(dtypes) == 1:
        dtype = str(data.dtypes[0])
    else:
        dtype = "mixed types"

    if is_column_major and is_row_major:
        return f"{frozen}{optimal}both-major pandas.DataFrame of {data.shape[0]}x{data.shape[1]} of {dtype}"
    if is_column_major:
        return f"{frozen}{optimal}column-major pandas.DataFrame of {data.shape[0]}x{data.shape[1]} of {dtype}"
    if is_row_major:
        return f"{frozen}{optimal}row-major pandas.DataFrame of {data.shape[0]}x{data.shape[1]} of {dtype}"
    return f"none-major pandas.DataFrame of {data.shape[0]}x{data.shape[1]} of {dtype}"


def _compressed_description(data: _fake_sparse.cs_matrix) -> str:
    text: List[str] = []
    if _freezing.is_frozen(data):
        text.append("frozen ")
    if not _optimization.is_optimal(data):
        text.append("non-optimal ")
    if isinstance(data, sp.csr_matrix):
        text.append("scipy.sparse.csr_matrix")
    else:
        text.append("scipy.sparse.csc_matrix")
    percent = data.nnz * 100 / (data.shape[0] * data.shape[1])
    text.append(f" of {data.shape[0]}x{data.shape[1]} of {data.dtype} with {percent:.2f}% nnz")
    are_frozen: Set[Optional[bool]] = set()
    if not _optimization.is_optimal(data):
        for field, value in (("data", data.data), ("indices", data.indices), ("indptr", data.indptr)):
            if not isinstance(value, np.ndarray):
                are_frozen.add(None)
            else:
                are_frozen.add(_freezing.is_frozen(value))
        report_frozen = len(are_frozen) > 1
        separator = " ("
        for field, value in (("data", data.data), ("indices", data.indices), ("indptr", data.indptr)):
            text.extend([separator, field, ": "])
            if isinstance(value, np.ndarray):
                if report_frozen and _freezing.is_frozen(value):
                    text.append("frozen ")
                text.append(str(value.ndim))
                text.append("D numpy.ndarray")
            else:
                text.extend([value.__class__.__module__, ".", value.__class__.__qualname__])
            separator = ", "
        text.append(")")

    return "".join(text)


def _sparse_description(data: _fake_sparse.spmatrix) -> str:
    percent = data.nnz * 100 / (data.shape[0] * data.shape[1])
    return (
        f"{data.__class__.__module__}.{data.__class__.__qualname__} "
        f"of {data.shape[0]}x{data.shape[1]} of {data.dtype} with {percent:.2f}% nnz"
    )


def assert_data(
    condition: bool,
    kind: str,
    data: Any,
    *,
    dtype: Optional[_dtypes.DTypes] = None,
    layout: Optional[_layouts.AnyMajor] = None,
) -> None:
    """
    Assert that the ``data`` satisfies some ``condition``, which tests it is of some ``kind`` (and optionally ``dtype``
    and ``layout``), with a friendly message if it fails.
    """
    if condition:
        return

    if kind == "pandas.DataFrame":
        assert False, f"expected: {kind}, got: {data_description(data)}"

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
    elif isinstance(dtype, (str, np.dtype)):
        dtype = str(dtype)
    else:
        dtype = " or ".join([str(expected_dtype) for expected_dtype in dtype])

    if layout is None:
        layout_prefix = ""
    else:
        layout_prefix = layout.name + " "

    assert False, f"expected: {layout_prefix}{kind} of {dtype}, got: {data_description(data)}"
