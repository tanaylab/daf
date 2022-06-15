"""
The types here describe a 1D ``pandas.Series``, which can be fetched from ``daf``.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Optional

try:
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

import numpy as np
import pandas as pd  # type: ignore

from . import descriptions as _descriptions
from . import dtypes as _dtypes
from . import fake_pandas as _fake_pandas

# pylint: enable=duplicate-code,cyclic-import


__all__ = [
    "is_series",
    "be_series",
]


def is_series(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, size: Optional[int] = None
) -> TypeGuard[_fake_pandas.Series]:
    """
    Check whether some ``data`` is a `.Series`, optionally only of some ``dtype``, optionally of some ``size``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return (
        isinstance(data, pd.Series)
        and (size is None or len(data) == size)
        and (
            (isinstance(data.values, np.ndarray) and data.values.ndim == 1) or str(data.dtype) in ("string", "category")
        )
        and _dtypes.has_dtype(data, dtype)
    )


def be_series(data: Any, *, dtype: Optional[_dtypes.DTypes] = None, size: Optional[int] = None) -> _fake_pandas.Series:
    """
    Assert that some ``data`` is a `.Series`, optionally only of some ``dtype``, optionally of some ``size``, and return
    it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_series(data, dtype=dtype, size=size), "pandas.Series", data, dtype=dtype, size=size)
    return data
