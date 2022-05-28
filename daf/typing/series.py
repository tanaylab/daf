"""
The types here describe a 1D ``pandas.Series``, which can be obtained from ``daf`` storage.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import NewType
from typing import Union

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
    "Series",
    "is_series",
    "be_series",
]


#: 1-dimensional ``pandas.Series``.
Series = NewType("Series", _fake_pandas.PandasSeries)


def is_series(data: Any, *, dtype: Union[None, str, Collection[str]] = None) -> TypeGuard[Series]:
    """
    Check whether some ``data`` is a `.Series`, optionally only of some ``dtype``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return (
        isinstance(data, pd.Series)
        and (
            (isinstance(data.values, np.ndarray) and data.values.ndim == 1) or str(data.dtype) in ("string", "category")
        )
        and _dtypes.is_dtype(str(data.dtype), dtype)
    )


def be_series(data: Any, *, dtype: Union[None, str, Collection[str]] = None) -> Series:
    """
    Assert that some ``data`` is a `.Series`, optionally only of some ``dtype``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_series(data, dtype=dtype), "pandas.Series", data, dtype)
    return data
