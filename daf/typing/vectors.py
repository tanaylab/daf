"""
The types here describe a 1D ``numpy.ndarray``, which can be fetched from ``daf``.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import NewType
from typing import Optional

try:
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import descriptions as _descriptions
from . import dtypes as _dtypes
from . import unions as _unions

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "Vector",
    "is_vector",
    "be_vector",
    "as_vector",
]


#: 1-dimensional ``numpy`` array of bool values.
Vector = NewType("Vector", np.ndarray)


def is_vector(data: Any, *, dtype: Optional[_dtypes.DTypes] = None, size: Optional[int] = None) -> TypeGuard[Vector]:
    """
    Check whether some ``data`` is a `.Vector`, optionally only of some ``dtype``, optionally only of some ``size``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return (
        isinstance(data, np.ndarray)
        and (size is None or data.size == size)
        and data.ndim == 1
        and _dtypes.has_dtype(data, dtype)
    )


def be_vector(data: Any, *, dtype: Optional[_dtypes.DTypes] = None, size: Optional[int] = None) -> Vector:
    """
    Assert that some ``data`` is a `.Vector`, optionally only of some ``dtype``, optionally only of some ``size``, and
    return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_vector(data, dtype=dtype, size=size), "1D numpy.ndarray", data, dtype=dtype, size=size)
    return data


def as_vector(data: _unions.AnyData, *, force_copy: bool = False) -> Vector:
    """
    Access the internal 1D ``numpy`` array, if possible; otherwise, or if ``force_copy``, return a copy of the 1D data
    as a ``numpy`` array.

    Accepts as input data types that aren't even a `.Vector`, such as lists or even 2D data with a single row or column.

    This ensures that ``pandas`` strings (even if categorical) will be converted to proper ``numpy`` strings.

    This will reshape any matrix with a single row or a single column into a 1D ``numpy`` array.

    This will convert lists (or any other sequence of values) into a 1D ``numpy`` array.
    """
    if isinstance(data, sp.spmatrix):
        _descriptions.assert_data(min(data.shape) < 2, "any 1D data", data)
        data = data.toarray()
        force_copy = False

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.values

    if force_copy or not isinstance(data, np.ndarray) or str(data.dtype) == "category" or isinstance(data, np.matrix):
        data = np.array(data)

    if data.ndim == 2:
        _descriptions.assert_data(min(data.shape) < 2, "any 1D data", data)
        data = np.reshape(data, -1)
    else:
        _descriptions.assert_data(data.ndim == 1, "any 1D data", data)

    return data  # type: ignore
