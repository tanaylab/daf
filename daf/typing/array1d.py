"""
The types here describe a 1D ``numpy.ndarray``, which is how 1D data is stored in ``daf``.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import NewType
from typing import Sequence
from typing import Union

try:
    from typing import Annotated  # pylint: disable=unused-import
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import descriptions as _descriptions
from . import dtypes as _dtypes

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "Array1D",
    "is_array1d",
    "be_array1d",
    "Data1D",
    "as_array1d",
]


#: 1-dimensional ``numpy`` array of bool values.
Array1D = NewType("Array1D", "Annotated[np.ndarray, '1D']")


def is_array1d(data: Any, *, dtype: Union[None, str, Collection[str]] = None) -> TypeGuard[Array1D]:
    """
    Check whether some ``data`` is an :py:obj:`~Array1D`, optionally only of some ``dtype``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    return isinstance(data, np.ndarray) and data.ndim == 1 and _dtypes.is_dtype(str(data.dtype), dtype)


def be_array1d(data: Any, *, dtype: Union[None, str, Collection[str]] = None) -> Array1D:
    """
    Assert that some ``data`` is an :py:obj:`~Array1D`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_array1d(data, dtype=dtype), "1D numpy.ndarray", data, dtype)
    return data


#: Anything that can be used to construct an :py:obj:`~Array1D`.
Data1D = Union[Sequence[Any], sp.spmatrix, pd.DataFrame, np.ndarray]


def as_array1d(data: Data1D, *, force_copy: bool = False) -> Array1D:
    """
    Access the internal 1D ``numpy`` array, if possible; otherwise, or if ``force_copy``, return a copy of the 1D data
    as a ``numpy`` array.

    Accepts as input data types that aren't even a :py:obj:`~daf.typing.vectors.Vector`, such as lists or even 2D data
    with a single row or column.

    This ensures that ``pandas`` strings (even if categorical) will be converted to proper ``numpy`` strings.

    This will reshape any matrix with a single row or a single column into a 1D ``numpy`` array.

    This will convert lists (or any other sequence of values) into a 1D ``numpy`` array.
    """
    if isinstance(data, sp.spmatrix):
        _descriptions.assert_data(min(data.shape) < 2, "1D data", data, None)
        data = data.toarray()
        force_copy = False

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.values

    if force_copy or not isinstance(data, np.ndarray) or data.dtype == "category" or isinstance(data, np.matrix):
        data = np.array(data)

    if data.ndim == 2:
        _descriptions.assert_data(min(data.shape) < 2, "1D data", data, None)
        data = np.reshape(data, -1)
    else:
        _descriptions.assert_data(data.ndim == 1, "1D data", data, None)

    return data  # type: ignore
