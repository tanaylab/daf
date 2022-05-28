"""
The types here describe any 1D data which may be obtained from ``daf`` storage. Currently there are only two such types:

* `.Array1D` is a one-dimensional ``numpy.ndarray``.

* `.Series` is a ``pandas.Series`` which combines an ``Array1D`` with an index of names.

The `.Vector` type annotations is their union, that is, allows for "any" 1D data.

Since 1D data is so much smaller than 2D data, and since ``scipy.sparse`` compressed vector types are so different and
so restricted, ``daf`` does not support a sparse 1D vector format. E.g., if you slice a single row or column of a sparse
matrix, you will need to convert it to a dense format (e.g. by calling `.as_array1d`) before storing it in ``daf``.

.. note::

    The `.Vector` type should be used in computations with **great care**, as some operations are subtly different for
    1D ``numpy.ndarray`` arrays and ``pandas.Series``. It is typically better to use one of the concrete types instead.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Union

try:
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

from . import array1d as _array1d
from . import descriptions as _descriptions
from . import series as _series

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "Vector",
    "is_vector",
    "be_vector",
]


#: Any 1D data.
Vector = Union["_array1d.Array1D", _series.Series]


def is_vector(data: Any, *, dtype: Union[None, str, Collection[str]] = None) -> TypeGuard[Vector]:
    """
    Assert that some ``data`` is a `.Vector`, optionally only of some ``dtype``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return _array1d.is_array1d(data, dtype=dtype) or _series.is_series(data, dtype=dtype)


def be_vector(data: Any, *, dtype: Union[None, str, Collection[str]] = None) -> Vector:
    """
    Assert that some ``data`` is a `.Vector`, optionally only of some ``dtype``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_vector(data, dtype=dtype), "vector", data, dtype)
    return data
