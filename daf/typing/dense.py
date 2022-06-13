"""
The types here describe a dense 2D ``numpy.ndarray`` matrices, which can be fetched from ``daf``.

.. note::

    We actively prevent using the deprecated data type ``numpy.matrix`` which is occasionally returned from some
    operations, ``isinstance`` of ``numpy.ndarray``, and behaves just differently enough from a normal ``numpy.ndarray``
    to cause subtle and difficult problems.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import NewType
from typing import Optional
from typing import Union
from typing import overload

try:
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import descriptions as _descriptions
from . import dtypes as _dtypes
from . import fake_sparse as _fake_sparse
from . import frames as _frames
from . import layouts as _layouts
from . import sparse as _sparse
from . import unions as _unions

# pylint: enable=duplicate-code,cyclic-import


__all__ = [
    "Dense",
    "is_dense",
    "be_dense",
    "as_dense",
    "DenseInRows",
    "is_dense_in_rows",
    "be_dense_in_rows",
    "DenseInColumns",
    "is_dense_in_columns",
    "be_dense_in_columns",
]

#: 2-dimensional ``numpy`` array (dense matrix) in `.ROW_MAJOR` layout.
DenseInRows = NewType("DenseInRows", np.ndarray)


def is_dense_in_rows(data: Any, *, dtype: Optional[_dtypes.DTypes] = None) -> TypeGuard[DenseInRows]:
    """
    Check whether some ``data`` is a `.DenseInRows`, optionally only of some ``dtype``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return is_dense(data, dtype=dtype, layout=_layouts.ROW_MAJOR)


def be_dense_in_rows(data: Any, *, dtype: Optional[_dtypes.DTypes] = None) -> DenseInRows:
    """
    Assert that some ``data`` is a `.DenseInRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_dense_in_rows(data, dtype=dtype), "row-major numpy.ndarray", data, dtype=dtype)
    return data


#: 2-dimensional ``numpy`` array (dense matrix) in `.COLUMN_MAJOR` layout.
DenseInColumns = NewType("DenseInColumns", np.ndarray)


def is_dense_in_columns(data: Any, *, dtype: Optional[_dtypes.DTypes] = None) -> TypeGuard[DenseInColumns]:
    """
    Check whether some ``data`` is a `.DenseInColumns`, optionally only of some ``dtype``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return is_dense(data, dtype=dtype, layout=_layouts.COLUMN_MAJOR)


def be_dense_in_columns(data: Any, *, dtype: Optional[_dtypes.DTypes] = None) -> DenseInColumns:
    """
    Assert that some ``data`` is a `.DenseInColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_dense_in_columns(data, dtype=dtype), "column-major numpy.ndarray", data, dtype=dtype)
    return data


#: 2-dimensional ``numpy`` array (dense matrix) in either `.ROW_MAJOR` or `.COLUMN_MAJOR` layout.
Dense = Union[DenseInRows, DenseInColumns]


def is_dense(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, layout: Optional[_layouts.AnyMajor] = None
) -> TypeGuard[Dense]:
    """
    Check whether some ``data`` is a `.Dense`, optionally only of some ``dtype``, optionally only of some ``layout``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    return (
        isinstance(data, np.ndarray)
        and data.ndim == 2
        and not isinstance(data, np.matrix)
        and _dtypes.has_dtype(data, dtype)
        and _layouts.has_layout(data, layout)
    )


def be_dense(
    data: Any,
    *,
    dtype: Optional[_dtypes.DTypes] = None,
    layout: Optional[_layouts.AnyMajor] = None,
) -> Dense:
    """
    Assert that some ``data`` is a `.Dense` optionally only of some ``dtype``, optionally only of of a specific
    ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    _descriptions.assert_data(
        is_dense(data, dtype=dtype, layout=layout), f"{layout.name} numpy.ndarray", data, dtype=dtype
    )
    return data


@overload
def as_dense(data: DenseInRows, *, force_copy: bool = False) -> DenseInRows:
    ...


@overload
def as_dense(data: DenseInColumns, *, force_copy: bool = False) -> DenseInColumns:
    ...


@overload
def as_dense(data: _sparse.SparseInRows, *, force_copy: bool = False) -> DenseInRows:
    ...


@overload
def as_dense(data: _sparse.SparseInColumns, *, force_copy: bool = False) -> DenseInColumns:
    ...


@overload
def as_dense(data: _fake_sparse.spmatrix, *, force_copy: bool = False) -> Dense:
    ...


@overload
def as_dense(data: _frames.FrameInRows, *, force_copy: bool = False) -> DenseInRows:
    ...


@overload
def as_dense(data: _frames.FrameInColumns, *, force_copy: bool = False) -> DenseInColumns:
    ...


@overload
def as_dense(data: _unions.AnyData, *, force_copy: bool = False) -> Dense:
    ...


def as_dense(data: _unions.AnyData, *, force_copy: bool = False) -> Dense:
    """
    Access the internal 2D ``numpy`` array, if possible; otherwise, or if ``force_copy``, return a copy of the 2D data
    as a 2D ``numpy`` array.

    Accepts as input many data types that aren't even a `.Matrix`, such as nested lists (basically anything that
    ``numpy.array`` can use to construct a 2D array).

    If the input is a ``pandas.DataFrame``, this will only work if all the data in the frame has the same type.

    This will convert ``pandas`` strings (even if categorical) to proper ``numpy`` strings.
    """
    if isinstance(data, sp.spmatrix):
        return data.toarray()

    if isinstance(data, pd.DataFrame):
        data = data.values

    if force_copy or not isinstance(data, np.ndarray) or str(data.dtype) == "category" or isinstance(data, np.matrix):
        data = np.array(data)

    _descriptions.assert_data(data.ndim == 2, "any 2D data", data)
    return data  # type: ignore
