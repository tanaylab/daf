"""
The types here describe 2D data without names (that is, not in a ``pandas.DataFrame``), which is how 2D data is stored
in ``daf``. Currently there are only two such types:

* `.typing.dense.Dense` is a 2D ``numpy.ndarray`` matrix.
* `.typing.sparse.Sparse` is a compressed sparse matrix (either ``scipy.sparse.csr_matrix`` or
  ``scipy.sparse.csc_matrix``).

The `.Matrix` type annotations is their union, that is, allows for "any" 2D data without names. While this isn't very
useful to directly perform operation on, it is very useful as the return type of accessing 2D data stored in ``daf``, as
the caller has no control over whether the data was stored as sparse, and forcing it to be dense would not be practical
for large data sets.

.. note::

    The `.Matrix` type should be only be directly used in computations with **great care**, as some operations are
    subtly different for ``numpy`` 2D arrays and ``scipy.sparse`` compressed matrices. It is typically better to use one
    of the concrete types instead.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union
from typing import overload

try:
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import dense as _dense
from . import descriptions as _descriptions
from . import dtypes as _dtypes
from . import fake_pandas as _fake_pandas
from . import fake_sparse as _fake_sparse
from . import frames as _frames
from . import layouts as _layouts
from . import sparse as _sparse
from . import unions as _unions

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "Matrix",
    "is_matrix",
    "be_matrix",
    "as_matrix",
    "MatrixInRows",
    "is_matrix_in_rows",
    "be_matrix_in_rows",
    "MatrixInColumns",
    "is_matrix_in_columns",
    "be_matrix_in_columns",
]

#: Any 2D data in `.ROW_MAJOR` layout, without names.
MatrixInRows = Union[_dense.DenseInRows, _sparse.SparseInRows]


def is_matrix_in_rows(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> TypeGuard[MatrixInRows]:
    """
    Assert that some ``data`` is a `.MatrixInRows`, optionally only of some ``dtype``, optionally only of some
    ``shape``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return _dense.is_dense_in_rows(data, dtype=dtype, shape=shape) or _sparse.is_sparse_in_rows(
        data, dtype=dtype, shape=shape
    )


def be_matrix_in_rows(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> MatrixInRows:
    """
    Assert that some ``data`` is a `.MatrixInRows`, optionally only of some ``dtype``, optionally only of some
    ``shape``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(
        is_matrix_in_rows(data, dtype=dtype, shape=shape), "row-major matrix", data, dtype=dtype, shape=shape
    )
    return data


#: Any 2D data in `.COLUMN_MAJOR` layout, without names.
MatrixInColumns = Union[_dense.DenseInColumns, _sparse.SparseInColumns]


def is_matrix_in_columns(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> TypeGuard[MatrixInColumns]:
    """
    Assert that some ``data`` is a `.MatrixInColumns`, optionally only of some ``dtype``, optionally only of some
    ``shape``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return _dense.is_dense_in_columns(data, dtype=dtype, shape=shape) or _sparse.is_sparse_in_columns(
        data, dtype=dtype, shape=shape
    )


def be_matrix_in_columns(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> MatrixInColumns:
    """
    Assert that some ``data`` is a `.MatrixInColumns`, optionally only of some ``dtype``, optionally only of some
    ``shape``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(
        is_matrix_in_columns(data, dtype=dtype, shape=shape), "column-major matrix", data, dtype=dtype, shape=shape
    )
    return data


#: Any 2D data, in either `.ROW_MAJOR` or `.COLUMN_MAJOR` layout, without names.
#:
#: .. note::
#:
#:  This is **not** to be confused with the deprecated ``numpy.matrix`` type which must never be used.
Matrix = Union[_dense.Dense, _sparse.Sparse]


def is_matrix(
    data: Any,
    *,
    dtype: Optional[_dtypes.DTypes] = None,
    shape: Optional[Tuple[int, int]] = None,
    layout: Optional[_layouts.AnyMajor] = None,
) -> TypeGuard[Matrix]:
    """
    Assert that some ``data`` is a `.Matrix`, optionally only of some ``dtype``, optionally only of some ``shape``,
    optionally only of some ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return _dense.is_dense(data, dtype=dtype, shape=shape, layout=layout) or _sparse.is_sparse(
        data, dtype=dtype, shape=shape, layout=layout
    )


def be_matrix(
    data: Any,
    *,
    dtype: Optional[_dtypes.DTypes] = None,
    shape: Optional[Tuple[int, int]] = None,
    layout: Optional[_layouts.AnyMajor] = None,
) -> Matrix:
    """
    Assert that some ``data`` is a `.Matrix`, optionally only of some ``dtype``, optionally only of some ``shape``,
    optionally only of some ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access

    # pylint: disable=duplicate-code

    _descriptions.assert_data(
        is_matrix(data, dtype=dtype, shape=shape, layout=layout),
        f"{layout.name} matrix",
        data,
        dtype=dtype,
        shape=shape,
    )

    # pylint: enable=duplicate-code

    return data


@overload
def as_matrix(data: _dense.DenseInRows, *, force_copy: bool = False) -> _dense.DenseInRows:
    ...


@overload
def as_matrix(data: _dense.DenseInColumns, *, force_copy: bool = False) -> _dense.DenseInColumns:
    ...


@overload
def as_matrix(data: _sparse.SparseInRows, *, force_copy: bool = False) -> _sparse.SparseInRows:
    ...


@overload
def as_matrix(data: _sparse.SparseInColumns, *, force_copy: bool = False) -> _sparse.SparseInColumns:
    ...


@overload
def as_matrix(data: _fake_sparse.spmatrix, *, force_copy: bool = False) -> _sparse.Sparse:
    ...


@overload
def as_matrix(data: _frames.FrameInRows, *, force_copy: bool = False) -> _dense.DenseInRows:
    ...


@overload
def as_matrix(data: _frames.FrameInColumns, *, force_copy: bool = False) -> _dense.DenseInColumns:
    ...


@overload
def as_matrix(data: _fake_pandas.DataFrame, *, force_copy: bool = False) -> _dense.Dense:
    ...


@overload
def as_matrix(data: _unions.AnyData, *, force_copy: bool = False) -> Matrix:
    ...


def as_matrix(data: _unions.AnyData, *, force_copy: bool = False) -> Matrix:
    """
    Access the internal 2D matrix, if possible; otherwise, or if ``force_copy``, return a copy of the 2D data as a
    ``numpy`` array.

    If the input is a ``pandas.DataFrame``, this will only work if all the data in the frame has the same type.
    """
    # In case someone sneaks a sparse matrix into a frame, which doesn't really work, but be nice...
    if isinstance(data, pd.DataFrame):
        data = data.values

    if isinstance(data, sp.spmatrix):
        if not isinstance(data, _layouts._ANY_MAJOR.sparse_class):  # pylint: disable=protected-access
            return _layouts.ROW_MAJOR.sparse_class(data)
        if force_copy:
            return data.copy()
        return data

    return _dense.as_dense(data, force_copy=force_copy)
