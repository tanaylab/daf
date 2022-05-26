"""
In contrast to 1D data, 2D data can be represented in too many data types and formats. In ``daf`` we only allow storing
a restricted set of types. These were chosen to cover "most" needs while minimizing the burden on the code (to pick
different code paths for the different types).

* :py:obj:`~Array2D` is a two-dimensional ``numpy.ndarray``.
* :py:obj:`~Sparse` is a compressed sparse matrix (``scipy.sparse.csr_matrix`` or ``scipy.sparse.csc_matrix``).
* :py:obj:`~Table` is a ``pandas.DataFrame`` which contains an ``Array2D`` of a single element type, with indices of
  names for the rows and the columns.

.. note::

    We also provide :py:obj:`~Frame` to represent a ``pandas.DataFrame`` which is a collection of 1D data columns of
    different data types, which is conceptually (and operationally) very different from a 2D data matrix that has a
    homogeneous element data type. That is, :py:obj:`~Frame` is **not** a :py:obj:`~Matrix`.

In addition, we provide the following 2D data union types:

* :py:obj:`~Grid` is a union of ``Array2D`` and ``Sparse``, that is, plain data without names.
* :py:obj:`~Dense` is a union of ``Array2D`` and ``Table``, that is, dense data without compression.
* :py:obj:`~Matrix`, defined here, is the most general 2D data, a union of all the above.

Since layout of 2D data is crucial for performance and impacts code paths, each of the above types can be suffixed with
``InRows`` or ``InColumns`` to indicate that it is stored in :py:obj:`~daf.typing.layouts.ROW_MAJOR` or
:py:obj:`~daf.typing.layouts.COLUMN_MAJOR` layout. This *matters* as performing computations on the wrong layout will
be *drastically* inefficient for non-trivial sizes; see the :py:obj:`~daf.typing.layouts` module for details, and
:py:obj:`~matrix_copy` for a safe way to copy 2D data while preserving its layout.

.. note::

    It seems that ``pandas`` isn't reliable when it comes to the 2D layout of tables of strings. In general it doesn't
    even have a concept of a string data type, so it falls back to viewing them as objects, which means it thinks the
    table is actually a frame (e.g., it forces it to be in column-major layout). At least this happens in some
    ``pandas`` versions - this doesn't seem to be documented well (or at all). You are therefore advised not to try to
    use row-major tables of strings, unless you are willing to deal with the subtle undocumented incompatibilities
    between different ``pandas`` versions. The layout of tables of numbers seems to work as expected across all
    versions, though.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Any
from typing import Collection
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

from . import array2d as _array2d
from . import descriptions as _descriptions
from . import dtypes as _dtypes
from . import layouts as _layouts  # pylint: disable=cyclic-import
from . import sparse as _sparse
from . import tables as _tables

# pylint: enable=duplicate-code

__all__ = [
    "Matrix",
    "is_matrix",
    "be_matrix",
    "MatrixInRows",
    "is_matrix_in_rows",
    "be_matrix_in_rows",
    "MatrixInColumns",
    "is_matrix_in_columns",
    "be_matrix_in_columns",
    "matrix_copy",
]

#: Any 2D data in row-major layout.
MatrixInRows = Union["_array2d.ArrayInRows", _sparse.SparseInRows, _tables.TableInRows]


def is_matrix_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[MatrixInRows]:
    """
    Assert that some ``data`` is an :py:obj:`~MatrixInRows`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    return (
        _array2d.is_array_in_rows(data, dtype=dtype)
        or _sparse.is_sparse_in_rows(data, dtype=dtype)
        or _tables.is_table_in_rows(data, dtype=dtype)
    )


def be_matrix_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> MatrixInRows:
    """
    Assert that some ``data`` is a :py:obj:`~MatrixInRows`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_matrix_in_rows(data, dtype=dtype), "row-major matrix", data, dtype)
    return data


#: Any 2D data in column-major layout.
MatrixInColumns = Union["_array2d.ArrayInColumns", _sparse.SparseInColumns, _tables.TableInColumns]


def is_matrix_in_columns(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None
) -> TypeGuard[MatrixInColumns]:
    """
    Assert that some ``data`` is an :py:obj:`~MatrixInColumns`, optionally only of some ``dtype``, and return it as
    such for ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    return (
        _array2d.is_array_in_columns(data, dtype=dtype)
        or _sparse.is_sparse_in_columns(data, dtype=dtype)
        or _tables.is_table_in_columns(data, dtype=dtype)
    )


def be_matrix_in_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> MatrixInColumns:
    """
    Assert that some ``data`` is a :py:obj:`~MatrixInColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_matrix_in_columns(data, dtype=dtype), "column-major matrix", data, dtype)
    return data


#: Any 2D data.
Matrix = Union["_array2d.Array2D", _sparse.Sparse, _tables.Table]


def is_matrix(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None, layout: Optional[_layouts.AnyMajor] = None
) -> TypeGuard[Matrix]:
    """
    Assert that some ``data`` is an :py:obj:`~Matrix`, optionally only of some ``dtype``, optionally of some
    ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    return (
        _array2d.is_array2d(data, dtype=dtype, layout=layout)
        or _sparse.is_sparse(data, dtype=dtype, layout=layout)
        or _tables.is_table(data, dtype=dtype, layout=layout)
    )


def be_matrix(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None, layout: Optional[_layouts.AnyMajor] = None
) -> Matrix:
    """
    Assert that some ``data`` is a :py:obj:`~Matrix`, optionally only of some ``dtype``, optionally of some ``layout``,
    and return it as such for ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    _descriptions.assert_data(is_matrix(data, dtype=dtype, layout=layout), f"{layout.name} matrix", data, dtype)
    return data


@overload
def matrix_copy(data: _array2d.ArrayInRows) -> _array2d.ArrayInRows:
    ...


@overload
def matrix_copy(data: _array2d.ArrayInColumns) -> _array2d.ArrayInColumns:
    ...


@overload
def matrix_copy(data: _sparse.SparseInRows) -> _sparse.SparseInRows:
    ...


@overload
def matrix_copy(data: _sparse.SparseInColumns) -> _sparse.SparseInColumns:
    ...


@overload
def matrix_copy(data: _tables.TableInRows) -> _tables.TableInRows:
    ...


@overload
def matrix_copy(data: _tables.TableInColumns) -> _tables.TableInColumns:
    ...


def matrix_copy(data: Matrix) -> Matrix:
    """
    Create a copy of a matrix.

    All the matrix data types (``numpy.ndarray``, ``scipy.sparse``, ``pandas.Frame``) have a ``copy()`` method, so you
    would think one can just write ``matrix.copy()`` and be done and that is *almost* true except that in their infinite
    wisdom ``numpy`` will always create the copy in row-major layout, and ``pandas`` will always create the copy in
    column-major layout, because "reasons". In fact in some (older) versions of ``pandas``/``numpy``, it seems this
    isn't even possible to achieve a row-major frame of strings.

    The code here will give you a proper copy of the data in the same layout as the original. Sigh.
    """
    if isinstance(data, np.ndarray):
        array2d_copy: _array2d.Array2D = np.array(data)  # type: ignore
        for layout in (_layouts.ROW_MAJOR, _layouts.COLUMN_MAJOR):
            assert layout.is_layout_of(array2d_copy) == layout.is_layout_of(data)
        return array2d_copy

    if isinstance(data, pd.DataFrame) and isinstance(data.values, np.ndarray):
        values_copy: _array2d.Array2D = matrix_copy(data.values)  # type: ignore
        frame_copy = pd.DataFrame(values_copy, index=data.index, columns=data.columns)
        # For ``object`` data, older ``pandas`` insists on column-major order no matter what.
        # This means we have needless duplicated the array above, since ``pandas`` will re-copy (and re-layout) it.
        # Since newer ``pandas`` seems to be doing the right thing, we keep the code above. Sigh.
        if not _dtypes.is_dtype(data.values.dtype, _dtypes.STR_DTYPE):
            for layout in (_layouts.ROW_MAJOR, _layouts.COLUMN_MAJOR):
                assert layout.is_layout_of(frame_copy.values) == layout.is_layout_of(data.values)  # type: ignore
        return frame_copy

    # This will keep non-optimal (e.g. COO data) as non-optimal.
    # Otherwise we'd have called this ``as_matrix`` with a ``force_copy`` instead of ``matrix_copy``.
    # However ``matrix_copy`` seems more useful in practice.
    _descriptions.assert_data(isinstance(data, sp.spmatrix), "matrix", data, None)
    return data.copy()
