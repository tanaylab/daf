"""
The types here describe 2D data without names (that is, not in a ``pandas.DataFrame``). Currently there are
only two:

* :py:const:`Array2D` is a two-dimensional ``numpy.ndarray``.
* :py:const:`Sparse` is a compressed sparse matrix (``scipy.sparse.csr_matrix`` or ``scipy.sparse.csc_matrix``).

The :py:const:`Grid` type annotations is their union, that is, allows for "any" 2D data without names. While this isn't
very useful to directly perform operation on, it is very useful as the return type of accessing 2D data stored in
``daf``, as the caller has no control over whether the data was stored as sparse, and forcing it to be dense would not
be practical for large data sets.

.. note::

    The :py:const:`Grid` type should be only be directly used in computations with **great care**, as some operations
    are subtly different for ``numpy`` 2D arrays and ``scipy.sparse`` compressed matrices. It is typically better to use
    one of the concrete types instead.
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

import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import array2d as _array2d
from . import descriptions as _descriptions
from . import layouts as _layouts  # pylint: disable=cyclic-import
from . import matrices as _matrices
from . import sparse as _sparse

# pylint: enable=duplicate-code

__all__ = [
    "Grid",
    "is_grid",
    "be_grid",
    "as_grid",
    "GridInRows",
    "is_grid_in_rows",
    "be_grid_in_rows",
    "GridInColumns",
    "is_grid_in_columns",
    "be_grid_in_columns",
]

#: Any 2D data in row-major layout.
GridInRows = Union[_array2d.ArrayInRows, _sparse.SparseInRows]


def is_grid_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[GridInRows]:
    """
    Assert that some ``data`` is an :py:const:`GridInRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    return _array2d.is_array_in_rows(data, dtype=dtype) or _sparse.is_sparse_in_rows(data, dtype=dtype)


def be_grid_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> GridInRows:
    """
    Assert that some ``data`` is a :py:const:`GridInRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_grid_in_rows(data, dtype=dtype), "row-major grid", data, dtype)
    return data


#: Any 2D data in column-major layout.
GridInColumns = Union[_array2d.ArrayInColumns, _sparse.SparseInColumns]


def is_grid_in_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[GridInColumns]:
    """
    Assert that some ``data`` is an :py:const:`GridInColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    return _array2d.is_array_in_columns(data, dtype=dtype) or _sparse.is_sparse_in_columns(data, dtype=dtype)


def be_grid_in_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> GridInColumns:
    """
    Assert that some ``data`` is a :py:const:`GridInColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_grid_in_columns(data, dtype=dtype), "column-major grid", data, dtype)
    return data


#: Any 2D data without names.
Grid = Union[_array2d.Array2D, _sparse.Sparse]


def is_grid(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None, layout: Optional[_layouts.AnyMajor] = None
) -> TypeGuard[Grid]:
    """
    Assert that some ``data`` is an :py:const:`Grid`, optionally only of some ``dtype``, optionally only of some
    ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    return _array2d.is_array2d(data, dtype=dtype, layout=layout) or _sparse.is_sparse(data, dtype=dtype, layout=layout)


def be_grid(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None, layout: Optional[_layouts.AnyMajor] = None
) -> Grid:
    """
    Assert that some ``data`` is a :py:const:`Grid`, optionally only of some ``dtype``, optionally only of some
    ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    _descriptions.assert_data(is_grid(data, dtype=dtype, layout=layout), f"{layout.name} grid", data, dtype)
    return data


@overload
def as_grid(data: _matrices.MatrixInRows, *, force_copy: bool = False) -> GridInRows:
    ...


@overload
def as_grid(data: _matrices.MatrixInColumns, *, force_copy: bool = False) -> GridInColumns:
    ...


@overload
def as_grid(data: _array2d.Data2D, *, force_copy: bool = False) -> Grid:
    ...


def as_grid(data: Any, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR) -> Grid:
    """
    Access the internal 2D grid, if possible; otherwise, or if ``force_copy``, return a copy of the 2D data as a
    ``numpy`` array.

    If the input is a ``pandas.DataFrame``, this will only work if all the data in the frame has the same type (that is,
    for a :py:const:`Table`).
    """
    # In case someone sneaks a sparse matrix into a frame, which doesn't really work, but be nice...
    if isinstance(data, pd.DataFrame):
        data = data.values

    if isinstance(data, sp.spmatrix):
        if not isinstance(data, _layouts._ANY_MAJOR.sparse_class):  # pylint: disable=protected-access
            return preferred_layout.sparse_class(data)  # type: ignore
        if force_copy:
            return data.copy()
        return data

    return _array2d.as_array2d(data, force_copy=force_copy)
