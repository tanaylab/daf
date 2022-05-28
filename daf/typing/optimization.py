"""
The generic data types provided by ``numpy``, ``pandas`` and ``scipy.sparse`` allow for representing data in ways which
aren't optimal for processing. While most operations that take optimal data as input return "optimal" data, this isn't
always the case and the documentation of the relevant libraries is mostly silent on this issue.

At the same time, some code (especially C++ extension code) relies on the data being in one of the optimal formats. Even
code that technically works will become much slower when applied to non-optimal data. Therefore, ``daf`` refuses to
store non-optimal data.

Examples of non-optimal data are strided ``numpy`` data, and ``scipy.sparse`` data that contains duplicate and/or
unsorted indices.

The functions here allow to test whether data is in an optimal format and allow converting data to an optimal format,
in-place if possible, optionally forcing a copy.

Most of the time, you can ignore these functions, until you see ``daf`` complaining about seeing non-optimal data, in
which case you'll need to inject an `.optimize` call into your code. If you aren't sure how the non-optimal data was
created, use `.is_optimal` and/or `.be_optimal` to isolate the offending operations.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import TypeVar
from typing import overload

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import array1d as _array1d
from . import array2d as _array2d
from . import descriptions as _descriptions  # pylint: disable=cyclic-import
from . import fake_pandas as _fake_pandas
from . import fake_sparse as _fake_sparse
from . import layouts as _layouts  # pylint: disable=cyclic-import
from . import series as _series
from . import sparse as _sparse
from . import tables as _tables

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "is_optimal",
    "be_optimal",
    "optimize",
]


def is_optimal(data: Any) -> bool:  # pylint: disable=too-many-return-statements
    """
    Whether the ``data`` is in one of the supported ``daf`` types and also in an "optimal" format.
    """
    if isinstance(data, sp.spmatrix):
        return (
            isinstance(data, (sp.csr_matrix, sp.csc_matrix))
            and data.has_canonical_format
            and data.has_sorted_indices
            and data.data.strides[0] == data.data.dtype.itemsize
            and data.indices.strides[0] == data.indices.dtype.itemsize
            and data.indptr.strides[0] == data.indptr.dtype.itemsize
        )

    if isinstance(data, pd.DataFrame):
        for dtype in data.dtypes:
            if str(dtype) == "category":
                return False
        data = data.values
        if not isinstance(data, np.ndarray):
            return False

    if isinstance(data, pd.Series):
        if str(data.dtype) == "category":
            return False
        data = data.values
        if not isinstance(data, np.ndarray):
            return False

    if isinstance(data, np.ndarray):
        if str(data.dtype) == "category" or isinstance(data, np.matrix):
            return False
        if data.ndim == 1:
            return data.strides[0] == data.dtype.itemsize
        if data.ndim == 2:
            return _layouts.ROW_MAJOR.is_layout_of(data) or _layouts.COLUMN_MAJOR.is_layout_of(data)  # type: ignore
        return False

    assert False, f"expected a matrix or a vector, got {_descriptions.data_description(data)}"


T = TypeVar("T")


def be_optimal(data: T) -> T:
    """
    Assert that some data is in "optimal" format and return it as-is.
    """
    _descriptions.assert_data(is_optimal(data), "optimal matrix or vector", data, None)
    return data


@overload
def optimize(
    data: _array1d.Array1D, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _array1d.Array1D:
    ...


@overload
def optimize(
    data: _series.Series, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _series.Series:
    ...


@overload
def optimize(
    data: _array2d.ArrayInRows, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _array2d.ArrayInRows:
    ...


@overload
def optimize(
    data: _array2d.ArrayInColumns, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _array2d.ArrayInColumns:
    ...


@overload
def optimize(
    data: np.ndarray, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> np.ndarray:
    ...


@overload
def optimize(
    data: _tables.TableInRows, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _tables.TableInRows:
    ...


@overload
def optimize(
    data: _tables.TableInColumns, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _tables.TableInColumns:
    ...


@overload
def optimize(
    data: _fake_pandas.PandasFrame,
    *,
    force_copy: bool = False,
    preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR,
) -> _fake_pandas.PandasFrame:
    ...


@overload
def optimize(
    data: _sparse.SparseInRows, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _sparse.SparseInRows:
    ...


@overload
def optimize(
    data: _sparse.SparseInColumns, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _sparse.SparseInColumns:
    ...


@overload
def optimize(
    data: _fake_sparse.SparseMatrix,
    *,
    force_copy: bool = False,
    preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR,
) -> _fake_sparse.SparseMatrix:
    ...


@overload
def optimize(
    data: sp.spmatrix, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> sp.spmatrix:
    ...


def optimize(data: Any, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR) -> Any:
    """
    Given some ``data`` in any (reasonable) format, return it in a supported, "optimal" format.

    If possible, and ``force_copy`` is not specified, this optimizes the data in-place. Otherwise, a copy is created.
    E.g. this can sort the indices of a CSR or CSC matrix in-place.

    If the data is a matrix, and it has no clear layout, a copy will be created using the ``preferred_layout``. E.g.
    this will determine whether a COO matrix will be converted to a CSR or CSC matrix. For vector data, this argument is
    ignored.
    """
    if isinstance(data, np.ndarray) and 1 <= data.ndim <= 2:
        if force_copy or not is_optimal(data):
            return np.array(data, order=preferred_layout.numpy_order)  # type: ignore
        return data

    if isinstance(data, pd.Series):
        if force_copy or not is_optimal(data):
            data = pd.Series(_array1d.as_array1d(data, force_copy=True), index=data.index)
        return data

    if isinstance(data, pd.DataFrame):
        if force_copy or not is_optimal(data):
            data = pd.DataFrame(
                np.array(data.values, order=preferred_layout.numpy_order),  # type: ignore
                index=data.index,
                columns=data.columns,
            )
        return data

    if isinstance(data, sp.spmatrix):
        if isinstance(data, (sp.csr_matrix, sp.csc_matrix)):
            klass = data.__class__
            force_copy = (
                force_copy or not is_optimal(data.data) or not is_optimal(data.indices) or not is_optimal(data.indptr)
            )
        elif preferred_layout == _layouts.ROW_MAJOR:
            force_copy = True
            klass = sp.csr_matrix
        else:
            assert preferred_layout == _layouts.COLUMN_MAJOR
            force_copy = True
            klass = sp.csc_matrix

        if force_copy:
            data = klass(data)

        data.sum_duplicates()
        data.sort_indices()
        return data

    assert False, f"expected a matrix or a vector, got {_descriptions.data_description(data)}"
