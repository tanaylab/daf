"""
The generic data types provided by ``numpy``, ``pandas`` and ``scipy.sparse`` allow for representing data in ways which
aren't optimal for processing. While most operations that take optimal data as input return "optimal" data, this isn't
always the case and the documentation of the relevant libraries is mostly silent on this issue.

At the same time, some code (especially C++ extension code) relies on the data being in one of the optimal formats. Even
code that technically works can become **much** slower when applied to non-optimal data. Therefore, all data fetched
from ``daf`` is always `.is_optimal`.

Examples of non-optimal data are strided ``numpy`` data, and ``scipy.sparse.spmatrix`` that isn't
``scipy.sparse.csr_matrix`` or ``scipy.sparse.csc_matrix``, or that is, but contains duplicate and/or unsorted indices.

The functions here allow to test whether data is in an optimal format, and allow converting data to an optimal format,
in-place if possible, optionally forcing a copy.

Most of the time, you can ignore these functions. However if you are writing serious processing code (e.g. a library),
they are useful in ensuring it will be correct and efficient.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import TypeVar
from typing import Union
from typing import overload

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import dense as _dense
from . import descriptions as _descriptions
from . import dtypes as _dtypes
from . import fake_pandas as _fake_pandas
from . import fake_sparse as _fake_sparse
from . import frames as _frames
from . import freezing as _freezing
from . import layouts as _layouts
from . import sparse as _sparse
from . import unions as _unions
from . import vectors as _vectors

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "KnownT",
    "is_optimal",
    "be_optimal",
    "optimize",
]


def is_optimal(data: _unions.Known) -> bool:  # pylint: disable=too-many-return-statements
    """
    Whether the ``data`` is in one of the supported ``daf`` types and also in an "optimal" format.
    """
    if isinstance(data, sp.spmatrix):
        return (
            isinstance(data, (sp.csr_matrix, sp.csc_matrix))
            and is_optimal(data.data)
            and is_optimal(data.indices)
            and is_optimal(data.indptr)
            and data.has_canonical_format
            and data.has_sorted_indices
            and data.indices.flags.writeable == data.indptr.flags.writeable == data.data.flags.writeable
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
            return _layouts.has_layout(data, _layouts.ROW_MAJOR) or _layouts.has_layout(data, _layouts.COLUMN_MAJOR)
        return False

    assert False, f"expected: known data, got: {_descriptions.data_description(data)}"


#: A ``TypeVar`` bound to `.Known`.
KnownT = TypeVar("KnownT", bound=_unions.Known)


def be_optimal(data: KnownT) -> KnownT:
    """
    Assert that some data is in "optimal" format and return it as-is.
    """
    _descriptions.assert_data(is_optimal(data), "optimal known data", data)
    return data


@overload
def optimize(
    data: _vectors.Vector, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _vectors.Vector:
    ...


@overload
def optimize(
    data: _dense.DenseInRows, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _dense.DenseInRows:
    ...


@overload
def optimize(
    data: _dense.DenseInColumns, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _dense.DenseInColumns:
    ...


@overload
def optimize(
    data: np.ndarray, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> Union[_vectors.Vector, _dense.Dense]:
    ...


@overload
def optimize(
    data: _fake_pandas.Series,
    *,
    force_copy: bool = False,
    preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR,
) -> _fake_pandas.Series:
    ...


@overload
def optimize(
    data: _frames.FrameInRows, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _frames.FrameInRows:
    ...


@overload
def optimize(
    data: _frames.FrameInColumns, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _frames.FrameInColumns:
    ...


@overload
def optimize(
    data: _fake_pandas.DataFrame,
    *,
    force_copy: bool = False,
    preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR,
) -> _frames.Frame:
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
    data: _fake_sparse.spmatrix,
    *,
    force_copy: bool = False,
    preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR,
) -> _sparse.Sparse:
    ...


def optimize(  # pylint: disable=too-many-branches,too-many-statements
    data: _unions.Known, *, force_copy: bool = False, preferred_layout: _layouts.AnyMajor = _layouts.ROW_MAJOR
) -> _unions.Proper:
    """
    Given some ``data`` in any `.Known2D` format, return it in a `.Proper` "optimal" format.

    If possible, and ``force_copy`` is not specified, this optimizes the data in-place. Otherwise, a copy is created.
    E.g. this can sort the indices of a CSR or CSC matrix in-place.

    If the data is 2D, and it has no clear layout, a copy will be created using the ``preferred_layout``. E.g. this will
    determine whether a COO matrix will be converted to a CSR or CSC matrix. For 1D data, this argument is ignored.

    If the data was copied and ``force_copy`` was not specified, and the data was `.is_frozen`, then so is the
    result; this ensures the code consuming the result will work regardless of whether a copy was done. If
    ``force_copy`` was specified, the result is never `.is_frozen`.

    This will fail if given a ``pandas.DataFrame`` with mixed data element types.

    .. note::

        This uses `.unfrozen` to modify a ``scipy.sparse.csr_matrix`` or a ``scipy.sparse.csc_matrix`` **in-place**,
        even if it is `.is_frozen` (unless ``force_copy`` is specified). This seems acceptable for in-memory sparse
        matrices, but will fail for read-only memory-mapped sparse matrices; this works because memory-mapped sparse
        matrices are only created by the `.FilesWriter`, which always writes them in the optimal format, so no in-place
        modification is done.
    """
    if isinstance(data, np.ndarray) and 1 <= data.ndim <= 2:
        if force_copy or not is_optimal(data):
            freeze = not force_copy and _freezing.is_frozen(data)
            data = np.array(data, order=preferred_layout.numpy_order)  # type: ignore
        else:
            freeze = False

    elif isinstance(data, pd.Series):
        if force_copy or not is_optimal(data):
            freeze = not force_copy and isinstance(data.values, np.ndarray) and _freezing.is_frozen(data)
            dtype = _dtypes.STR_DTYPE if _dtypes.has_dtype(data, _dtypes.STR_DTYPE) else data.dtype
            data = pd.Series(_vectors.as_vector(data, force_copy=True), index=data.index, dtype=dtype)
        else:
            freeze = False

    elif isinstance(data, pd.DataFrame):
        if not force_copy and is_optimal(data):
            freeze = False
        else:
            freeze = not force_copy and is_optimal(data.values) and _freezing.is_frozen(data.values)
            if len(set(data.dtypes)) == 1:
                dtype = str(data.dtypes[0])
                if _dtypes.is_dtype(dtype, _dtypes.STR_DTYPE):
                    dtype = _dtypes.STR_DTYPE
                data = pd.DataFrame(
                    np.array(data.values, order=preferred_layout.numpy_order),  # type: ignore
                    index=data.index,
                    columns=data.columns,
                    dtype=dtype,
                )
            else:
                data = pd.DataFrame({name: optimize(data[name]) for name in data}, index=data.index)

    elif isinstance(data, sp.spmatrix):
        if isinstance(data, (sp.csr_matrix, sp.csc_matrix)):
            klass = data.__class__
            freeze = not force_copy and _freezing.is_frozen(data)
            force_copy = (
                force_copy or not is_optimal(data.data) or not is_optimal(data.indices) or not is_optimal(data.indptr)
            )

        elif preferred_layout == _layouts.ROW_MAJOR:
            klass = sp.csr_matrix
            freeze = not force_copy
            force_copy = True

        else:
            assert preferred_layout == _layouts.COLUMN_MAJOR
            klass = sp.csc_matrix
            freeze = not force_copy
            force_copy = True

        if force_copy:
            data = klass(data)
        else:
            freeze = False

        with _freezing.unfrozen(data) as melted:  # type: ignore
            melted.sum_duplicates()  # type: ignore
            melted.sort_indices()  # type: ignore

    else:
        assert False, f"expected known data, got: {_descriptions.data_description(data)}"

    if freeze:
        data = _freezing.freeze(data)  # type: ignore
    return data  # type: ignore
