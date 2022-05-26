"""
2D data can be arranged in many layouts. The choice of layout is, in theory, transparent to the code, as one can still
access any element using its row and column index. In practice, the choice of layout is crucial to get reasonable
performance, as accessing data "against the grain" results in orders of magnitude loss of performance.

We restrict 2D data stored in ``daf`` to two laytouts: :py:obj:`~ROW_MAJOR` and :py:obj:`~COLUMN_MAJOR`. This applies
both to dense data and also to sparse data (where "row-major" data means "CSR" and "column-major" means "CSC").

We provide explicit data type annotations expressing the distinction between these layouts by suffixing the base type
with ``InRows`` or ``InColumns`` (e.g., :py:obj:`~daf.typing.matrices.TableInRows` vs.
:py:obj:`~daf.typing.matrices.TableInColumns`). This makes it easier to ensure that operations get data in the correct
layout, e.g. summing each row of row-major data would be much, much faster than summing the rows of column-major data.
Arguably clever implementation of the algorithms could mitigate this to some degree, but libraries almost never do these
difficult optimizations.

The code here provides functions to test for the layout of 2D data and to convert data to the desired layout, providing
a somewhat more efficient algorithm to do so than is provided by ``numpy``.

Of course you are free to just ignore the layout (or the type annotations altogether). This may be acceptable for very
small data sets, but for "serious" code working on non-trivial data, controlling the 2D data layout is vital.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import ceil
from math import sqrt
from typing import Any
from typing import Union
from typing import overload

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import array2d as _array2d  # pylint: disable=cyclic-import
from . import dense as _dense  # pylint: disable=cyclic-import
from . import frames as _frames
from . import matrices as _matrices  # pylint: disable=cyclic-import
from . import sparse as _sparse  # pylint: disable=cyclic-import
from . import tables as _tables  # pylint: disable=cyclic-import

__all__ = [
    "AnyMajor",
    "ROW_MAJOR",
    "COLUMN_MAJOR",
    "as_layout",
]


class AnyMajor:
    """
    Allow for either row-major or column-major matrix layout.

    This does **not** allow for other (e.g., strided or COO) layouts.
    """

    #: Name for messages.
    name = "any-major"

    #: The ``numpy`` order for the layout.
    numpy_order = "?"

    #: The ``numpy`` flag for the layout.
    numpy_flag = "?"

    #: The axis which should be contiguous for the layout.
    contiguous_axis = -1

    #: The sparse matrix class for this layout.
    sparse_class = (sp.csr_matrix, sp.csc_matrix)

    #: The name of the sparse matrix class for this layout.
    sparse_class_name = "scipy.sparse.csr/csc_matrix"

    def __eq__(self, other: Any) -> bool:
        return id(self.__class__) == id(other.__class__)

    def __ne__(self, other: Any) -> bool:
        return id(self.__class__) != id(other.__class__)

    def __str__(self) -> str:
        return self.name

    def is_layout_of(self, matrix: _matrices.Matrix) -> bool:
        """
        Test whether the given ``matrix`` is in this layout.

        .. note::

            A dense matrix with one row or one column is considered to be both row-major and column-major, if its
            elements are contiguous in memory.
        """
        if isinstance(matrix, self.sparse_class):
            return True
        if isinstance(matrix, sp.spmatrix):
            return False

        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.values

        if not isinstance(matrix, np.ndarray) or isinstance(matrix, np.matrix):
            return False

        if self.contiguous_axis >= 0:
            return matrix.flags[self.numpy_flag] or matrix.strides[self.contiguous_axis] == matrix.dtype.itemsize

        return (
            matrix.flags["C_CONTIGUOUS"]
            or matrix.flags["F_CONTIGUOUS"]
            or matrix.strides[0] == matrix.dtype.itemsize
            or matrix.strides[1] == matrix.dtype.itemsize
        )


#: Require either row-major or column-major layout.
_ANY_MAJOR: AnyMajor = AnyMajor()


class RowMajor(AnyMajor):  # pylint: disable=too-few-public-methods
    """
    Require row-major layout.

    In this layout, the elements of each row are stored contiguously in memory. For sparse matrices,
    only the non-zero elements are stored ("CSR" format).
    """

    name = "row-major"
    numpy_order = "C"
    numpy_flag = "C_CONTIGUOUS"
    contiguous_axis = 1
    sparse_class = sp.csr_matrix
    sparse_class_name = "scipy.sparse.csr_matrix"


#: Require row-major layout.
ROW_MAJOR: RowMajor = RowMajor()


class ColumnMajor(AnyMajor):  # pylint: disable=too-few-public-methods
    """
    Require column-major layout.

    In this layout, the elements of each column are stored contiguously in memory. For sparse matrices,
    only the non-zero elements are stored ("CSC" format).
    """

    name = "column-major"
    numpy_order = "F"
    numpy_flag = "F_CONTIGUOUS"
    contiguous_axis = 0
    sparse_class = sp.csc_matrix
    sparse_class_name = "scipy.sparse.csc_matrix"


#: Require column-major layout.
COLUMN_MAJOR: ColumnMajor = ColumnMajor()


@overload
def as_layout(
    matrix: _array2d.Array2D,
    layout: RowMajor,
    force_copy: bool = False,
    max_workers: int = 1,
    block_size: int = 8 * 1024 * 1024,
) -> _array2d.ArrayInRows:
    ...


@overload
def as_layout(
    matrix: _array2d.Array2D,
    layout: ColumnMajor,
    force_copy: bool = False,
    max_workers: int = 1,
    block_size: int = 8 * 1024 * 1024,
) -> _array2d.ArrayInColumns:
    ...


@overload
def as_layout(
    matrix: _sparse.Sparse,
    layout: RowMajor,
    force_copy: bool = False,
    max_workers: int = 1,
    block_size: int = 8 * 1024 * 1024,
) -> _sparse.SparseInRows:
    ...


@overload
def as_layout(
    matrix: _sparse.Sparse,
    layout: ColumnMajor,
    force_copy: bool = False,
    max_workers: int = 1,
    block_size: int = 8 * 1024 * 1024,
) -> _sparse.SparseInColumns:
    ...


@overload
def as_layout(
    matrix: _tables.Table,
    layout: RowMajor,
    force_copy: bool = False,
    max_workers: int = 1,
    block_size: int = 8 * 1024 * 1024,
) -> _tables.TableInRows:
    ...


@overload
def as_layout(
    matrix: _tables.Table,
    layout: ColumnMajor,
    force_copy: bool = False,
    max_workers: int = 1,
    block_size: int = 8 * 1024 * 1024,
) -> _tables.TableInColumns:
    ...


def as_layout(
    matrix: Union[_matrices.Matrix, sp.spmatrix],
    layout: AnyMajor,
    force_copy: bool = False,
    max_workers: int = 1,
    block_size: int = 8 * 1024 * 1024,
) -> _matrices.Matrix:
    """
    Access the data in a specific layout.

    If ``force_copy``, return a copy even if the data is already in the required layout.

    If we need to actually re-layout the data, and it is in a dense format, and it is "large", it may take a long time
    to do so without parallelization. The code here is able to use multiple threads in such a case, up to
    ``max_workers``. Since we might be invoked from within some parallel code, there's no way for the code here to
    figure out what is the available number of workers we can use, so we default to the only safe value of just one.

    It also turns out that for large dense data it is more efficient to work on the data in blocks that fit a
    "reasonable" HW cache level. The code here uses a default ``block_size`` of 8 megabytes, which seems to work well in
    CPUs circa 2022. This optimization should really have been in ``numpy`` itself.
    """
    if _matrices.is_matrix(matrix, layout=layout):
        if force_copy:
            return _matrices.matrix_copy(matrix)
        return matrix

    if isinstance(matrix, sp.spmatrix):
        if layout == ROW_MAJOR:
            return sp.csr_matrix(matrix)
        assert layout == COLUMN_MAJOR
        return sp.csc_matrix(matrix)

    assert _dense.is_dense(matrix)
    array2d = _relayout_array2d(_array2d.as_array2d(matrix), layout, max_workers, block_size)

    if _array2d.is_array2d(matrix):
        return array2d

    assert _frames.is_frame(matrix)
    return pd.DataFrame(array2d, index=matrix.index, columns=matrix.columns)


@overload
def _relayout_array2d(
    array2d: _array2d.Array2D, layout: RowMajor, max_workers: int, block_size: int
) -> _array2d.ArrayInRows:
    ...


@overload
def _relayout_array2d(
    array2d: _array2d.Array2D, layout: ColumnMajor, max_workers: int, block_size: int
) -> _array2d.ArrayInColumns:
    ...


@overload
def _relayout_array2d(
    array2d: _array2d.Array2D, layout: AnyMajor, max_workers: int, block_size: int
) -> _array2d.Array2D:
    ...


def _relayout_array2d(
    array2d: _array2d.Array2D, layout: AnyMajor, max_workers: int, block_size: int
) -> _array2d.Array2D:
    block_elements = int(sqrt(block_size / array2d.dtype.itemsize))

    row_elements = array2d.shape[0]
    column_elements = array2d.shape[1]

    row_blocks = int(ceil(row_elements / block_elements))
    column_blocks = int(ceil(column_elements / block_elements))

    result = np.empty(array2d.shape, dtype=array2d.dtype, order=layout.numpy_order)  # type: ignore

    def _relayout_block(block: int) -> None:
        column_block = block // row_blocks
        row_block = block % row_blocks

        start_row = int(round(row_block * row_elements / row_blocks))
        stop_row = int(round((row_block + 1) * row_elements / row_blocks))

        start_column = int(round(column_block * column_elements / column_blocks))
        stop_column = int(round((column_block + 1) * column_elements / column_blocks))

        result[start_row:stop_row, start_column:stop_column] = array2d[start_row:stop_row, start_column:stop_column]

    blocks_count = row_blocks * column_blocks
    max_workers = min(max_workers, blocks_count)

    if max_workers == 1:
        for block in range(blocks_count):
            _relayout_block(block)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(_relayout_block, range(blocks_count)):
                pass

    assert layout.is_layout_of(result)  # type: ignore
    return result  # type: ignore
