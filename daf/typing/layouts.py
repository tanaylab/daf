"""
2D data can be arranged in many layouts. The choice of layout is, in theory, transparent to the code, as one can still
access any element using its row and column index. In practice, the choice of layout is crucial to get reasonable
performance, as accessing data "against the grain" results in orders of magnitude loss of performance.

We restrict 2D data stored in ``daf`` to two layouts: `.ROW_MAJOR` and `.COLUMN_MAJOR`. This applies both to dense data
and also to sparse data (where "row-major" data means "CSR" and "column-major" means "CSC").

We provide explicit data type annotations expressing the distinction between these layouts by suffixing the base type
with ``InRows`` or ``InColumns`` (e.g., `.FrameInRows` vs. `.FrameInColumns`). This makes it easier to ensure that
operations get data in the correct layout, e.g. summing each row of row-major data would be much, much faster than
summing the rows of column-major data. Arguably clever implementation of the algorithms could mitigate this to some
degree, but libraries almost never do these difficult optimizations.

The code here provides functions to test for the layout of 2D data and to convert data to the desired layout, providing
a somewhat more efficient algorithm to do so than is provided by ``numpy``.

Of course you are free to just ignore the layout (or the type annotations altogether). This may be acceptable for very
small data sets, but for "serious" code working on non-trivial data, controlling the 2D data layout is vital.
"""

# pylint: disable=duplicate-code,cyclic-import

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

from . import array2d as _array2d
from . import dense as _dense
from . import descriptions as _descriptions
from . import frames as _frames
from . import matrices as _matrices
from . import sparse as _sparse

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "AnyMajor",
    "ROW_MAJOR",
    "COLUMN_MAJOR",
    "as_layout",
    "LARGE_BLOCK_SIZE",
    "SMALL_BLOCK_SIZE",
]


class AnyMajor:
    """
    Allow for either `.ROW_MAJOR` or `.COLUMN_MAJOR` matrix layout (which are the only valid instances of this class).

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


#: Require either `.ROW_MAJOR` or `.COLUMN_MAJOR` layout.
_ANY_MAJOR: AnyMajor = AnyMajor()


class RowMajor(AnyMajor):  # pylint: disable=too-few-public-methods
    """
    Require row-major layout.
    """

    name = "row-major"
    numpy_order = "C"
    numpy_flag = "C_CONTIGUOUS"
    contiguous_axis = 1
    sparse_class = sp.csr_matrix
    sparse_class_name = "scipy.sparse.csr_matrix"


#: Require row-major layout.
#:
#:
#: In this layout, the elements of each row are stored contiguously in memory. For sparse matrices, only the non-zero
#: elements are stored ("CSR" format).
ROW_MAJOR: RowMajor = RowMajor()


class ColumnMajor(AnyMajor):  # pylint: disable=too-few-public-methods
    """
    Require column-major layout.
    """

    name = "column-major"
    numpy_order = "F"
    numpy_flag = "F_CONTIGUOUS"
    contiguous_axis = 0
    sparse_class = sp.csc_matrix
    sparse_class_name = "scipy.sparse.csc_matrix"


#: Require column-major layout.
#:
#: In this layout, the elements of each column are stored contiguously in memory. For sparse matrices, only the non-zero
#: elements are stored ("CSC" format).
COLUMN_MAJOR: ColumnMajor = ColumnMajor()

#: The default size of the block to use in the L1 (two copies should fit w/o thrasing it).
SMALL_BLOCK_SIZE = 12 * 1024

#: The default size of the block to use in the L1 (two copies should fit in it).
LARGE_BLOCK_SIZE = 1024 * 1024


@overload
def as_layout(
    matrix: _array2d.Array2D,
    layout: RowMajor,
    *,
    force_copy: bool = False,
    max_workers: int = 1,
    small_block_size: int = SMALL_BLOCK_SIZE,
    large_block_size: int = LARGE_BLOCK_SIZE,
) -> _array2d.ArrayInRows:
    ...


@overload
def as_layout(
    matrix: _array2d.Array2D,
    layout: ColumnMajor,
    *,
    force_copy: bool = False,
    max_workers: int = 1,
    small_block_size: int = SMALL_BLOCK_SIZE,
    large_block_size: int = LARGE_BLOCK_SIZE,
) -> _array2d.ArrayInColumns:
    ...


@overload
def as_layout(
    matrix: _sparse.Sparse,
    layout: RowMajor,
    *,
    force_copy: bool = False,
    max_workers: int = 1,
    small_block_size: int = SMALL_BLOCK_SIZE,
    large_block_size: int = LARGE_BLOCK_SIZE,
) -> _sparse.SparseInRows:
    ...


@overload
def as_layout(
    matrix: _sparse.Sparse,
    layout: ColumnMajor,
    *,
    force_copy: bool = False,
    max_workers: int = 1,
    small_block_size: int = SMALL_BLOCK_SIZE,
    large_block_size: int = LARGE_BLOCK_SIZE,
) -> _sparse.SparseInColumns:
    ...


@overload
def as_layout(
    matrix: _frames.Frame,
    layout: RowMajor,
    *,
    force_copy: bool = False,
    max_workers: int = 1,
    small_block_size: int = SMALL_BLOCK_SIZE,
    large_block_size: int = LARGE_BLOCK_SIZE,
) -> _frames.FrameInRows:
    ...


@overload
def as_layout(
    matrix: _frames.Frame,
    layout: ColumnMajor,
    *,
    force_copy: bool = False,
    max_workers: int = 1,
    small_block_size: int = SMALL_BLOCK_SIZE,
    large_block_size: int = LARGE_BLOCK_SIZE,
) -> _frames.FrameInColumns:
    ...


def as_layout(
    matrix: Union[_matrices.Matrix, sp.spmatrix],
    layout: AnyMajor,
    *,
    force_copy: bool = False,
    max_workers: int = 1,
    small_block_size: int = SMALL_BLOCK_SIZE,
    large_block_size: int = LARGE_BLOCK_SIZE,
) -> _matrices.Matrix:
    """
    Access the data in a specific layout.

    If ``force_copy``, return a copy even if the data is already in the required layout.

    If we need to actually re-layout the data, and it is in a dense format, and it is "large", it may take a long time
    to do so without parallelization. The code here is able to use multiple threads in such a case, up to
    ``max_workers``. Since we might be invoked from within some parallel code, there's no way for the code here to
    figure out what is the available number of workers we can use, so we default to the only safe value of just one.

    It also turns out that for large dense data it is more efficient to work on the data in blocks that fit a
    "reasonable" HW cache levels. The code here uses a default ``small_block_size`` of 12KB (two copies of this should
    fit in the L1 w/o completely trashing it), and a default ``large_block_size`` of 1MB (two copies of this "should"
    fit in the L2); these values seems to work well in CPUs circa 2022. This optimization should really have been in
    ``numpy`` itself.

    .. todo::

        If/when https://github.com/numpy/numpy/issues/21655 is implemented, change the code here to use it.
    """
    if _matrices.is_matrix(matrix, layout=layout):
        if force_copy:
            return _matrices.matrix_copy(matrix)
        return matrix

    if isinstance(matrix, sp.spmatrix):
        assert layout in (ROW_MAJOR, COLUMN_MAJOR)
        return layout.sparse_class(matrix)  # type: ignore

    assert _dense.is_dense(matrix)
    array2d = _relayout_array2d(_array2d.as_array2d(matrix), layout, max_workers, small_block_size, large_block_size)

    if _array2d.is_array2d(matrix):
        return array2d

    if isinstance(matrix, pd.DataFrame):
        return pd.DataFrame(array2d, index=matrix.index, columns=matrix.columns)

    _descriptions.assert_data(False, "matrix", matrix, None)
    assert False, "never happens"


@overload
def _relayout_array2d(
    array2d: _array2d.Array2D, layout: RowMajor, max_workers: int, small_block_size: int, large_block_size: int
) -> _array2d.ArrayInRows:
    ...


@overload
def _relayout_array2d(
    array2d: _array2d.Array2D, layout: ColumnMajor, max_workers: int, small_block_size: int, large_block_size: int
) -> _array2d.ArrayInColumns:
    ...


@overload
def _relayout_array2d(
    array2d: _array2d.Array2D, layout: AnyMajor, max_workers: int, small_block_size: int, large_block_size: int
) -> _array2d.Array2D:
    ...


def _relayout_array2d(  # pylint: disable=too-many-locals
    array2d: _array2d.Array2D, layout: AnyMajor, max_workers: int, small_block_size: int, large_block_size: int
) -> _array2d.Array2D:
    large_block_elements = int(sqrt(large_block_size / array2d.dtype.itemsize))
    small_block_elements = int(sqrt(small_block_size / array2d.dtype.itemsize))

    large_row_elements = array2d.shape[0]
    large_column_elements = array2d.shape[1]

    large_row_blocks = int(ceil(large_row_elements / large_block_elements))
    large_column_blocks = int(ceil(large_column_elements / small_block_elements))

    result = np.empty(array2d.shape, dtype=array2d.dtype, order=layout.numpy_order)  # type: ignore

    def _relayout_large_block(large_block: int) -> None:
        large_column_block = large_block // large_row_blocks
        large_row_block = large_block % large_row_blocks

        large_start_row = int(round(large_row_block * large_row_elements / large_row_blocks))
        large_stop_row = int(round((large_row_block + 1) * large_row_elements / large_row_blocks))

        large_start_column = int(round(large_column_block * large_column_elements / large_column_blocks))
        large_stop_column = int(round((large_column_block + 1) * large_column_elements / large_column_blocks))

        small_row_elements = large_stop_row - large_start_row
        small_column_elements = large_stop_column - large_start_column

        small_row_blocks = int(ceil(small_row_elements / small_block_elements))
        small_column_blocks = int(ceil(small_column_elements / small_block_elements))

        small_blocks_count = small_row_blocks * small_column_blocks

        def _relayout_small_block(small_block: int) -> None:
            small_column_block = small_block // small_row_blocks
            small_row_block = small_block % small_row_blocks

            small_start_row = large_start_row + int(round(small_row_block * small_row_elements / small_row_blocks))
            small_stop_row = large_start_row + int(round((small_row_block + 1) * small_row_elements / small_row_blocks))

            small_start_column = large_start_column + int(
                round(small_column_block * small_column_elements / small_column_blocks)
            )
            small_stop_column = large_start_column + int(
                round((small_column_block + 1) * small_column_elements / small_column_blocks)
            )

            result[small_start_row:small_stop_row, small_start_column:small_stop_column] = array2d[
                small_start_row:small_stop_row, small_start_column:small_stop_column
            ]

        for small_block in range(small_blocks_count):
            _relayout_small_block(small_block)

    large_blocks_count = large_row_blocks * large_column_blocks
    max_workers = min(max_workers, large_blocks_count)

    if max_workers == 1:
        for large_block in range(large_blocks_count):
            _relayout_large_block(large_block)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(_relayout_large_block, range(large_blocks_count)):
                pass

    assert _array2d.is_array2d(result)
    assert layout.is_layout_of(result)
    return result
