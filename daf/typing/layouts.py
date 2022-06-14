"""
2D data can be arranged in many layouts. The choice of layout should, in theory, be transparent to the code. In
practice, the layout is crucial for getting reasonable performance, as accessing data "against the grain" results in
orders of magnitude loss of performance.

We restrict 2D data stored in ``daf`` to two layouts: `.ROW_MAJOR` and `.COLUMN_MAJOR`. This applies both to dense data
and also to sparse data (where "row-major" data means "CSR" and "column-major" means "CSC").

We provide explicit data type annotations expressing the distinction between these layouts by suffixing the base type
with ``InRows`` or ``InColumns`` (e.g., `.DenseInRows` vs. `.DenseInColumns`). This makes it easier to ensure that
operations get data in the correct layout, e.g. summing each row of row-major data would be much, much faster than
summing the rows of column-major data. Arguably clever implementation of the algorithms could mitigate this to a large
degree, but libraries almost never do these (non-trivial) optimizations.

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
from typing import Optional
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
from . import optimization as _optimization
from . import sparse as _sparse
from . import unions as _unions

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "AnyMajor",
    "ROW_MAJOR",
    "COLUMN_MAJOR",
    "has_layout",
    "copy2d",
    "fast_all_close",
    "as_layout",
    "LARGE_BLOCK_SIZE",
    "SMALL_BLOCK_SIZE",
    "MAX_WORKERS",
]


class AnyMajor:
    """
    Allow for either `.ROW_MAJOR` or `.COLUMN_MAJOR` 2D data layout (which are the only valid instances of this class).

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


def has_layout(data2d: Any, layout: AnyMajor) -> bool:
    """
    Test whether the given 2D ``data2d`` is in some ``layout``.

    If given non-`.Known2D` data, will always return ``False``.

    .. note::

        Non-sparse 2D data with one row or one column is considered to be both row-major and column-major, if its
        elements are contiguous in memory.
    """
    if isinstance(data2d, layout.sparse_class):
        return True

    if isinstance(data2d, pd.DataFrame):
        data2d = data2d.values

    if not isinstance(data2d, np.ndarray) or data2d.ndim != 2:
        return False

    if layout.contiguous_axis >= 0:
        return data2d.flags[layout.numpy_flag] or data2d.strides[layout.contiguous_axis] == data2d.dtype.itemsize

    return (
        data2d.flags["C_CONTIGUOUS"]
        or data2d.flags["F_CONTIGUOUS"]
        or data2d.strides[0] == data2d.dtype.itemsize
        or data2d.strides[1] == data2d.dtype.itemsize
    )


@overload
def copy2d(data2d: _dense.DenseInRows) -> _dense.DenseInRows:
    ...


@overload
def copy2d(data2d: _dense.DenseInColumns) -> _dense.DenseInColumns:
    ...


@overload
def copy2d(data2d: _sparse.SparseInRows) -> _sparse.SparseInRows:
    ...


@overload
def copy2d(data2d: _sparse.SparseInColumns) -> _sparse.SparseInColumns:
    ...


@overload
def copy2d(data2d: _frames.FrameInRows) -> _frames.FrameInRows:
    ...


@overload
def copy2d(data2d: _frames.FrameInColumns) -> _frames.FrameInColumns:
    ...


@overload
def copy2d(data2d: np.ndarray) -> np.ndarray:
    ...


@overload
def copy2d(data2d: _fake_sparse.spmatrix) -> _fake_sparse.spmatrix:
    ...


@overload
def copy2d(data2d: _fake_pandas.DataFrame) -> _fake_pandas.DataFrame:
    ...


def copy2d(data2d: _unions.Known2D) -> _unions.Known2D:
    """
    Create a copy of 2D data in the same layout.

    All `.Known2D` data types (``numpy.ndarray``, ``scipy.sparse.spmatrix`` and ``pandas.DataFrame``) have a ``copy()``
    method, so you would think one can just write ``data2d.copy()`` and be done. That is *almost* true except that in
    their infinite wisdom ``numpy`` will always create the copy in `.ROW_MAJOR` layout, and ``pandas`` will always
    create the copy in `.COLUMN_MAJOR` layout, because "reasons". Sure, ``numpy`` allows specifying ``order="K"`` but
    ``pandas`` does not, and such a flag makes no sense for ``scipy.sparse.spmatrix`` in the first place.

    The code here will give you a proper copy of the data in the same layout as the original. Sigh.

    .. note::

        In some (older) versions of ``pandas``, it seems it just isn't even possible to create a `.ROW_MAJOR` frame of
        strings.
    """
    if isinstance(data2d, np.ndarray):
        dense_copy: _dense.Dense = np.array(data2d)  # type: ignore
        for layout in (ROW_MAJOR, COLUMN_MAJOR):
            assert has_layout(dense_copy, layout) or not has_layout(data2d, layout)
        return dense_copy

    if isinstance(data2d, pd.DataFrame) and isinstance(data2d.values, np.ndarray):
        values_copy: _dense.Dense = copy2d(data2d.values)  # type: ignore
        frame_copy = pd.DataFrame(values_copy, index=data2d.index, columns=data2d.columns)
        # For ``object`` data, older ``pandas`` insists on column-major order no matter what.
        # This means we have needless duplicated the array above, since ``pandas`` will re-copy (and re-layout) it.
        # Since newer ``pandas`` seems to be doing the right thing, we keep the code above. Sigh.
        if not _dtypes.has_dtype(data2d, _dtypes.STR_DTYPE):
            for layout in (ROW_MAJOR, COLUMN_MAJOR):
                assert has_layout(frame_copy.values, layout) or not has_layout(data2d.values, layout)
        return frame_copy

    _descriptions.assert_data(isinstance(data2d, sp.spmatrix), "known 2D data", data2d)
    return data2d.copy()


def fast_all_close(
    left: _unions.Known,
    right: _unions.Known,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """
    Generalize ``numpy.allclose`` to handle more types, and restrict it to only support efficient comparisons, which
    requires both ``left`` and ``right`` to have the same layout (if they are 2D).

    Specifically:

    * Both values must be 1D (``numpy.ndarray`` or ``pandas.Series``), or

    * Both values must be must be `.typing.sparse.Sparse` matrices, or

    * Both values must be must be either `.Matrix` or ``pandas.DataFrame``.

    And if the values are 2D data:

    * Both values must be in `.ROW_MAJOR` layout, or

    * Both values must be in `.COLUMN_MAJOR` layout.

    Otherwise the code will ``assert`` with a hopefully helpful message.

    .. note::

        When comparing `.typing.sparse.Sparse` matrices, the ``rtol``, ``atol`` and ``equal_nan`` values are only used
        to compare the non-zero values, after ensuring their structure is identical in both matrices. This requires both
        matrices to be `.is_optimal`.
    """

    if left.shape != right.shape:
        return False

    if (  # pylint: disable=too-many-boolean-expressions
        (
            (isinstance(left, sp.csr_matrix) and isinstance(right, sp.csr_matrix))
            or (isinstance(left, sp.csc_matrix) and isinstance(right, sp.csc_matrix))
        )
        and _optimization.is_optimal(left)
        and _optimization.is_optimal(right)
    ):
        return (
            np.allclose(left.indptr, right.indptr)
            and np.allclose(left.indices, right.indices)
            and np.allclose(left.data, right.data, rtol=rtol, atol=atol, equal_nan=equal_nan)
        )

    if (  # pylint: disable=too-many-boolean-expressions
        isinstance(left, (np.ndarray, pd.DataFrame, pd.Series))
        and isinstance(right, (np.ndarray, pd.DataFrame, pd.Series))
        and (
            left.ndim == right.ndim == 1
            or (
                left.ndim == right.ndim == 2
                and (
                    (has_layout(left, ROW_MAJOR) and has_layout(right, ROW_MAJOR))
                    or (has_layout(left, COLUMN_MAJOR) and has_layout(right, COLUMN_MAJOR))
                )
            )
        )
    ):
        return np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=equal_nan)

    assert False, f"comparing a {_descriptions.data_description(left)} with a {_descriptions.data_description(right)}"


#: The default number of parallel worker threads to use in `.as_layout`.
#:
#: By default this is ``1``, that is, we restrict the code to a single thread, as this is the only safe choice. Tempting
#: though it may be to take over all the CPUs of the machine, this isn't necessarily useful (unless the data is **very**
#: large), and if ``daf`` is invoked from multiple processes, it would be a disaster.
#:
#: Since ``daf`` sometimes needs to call ``.as_layout`` from its internal code, you can set this to control how many
#: workers will be used.
MAX_WORKERS = 1

#: The default size of the block to use in the L1 (two copies should fit w/o thrasing it).
SMALL_BLOCK_SIZE = 12 * 1024

#: The default size of the block to use in the L2 (two copies should fit in it).
LARGE_BLOCK_SIZE = 1024 * 1024


@overload
def as_layout(
    data2d: np.ndarray,
    layout: RowMajor,
    *,
    force_copy: bool = False,
    max_workers: Optional[int] = None,
    small_block_size: Optional[int] = None,
    large_block_size: Optional[int] = None,
) -> _dense.DenseInRows:
    ...


@overload
def as_layout(
    data2d: np.ndarray,
    layout: ColumnMajor,
    *,
    force_copy: bool = False,
    max_workers: Optional[int] = None,
    small_block_size: Optional[int] = None,
    large_block_size: Optional[int] = None,
) -> _dense.DenseInColumns:
    ...


@overload
def as_layout(
    data2d: _fake_sparse.spmatrix,
    layout: RowMajor,
    *,
    force_copy: bool = False,
    max_workers: Optional[int] = None,
    small_block_size: Optional[int] = None,
    large_block_size: Optional[int] = None,
) -> _sparse.SparseInRows:
    ...


@overload
def as_layout(
    data2d: _fake_sparse.spmatrix,
    layout: ColumnMajor,
    *,
    force_copy: bool = False,
    max_workers: Optional[int] = None,
    small_block_size: Optional[int] = None,
    large_block_size: Optional[int] = None,
) -> _sparse.SparseInColumns:
    ...


@overload
def as_layout(
    data2d: _fake_pandas.DataFrame,
    layout: RowMajor,
    *,
    force_copy: bool = False,
    max_workers: Optional[int] = None,
    small_block_size: Optional[int] = None,
    large_block_size: Optional[int] = None,
) -> _frames.FrameInRows:
    ...


def as_layout(
    data2d: _unions.Known2D,
    layout: AnyMajor,
    *,
    force_copy: bool = False,
    max_workers: Optional[int] = None,
    small_block_size: Optional[int] = None,
    large_block_size: Optional[int] = None,
) -> _unions.Known2D:
    """
    Access the 2D data in a specific layout.

    If ``force_copy``, return a copy even if the data is already in the required layout.

    If we need to actually re-layout the data, and it is "large", it may take a long time to do so without
    parallelization. The code here is able to use multiple threads (at least for `.typing.dense.Dense` data), up to
    ``max_workers`` Since we might be invoked from within some parallel code, there's no way for the code here to figure
    out what is the available number of workers we can use, so we default to the only safe value of ``1``.

    It turns out that for large dense data it is more efficient to work on the data in blocks that fit a "reasonable" HW
    cache levels. The code here uses a default ``small_block_size`` of 12KB (two copies of this should fit in the L1 w/o
    completely trashing it), and a default ``large_block_size`` of 1MB (two copies of this "should" fit in the
    L2); these values seems to work well in CPUs circa 2022. This optimization should really have been in ``numpy``
    itself, ideally using some cache-oblivious algorithm, which would negate need for specifying buffer sizes.

    .. todo::

        If/when https://github.com/numpy/numpy/issues/21655 is implemented, change the `.as_layout` implementation to
        use it.
    """
    _descriptions.assert_data(hasattr(data2d, "ndim") and getattr(data2d, "ndim") == 2, "known 2D data", data2d)

    if has_layout(data2d, layout):
        if force_copy:
            return copy2d(data2d)
        return data2d

    if isinstance(data2d, sp.spmatrix):
        assert layout in (ROW_MAJOR, COLUMN_MAJOR)
        return layout.sparse_class(data2d)  # type: ignore

    max_workers = max_workers or MAX_WORKERS
    small_block_size = small_block_size or SMALL_BLOCK_SIZE
    large_block_size = large_block_size or LARGE_BLOCK_SIZE

    assert max_workers > 0
    assert small_block_size > 0
    assert large_block_size > small_block_size

    assert _dense.is_dense(data2d) or _frames.is_frame(data2d)
    dense = _relayout_dense(_dense.as_dense(data2d), layout, max_workers, small_block_size, large_block_size)

    if _dense.is_dense(data2d):
        return dense

    if isinstance(data2d, pd.DataFrame):
        return pd.DataFrame(dense, index=data2d.index, columns=data2d.columns)

    _descriptions.assert_data(False, "known 2D data", data2d)
    assert False, "never happens"


@overload
def _relayout_dense(
    dense: _dense.Dense, layout: RowMajor, max_workers: int, small_block_size: int, large_block_size: int
) -> _dense.DenseInRows:
    ...


@overload
def _relayout_dense(
    dense: _dense.Dense, layout: ColumnMajor, max_workers: int, small_block_size: int, large_block_size: int
) -> _dense.DenseInColumns:
    ...


@overload
def _relayout_dense(
    dense: _dense.Dense, layout: AnyMajor, max_workers: int, small_block_size: int, large_block_size: int
) -> _dense.Dense:
    ...


def _relayout_dense(  # pylint: disable=too-many-locals
    dense: _dense.Dense, layout: AnyMajor, max_workers: int, small_block_size: int, large_block_size: int
) -> _dense.Dense:
    large_block_elements = int(sqrt(large_block_size / dense.dtype.itemsize))
    small_block_elements = int(sqrt(small_block_size / dense.dtype.itemsize))

    large_row_elements = dense.shape[0]
    large_column_elements = dense.shape[1]

    large_row_blocks = int(ceil(large_row_elements / large_block_elements))
    large_column_blocks = int(ceil(large_column_elements / small_block_elements))

    result = np.empty(dense.shape, dtype=dense.dtype, order=layout.numpy_order)  # type: ignore

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

            result[small_start_row:small_stop_row, small_start_column:small_stop_column] = dense[
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

    assert _dense.is_dense(result)
    assert has_layout(result, layout)
    return result
