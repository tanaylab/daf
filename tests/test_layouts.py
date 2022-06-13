"""
Test ``daf.typing.layouts``.
"""

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.dense import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.layouts import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.sparse import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_raise

# pylint: disable=missing-function-docstring


def test_compare_vector() -> None:
    assert fast_all_close(np.array([0]), np.array([0]))
    assert not fast_all_close(np.array([0]), np.array([1]))
    assert not fast_all_close(np.array([0]), np.array([0, 0]))


def test_compare_series() -> None:
    assert fast_all_close(pd.Series([0]), pd.Series([0]))
    assert not fast_all_close(pd.Series([0]), pd.Series([1]))
    assert not fast_all_close(pd.Series([0]), pd.Series([0, 0]))


def test_compare_vectors() -> None:
    assert fast_all_close(np.array([0]), pd.Series([0]))
    assert not fast_all_close(np.array([0]), pd.Series([1]))
    assert not fast_all_close(np.array([0]), pd.Series([0, 0]))


def test_compare_dense() -> None:
    assert fast_all_close(np.array([[0]]), np.array([[0]]))
    assert not fast_all_close(np.array([[0]]), np.array([[1]]))
    assert not fast_all_close(np.array([[0]]), np.array([[0], [0]]))


def test_compare_tables() -> None:
    assert fast_all_close(pd.DataFrame([[0]]), pd.DataFrame([[0]]))
    assert not fast_all_close(pd.DataFrame([[0]]), pd.DataFrame([[1]]))
    assert not fast_all_close(pd.DataFrame([[0]]), np.array([[0], [0]]))


def test_compare_dense_frames() -> None:
    assert fast_all_close(np.array([[0]]), pd.DataFrame([[0]]))
    assert not fast_all_close(np.array([[0]]), pd.DataFrame([[1]]))
    assert not fast_all_close(np.array([[0]]), pd.DataFrame([[0], [0]]))


def test_compare_sparse() -> None:
    assert fast_all_close(sp.csr_matrix([[0, 1], [1, 0]]), sp.csr_matrix([[0, 1], [1, 0]]))
    assert fast_all_close(sp.csc_matrix([[0, 1], [1, 0]]), sp.csc_matrix([[0, 1], [1, 0]]))
    assert not fast_all_close(sp.csr_matrix([[0, 1], [1, 0]]), sp.csr_matrix([[0, 1], [1, 1]]))
    assert not fast_all_close(sp.csc_matrix([[0, 1], [1, 0]]), sp.csc_matrix([[0, 1], [1, 1]]))
    assert not fast_all_close(sp.csr_matrix([[0, 1], [1, 0]]), sp.csr_matrix([[0, 1], [1, 0], [1, 1]]))
    assert not fast_all_close(sp.csc_matrix([[0, 1], [1, 0]]), sp.csc_matrix([[0, 1], [1, 0], [1, 1]]))


def test_compare_fail() -> None:
    with expect_raise(
        f"comparing a {sp.coo_matrix.__module__}.{sp.coo_matrix.__qualname__} of 2x2 of int64 with 50.00% nnz "
        f"with a {sp.coo_matrix.__module__}.{sp.coo_matrix.__qualname__} of 2x2 of int64 with 50.00% nnz"
    ):
        fast_all_close(sp.coo_matrix([[0, 1], [1, 0]]), sp.coo_matrix([[0, 1], [1, 0]]))

    with expect_raise(
        "comparing a row-major numpy.ndarray of 2x2 of int64 with a column-major numpy.ndarray of 2x2 of int64"
    ):
        fast_all_close(np.array([[0, 1], [1, 0]], order="C"), np.array([[0, 1], [1, 0]], order="F"))


def test_has_layout() -> None:
    for both_major in (
        np.array([[0]], order="C"),
        np.array([[0]], order="F"),
        np.array([[0, 1]], order="C"),
        np.array([[0, 1]], order="F"),
        np.array([[0], [1]], order="C"),
        np.array([[0], [1]], order="F"),
    ):
        assert has_layout(be_dense(both_major), ROW_MAJOR)
        assert has_layout(pd.DataFrame(both_major), ROW_MAJOR)
        assert has_layout(be_dense(both_major), COLUMN_MAJOR)
        assert has_layout(pd.DataFrame(both_major), COLUMN_MAJOR)

    row_major = be_dense(np.array(np.zeros((2, 2)), order="C"))
    assert has_layout(row_major, ROW_MAJOR)
    assert has_layout(pd.DataFrame(row_major), ROW_MAJOR)
    assert not has_layout(row_major, COLUMN_MAJOR)
    assert not has_layout(pd.DataFrame(row_major), COLUMN_MAJOR)
    assert has_layout(row_major.T, COLUMN_MAJOR)
    assert has_layout(pd.DataFrame(row_major.T), COLUMN_MAJOR)
    assert has_layout(pd.DataFrame(row_major).T, COLUMN_MAJOR)

    column_major = be_dense(np.array(np.zeros((2, 2)), order="F"))
    assert not has_layout(column_major, ROW_MAJOR)
    assert not has_layout(pd.DataFrame(column_major), ROW_MAJOR)
    assert has_layout(column_major, COLUMN_MAJOR)
    assert has_layout(pd.DataFrame(column_major), COLUMN_MAJOR)
    assert has_layout(column_major.T, ROW_MAJOR)
    assert has_layout(pd.DataFrame(column_major.T), ROW_MAJOR)
    assert has_layout(pd.DataFrame(column_major).T, ROW_MAJOR)

    none_major = sp.coo_matrix((2, 2))
    assert not has_layout(none_major, ROW_MAJOR)
    assert not has_layout(none_major, COLUMN_MAJOR)

    sparse_rows = be_sparse(sp.csr_matrix((2, 2)))
    assert has_layout(sparse_rows, ROW_MAJOR)
    assert not has_layout(sparse_rows, COLUMN_MAJOR)
    assert has_layout(sparse_rows.T, COLUMN_MAJOR)

    sparse_columns = be_sparse(sp.csc_matrix((2, 2)))
    assert not has_layout(sparse_columns, ROW_MAJOR)
    assert has_layout(sparse_columns, COLUMN_MAJOR)
    assert has_layout(sparse_columns.T, ROW_MAJOR)


def test_as_layout() -> None:
    np.random.seed(123456)
    csr_matrix = be_sparse(sp.random(10, 20, density=0.5, format="csr"))
    assert has_layout(csr_matrix, ROW_MAJOR)

    row_major = be_dense(csr_matrix.toarray())
    assert has_layout(csr_matrix, ROW_MAJOR)
    assert np.allclose(row_major, csr_matrix.toarray())

    assert id(as_layout(csr_matrix, ROW_MAJOR)) == id(csr_matrix)
    copied_csr_matrix = as_layout(csr_matrix, ROW_MAJOR, force_copy=True)
    assert isinstance(copied_csr_matrix, sp.csr_matrix)
    assert id(copied_csr_matrix) != id(csr_matrix)
    assert np.allclose(copied_csr_matrix.toarray(), row_major)

    csc_matrix = as_layout(csr_matrix, COLUMN_MAJOR)
    assert isinstance(csc_matrix, sp.csc_matrix)
    assert has_layout(csc_matrix, COLUMN_MAJOR)
    assert np.allclose(csc_matrix.toarray(), row_major)

    assert id(as_layout(row_major, ROW_MAJOR)) == id(row_major)
    copied_row_major = as_layout(row_major, ROW_MAJOR, force_copy=True)
    assert isinstance(copied_row_major, np.ndarray)
    assert id(copied_row_major) != id(row_major)
    assert np.allclose(copied_row_major, row_major)

    column_major = as_layout(row_major, COLUMN_MAJOR)
    assert id(column_major) != id(row_major)
    assert isinstance(column_major, np.ndarray)
    assert has_layout(column_major, COLUMN_MAJOR)
    assert np.allclose(column_major, row_major)

    table_in_rows = pd.DataFrame(row_major)
    assert has_layout(table_in_rows, ROW_MAJOR)
    assert id(as_layout(table_in_rows, ROW_MAJOR)) == id(table_in_rows)

    copied_table_in_rows = as_layout(table_in_rows, ROW_MAJOR, force_copy=True)
    assert id(copied_table_in_rows) != id(table_in_rows)
    assert id(copied_table_in_rows.values) != id(table_in_rows.values)
    assert isinstance(copied_table_in_rows, pd.DataFrame)
    assert has_layout(copied_table_in_rows, ROW_MAJOR)
    assert np.allclose(copied_table_in_rows.values, row_major)

    table_in_columns = as_layout(table_in_rows, COLUMN_MAJOR)
    assert id(table_in_columns) != id(table_in_rows)
    assert id(table_in_columns.values) != id(table_in_rows.values)
    assert isinstance(table_in_columns, pd.DataFrame)
    assert has_layout(table_in_columns, COLUMN_MAJOR)
    assert np.allclose(table_in_columns.values, row_major)

    non_major = sp.coo_matrix(csr_matrix)

    csr_matrix = as_layout(non_major, ROW_MAJOR)
    assert isinstance(csr_matrix, sp.csr_matrix)
    assert has_layout(csr_matrix, ROW_MAJOR)
    assert np.allclose(row_major, csr_matrix.toarray())

    csc_matrix = as_layout(non_major, COLUMN_MAJOR)
    assert isinstance(csc_matrix, sp.csc_matrix)
    assert has_layout(csc_matrix, COLUMN_MAJOR)
    assert np.allclose(row_major, csc_matrix.toarray())


def test_big_as_layout() -> None:
    np.random.seed(123456)
    row_major = be_dense(np.random.rand(10, 20))
    serial_column_major = as_layout(row_major, COLUMN_MAJOR, small_block_size=10, large_block_size=40)
    parallel_column_major = as_layout(row_major, COLUMN_MAJOR, max_workers=2, small_block_size=10, large_block_size=40)
    assert isinstance(serial_column_major, np.ndarray)
    assert isinstance(parallel_column_major, np.ndarray)
    assert np.allclose(row_major, serial_column_major)
    assert np.allclose(row_major, parallel_column_major)
