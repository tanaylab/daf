"""
Test ``daf.typing.layouts``.
"""

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.array2d import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.layouts import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.sparse import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-function-docstring


def test_is_layout() -> None:
    for both_major in (
        np.array([[0]], order="C"),
        np.array([[0]], order="F"),
        np.array([[0, 1]], order="C"),
        np.array([[0, 1]], order="F"),
        np.array([[0], [1]], order="C"),
        np.array([[0], [1]], order="F"),
    ):
        assert ROW_MAJOR.is_layout_of(be_array2d(both_major))
        assert ROW_MAJOR.is_layout_of(pd.DataFrame(both_major))
        assert COLUMN_MAJOR.is_layout_of(be_array2d(both_major))
        assert COLUMN_MAJOR.is_layout_of(pd.DataFrame(both_major))

    row_major = be_array2d(np.array(np.zeros((2, 2)), order="C"))
    assert ROW_MAJOR.is_layout_of(row_major)
    assert ROW_MAJOR.is_layout_of(pd.DataFrame(row_major))
    assert not COLUMN_MAJOR.is_layout_of(row_major)
    assert not COLUMN_MAJOR.is_layout_of(pd.DataFrame(row_major))
    assert COLUMN_MAJOR.is_layout_of(row_major.T)
    assert COLUMN_MAJOR.is_layout_of(pd.DataFrame(row_major.T))
    assert COLUMN_MAJOR.is_layout_of(pd.DataFrame(row_major).T)

    column_major = be_array2d(np.array(np.zeros((2, 2)), order="F"))
    assert not ROW_MAJOR.is_layout_of(column_major)
    assert not ROW_MAJOR.is_layout_of(pd.DataFrame(column_major))
    assert COLUMN_MAJOR.is_layout_of(column_major)
    assert COLUMN_MAJOR.is_layout_of(pd.DataFrame(column_major))
    assert ROW_MAJOR.is_layout_of(column_major.T)
    assert ROW_MAJOR.is_layout_of(pd.DataFrame(column_major.T))
    assert ROW_MAJOR.is_layout_of(pd.DataFrame(column_major).T)

    none_major = sp.coo_matrix((2, 2))
    assert not ROW_MAJOR.is_layout_of(none_major)
    assert not COLUMN_MAJOR.is_layout_of(none_major)

    sparse_rows = be_sparse(sp.csr_matrix((2, 2)))
    assert ROW_MAJOR.is_layout_of(sparse_rows)
    assert not COLUMN_MAJOR.is_layout_of(sparse_rows)
    assert COLUMN_MAJOR.is_layout_of(sparse_rows.T)

    sparse_columns = be_sparse(sp.csc_matrix((2, 2)))
    assert not ROW_MAJOR.is_layout_of(sparse_columns)
    assert COLUMN_MAJOR.is_layout_of(sparse_columns)
    assert ROW_MAJOR.is_layout_of(sparse_columns.T)


def test_as_layout() -> None:
    np.random.seed(123456)
    csr_matrix = be_sparse(sp.random(10, 20, density=0.5, format="csr"))
    assert ROW_MAJOR.is_layout_of(csr_matrix)

    row_major = be_array2d(csr_matrix.toarray())
    assert ROW_MAJOR.is_layout_of(csr_matrix)
    assert np.allclose(row_major, csr_matrix.toarray())

    assert id(as_layout(csr_matrix, ROW_MAJOR)) == id(csr_matrix)
    copied_csr_matrix = as_layout(csr_matrix, ROW_MAJOR, force_copy=True)
    assert isinstance(copied_csr_matrix, sp.csr_matrix)
    assert id(copied_csr_matrix) != id(csr_matrix)
    assert np.allclose(copied_csr_matrix.toarray(), row_major)

    csc_matrix = as_layout(csr_matrix, COLUMN_MAJOR)
    assert isinstance(csc_matrix, sp.csc_matrix)
    assert COLUMN_MAJOR.is_layout_of(csc_matrix)
    assert np.allclose(csc_matrix.toarray(), row_major)

    assert id(as_layout(row_major, ROW_MAJOR)) == id(row_major)
    copied_row_major = as_layout(row_major, ROW_MAJOR, force_copy=True)
    assert isinstance(copied_row_major, np.ndarray)
    assert id(copied_row_major) != id(row_major)
    assert np.allclose(copied_row_major, row_major)

    column_major = as_layout(row_major, COLUMN_MAJOR)
    assert id(column_major) != id(row_major)
    assert isinstance(column_major, np.ndarray)
    assert COLUMN_MAJOR.is_layout_of(column_major)
    assert np.allclose(column_major, row_major)

    table_in_rows = pd.DataFrame(row_major)
    assert ROW_MAJOR.is_layout_of(table_in_rows)
    assert id(as_layout(table_in_rows, ROW_MAJOR)) == id(table_in_rows)

    copied_table_in_rows = as_layout(table_in_rows, ROW_MAJOR, force_copy=True)
    assert id(copied_table_in_rows) != id(table_in_rows)
    assert id(copied_table_in_rows.values) != id(table_in_rows.values)
    assert isinstance(copied_table_in_rows, pd.DataFrame)
    assert ROW_MAJOR.is_layout_of(copied_table_in_rows)
    assert np.allclose(copied_table_in_rows.values, row_major)

    table_in_columns = as_layout(table_in_rows, COLUMN_MAJOR)
    assert id(table_in_columns) != id(table_in_rows)
    assert id(table_in_columns.values) != id(table_in_rows.values)
    assert isinstance(table_in_columns, pd.DataFrame)
    assert COLUMN_MAJOR.is_layout_of(table_in_columns)
    assert np.allclose(table_in_columns.values, row_major)

    non_major = sp.coo_matrix(csr_matrix)

    csr_matrix = as_layout(non_major, ROW_MAJOR)
    assert isinstance(csr_matrix, sp.csr_matrix)
    assert ROW_MAJOR.is_layout_of(csr_matrix)
    assert np.allclose(row_major, csr_matrix.toarray())

    csc_matrix = as_layout(non_major, COLUMN_MAJOR)
    assert isinstance(csc_matrix, sp.csc_matrix)
    assert COLUMN_MAJOR.is_layout_of(csc_matrix)
    assert np.allclose(row_major, csc_matrix.toarray())


def test_big_as_layout() -> None:
    np.random.seed(123456)
    row_major = be_array2d(np.random.rand(10, 20))
    serial_column_major = as_layout(row_major, COLUMN_MAJOR, block_size=40)
    parallel_column_major = as_layout(row_major, COLUMN_MAJOR, max_workers=2, block_size=40)
    assert isinstance(serial_column_major, np.ndarray)
    assert isinstance(parallel_column_major, np.ndarray)
    assert np.allclose(row_major, serial_column_major)
    assert np.allclose(row_major, parallel_column_major)
