"""
Test ``daf.typing.comparisons``.
"""

# pylint: disable=duplicate-code

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.comparisons import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_raise

# pylint: disable=duplicate-code

# pylint: disable=missing-function-docstring


def test_compare_array1d() -> None:
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


def test_compare_array2d() -> None:
    assert fast_all_close(np.array([[0]]), np.array([[0]]))
    assert not fast_all_close(np.array([[0]]), np.array([[1]]))
    assert not fast_all_close(np.array([[0]]), np.array([[0], [0]]))


def test_compare_tables() -> None:
    assert fast_all_close(pd.DataFrame([[0]]), pd.DataFrame([[0]]))
    assert not fast_all_close(pd.DataFrame([[0]]), pd.DataFrame([[1]]))
    assert not fast_all_close(pd.DataFrame([[0]]), np.array([[0], [0]]))


def test_compare_dense() -> None:
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
