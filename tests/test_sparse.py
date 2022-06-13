"""
Test ``daf.typing.sparse``.
"""

# pylint: disable=duplicate-code

from typing import Any

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.layouts import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.matrices import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.sparse import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.unions import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import allow_np_matrix
from . import expect_raise

# pylint: disable=missing-function-docstring


def assert_is_sparse(data: Any, **kwargs: Any) -> None:
    assert is_sparse(data, **kwargs)
    assert is_matrix(data, **kwargs)
    if "layout" in kwargs:
        del kwargs["layout"]
    assert is_proper2d(data, **kwargs)
    assert is_proper(data, **kwargs)
    assert is_known2d(data)
    assert is_known(data)


def assert_is_sparse_in_rows(data: Any, **kwargs: Any) -> None:
    assert is_sparse_in_rows(data, **kwargs)
    assert is_matrix_in_rows(data, **kwargs)
    assert is_proper_in_rows(data, **kwargs)
    assert is_known2d(data)
    assert is_known(data)


def assert_is_sparse_in_columns(data: Any, **kwargs: Any) -> None:
    assert is_sparse_in_columns(data, **kwargs)
    assert is_matrix_in_columns(data, **kwargs)
    assert is_proper_in_columns(data, **kwargs)
    assert is_known2d(data)
    assert is_known(data)


# pylint: enable=duplicate-code


def test_is_sparse() -> None:
    assert_is_sparse(sp.csr_matrix((10, 10)))
    assert_is_sparse(sp.csc_matrix((10, 10)))
    assert not is_sparse(sp.coo_matrix((10, 10)))
    assert not is_sparse(np.array([[0]]))
    assert not is_sparse(pd.DataFrame([[0]]))

    row_major = sp.csr_matrix((10, 10))
    assert_is_sparse(row_major, layout=ROW_MAJOR)
    assert not is_sparse(row_major, layout=COLUMN_MAJOR)

    assert_is_sparse_in_rows(row_major)
    assert_is_sparse_in_rows(copy2d(row_major))
    assert not is_sparse_in_columns(row_major)

    assert_is_sparse_in_columns(row_major.T)
    assert_is_sparse_in_columns(copy2d(row_major.T))
    assert not is_sparse_in_rows(row_major.T)

    allow_np_matrix()

    assert not is_sparse(np.matrix([[0], [1]]))

    assert not is_sparse(pd.DataFrame(dict(a=[0], b=["1"])))


# pylint: disable=duplicate-code


def assert_be_sparse(data: Any, **kwargs: Any) -> None:
    assert id(be_sparse(data, **kwargs)) == id(data)
    assert id(be_matrix(data, **kwargs)) == id(data)
    if "layout" in kwargs:
        del kwargs["layout"]
    assert id(be_proper2d(data, **kwargs)) == id(data)
    assert id(be_proper(data, **kwargs)) == id(data)
    assert id(be_known2d(data)) == id(data)
    assert id(be_known(data)) == id(data)


def assert_be_sparse_in_rows(data: Any, **kwargs: Any) -> None:
    assert id(be_sparse_in_rows(data, **kwargs)) == id(data)
    assert id(be_matrix_in_rows(data, **kwargs)) == id(data)
    assert id(be_proper_in_rows(data, **kwargs)) == id(data)
    assert id(be_known2d(data)) == id(data)
    assert id(be_known(data)) == id(data)


def assert_be_sparse_in_columns(data: Any, **kwargs: Any) -> None:
    assert id(be_sparse_in_columns(data, **kwargs)) == id(data)
    assert id(be_matrix_in_columns(data, **kwargs)) == id(data)
    assert id(be_proper_in_columns(data, **kwargs)) == id(data)
    assert id(be_known2d(data)) == id(data)
    assert id(be_known(data)) == id(data)


def assert_not_be_sparse(message: str, data: Any, **kwargs: Any) -> None:
    with expect_raise(message):
        be_sparse(data, **kwargs)


def assert_not_be_sparse_in_rows(message: str, data: Any, **kwargs: Any) -> None:
    with expect_raise(message):
        be_sparse_in_rows(data, **kwargs)


def assert_not_be_sparse_in_columns(message: str, data: Any, **kwargs: Any) -> None:
    with expect_raise(message):
        be_sparse_in_columns(data, **kwargs)


# pylint: enable=duplicate-code


def test_be_sparse() -> None:
    assert_be_sparse(sp.csr_matrix((10, 10)))
    assert_be_sparse(sp.csc_matrix((10, 10)))
    assert_not_be_sparse(
        "expected: scipy.sparse.csr/csc_matrix of any reasonable type, "
        f"got: {sp.coo_matrix.__module__}.{sp.coo_matrix.__qualname__} of 10x10 of float64 with 0.00% nnz",
        sp.coo_matrix((10, 10)),
    )
    assert_not_be_sparse(
        "expected: scipy.sparse.csr/csc_matrix of any reasonable type, "
        "got: both-major numpy.ndarray of 1x1 of float64",
        np.array([[0.0]]),
    )
    assert_not_be_sparse(
        "expected: scipy.sparse.csr/csc_matrix of any reasonable type, "
        "got: both-major pandas.DataFrame of 1x1 of int64",
        pd.DataFrame([[0]]),
    )

    row_major = sp.csr_matrix((10, 5))
    assert_be_sparse(row_major, layout=ROW_MAJOR)

    assert_be_sparse_in_rows(row_major)
    assert_be_sparse_in_rows(copy2d(row_major))
    assert_not_be_sparse_in_columns(
        "expected: scipy.sparse.csc_matrix of any reasonable type, "
        "got: scipy.sparse.csr_matrix of 10x5 of float64 with 0.00% nnz",
        row_major,
    )

    assert_be_sparse_in_columns(row_major.T)
    assert_be_sparse_in_columns(copy2d(row_major.T))
    assert_not_be_sparse_in_rows(
        "expected: scipy.sparse.csr_matrix of any reasonable type, "
        "got: scipy.sparse.csc_matrix of 5x10 of float64 with 0.00% nnz",
        row_major.T,
    )

    allow_np_matrix()

    assert_not_be_sparse(
        "expected: scipy.sparse.csr/csc_matrix of any reasonable type, got: numpy.matrix of 2x1 of int64",
        np.matrix([[0], [1]]),
    )

    assert_not_be_sparse(
        "expected: scipy.sparse.csr/csc_matrix of any reasonable type, "
        "got: both-major pandas.DataFrame of 1x2 of mixed types",
        pd.DataFrame(dict(a=[0], b=["1"])),
    )
