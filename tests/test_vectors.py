"""
Test ``daf.typing.vector``.
"""

# pylint: disable=duplicate-code

from typing import Any
from typing import Optional

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.unions import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.vectors import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import allow_np_matrix
from . import expect_raise

# pylint: disable=missing-function-docstring


def assert_is_vector(data: Any, **kwargs: Any) -> None:
    assert is_vector(data, **kwargs)
    assert is_proper1d(data, **kwargs)
    assert is_proper(data, **kwargs)
    assert is_known1d(data)
    assert is_known(data)


# pylint: enable=duplicate-code


def test_is_vector() -> None:
    assert_is_vector(np.array([0]))
    assert_is_vector(np.array([0], dtype="bool"), dtype="bool")

    assert not is_vector(np.array([0], dtype="bool"), dtype="int16")
    assert not is_vector(np.array([[0], [1]]))
    assert not is_vector(pd.Series([0]))

    for sparse_format in ("csr", "csc"):
        np.random.seed(123456)
        sparse = sp.random(10, 10, density=0.5, format=sparse_format)
        assert not is_vector(sparse[0, :])
        assert not is_vector(sparse[:, 0])

    dense = np.zeros((10, 10))
    assert_is_vector(dense[0, :])
    assert_is_vector(dense[:, 0])

    table = pd.DataFrame(dense)
    assert not is_vector(table.iloc[0, :])
    assert not is_vector(table.iloc[:, 0])


def assert_be_vector(data: Any, **kwargs: Any) -> None:
    assert id(be_vector(data, **kwargs)) == id(data)
    assert id(be_proper1d(data, **kwargs)) == id(data)
    assert id(be_proper(data, **kwargs)) == id(data)
    assert id(be_known1d(data)) == id(data)
    assert id(be_known(data)) == id(data)


def assert_not_be_vector(message: str, data: Any, **kwargs: Any) -> None:
    with expect_raise(message):
        be_vector(data, **kwargs)


def test_be_vector() -> None:
    assert_be_vector(np.array([0]))
    assert_be_vector(np.array([0], dtype="bool"), dtype="bool")

    assert_not_be_vector(
        "expected: 1D numpy.ndarray of int16, got: 1D numpy.ndarray of 1 of bool (all false)",
        np.array([0], dtype="bool"),
        dtype="int16",
    )
    assert_not_be_vector(
        "expected: 1D numpy.ndarray of any reasonable type, got: both-major numpy.ndarray of 2x1 of int64",
        np.array([[0], [1]]),
    )
    assert_not_be_vector(
        "expected: 1D numpy.ndarray of any reasonable type, got: pandas.Series of 1 of int64",
        data=pd.Series([0]),
    )

    for sparse_format in ("csr", "csc"):
        np.random.seed(123456)
        sparse = sp.random(10, 10, density=0.5, format=sparse_format)
        assert_not_be_vector(
            "expected: 1D numpy.ndarray of any reasonable type, "
            f"got: scipy.sparse.{sparse_format}_matrix of 1x10 of float64 with 40.00% nnz",
            sparse[0, :],
        )
        assert_not_be_vector(
            "expected: 1D numpy.ndarray of any reasonable type, "
            f"got: scipy.sparse.{sparse_format}_matrix of 10x1 of float64 with 60.00% nnz",
            sparse[:, 0],
        )

    dense = np.zeros((1, 10))
    assert_not_be_vector(
        "expected: 1D numpy.ndarray of any reasonable type, got: both-major numpy.ndarray of 1x10 of float64",
        dense,
    )
    assert_not_be_vector(
        "expected: 1D numpy.ndarray of any reasonable type, got: both-major numpy.ndarray of 10x1 of float64",
        dense.T,
    )

    dense = np.zeros((10, 10))
    assert_be_vector(dense[0, :])
    assert_be_vector(dense[:, 0])

    table = pd.DataFrame(dense)
    assert_not_be_vector(
        "expected: 1D numpy.ndarray of any reasonable type, got: pandas.Series of 10 of float64",
        table.iloc[0, :],
    )
    assert_not_be_vector(
        "expected: 1D numpy.ndarray of any reasonable type, got: pandas.Series of 10 of float64",
        table.iloc[:, 0],
    )

    allow_np_matrix()

    assert_not_be_vector(
        "expected: 1D numpy.ndarray of any reasonable type, got: numpy.matrix of 2x1 of int64",
        np.matrix([[0], [1]]),
    )
    assert_not_be_vector(
        "expected: 1D numpy.ndarray of any reasonable type, got: numpy.matrix of 1x2 of int64",
        np.matrix([[0, 1]]),
    )


# pylint: disable=duplicate-code


def assert_as_is_vector(data: Any, expected_id: Optional[int] = None) -> None:
    if expected_id is None:
        expected_id = id(data)

    assert id(as_vector(data)) == expected_id, "as_vector returned a different object"
    assert id(as_vector(data, force_copy=True)) != expected_id, "as_vector did not force a copy"


def assert_as_can_be_vector(data: Any) -> None:
    assert id(as_vector(data)) != id(data), "as_vector returned same object"


# pylint: enable=duplicate-code


def test_as_vector() -> None:
    assert_as_is_vector(np.array([0]))

    series = pd.Series(np.array([0]))
    assert_as_is_vector(series, id(series.values))

    np.random.seed(123456)
    for sparse_format in ("csr", "csc", "coo"):
        for shape in ((1, 10), (10, 1)):
            sparse = sp.random(*shape, density=0.5, format=sparse_format)
            assert_as_can_be_vector(sparse)

    dense = np.zeros((10, 10))
    assert_as_is_vector(dense[0, :])
    assert_as_is_vector(dense[:, 0])

    series = pd.Series(dense[0, :])
    assert_as_is_vector(series, id(series.values))

    table = pd.DataFrame(dense)
    assert_as_can_be_vector(table.iloc[0, :])
    assert_as_can_be_vector(table.iloc[:, 0])

    allow_np_matrix()

    assert_as_can_be_vector(np.matrix([[0], [1]]))
    assert_as_can_be_vector(np.matrix([[0, 1]]))

    assert_as_can_be_vector([1, 2, 3])
