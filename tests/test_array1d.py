"""
Test ``daf.typing.array1d``.
"""

# pylint: disable=duplicate-code

from typing import Any
from typing import Optional

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.array1d import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.vectors import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import allow_np_matrix
from . import expect_raise

# pylint: disable=missing-function-docstring


def assert_is_array1d(data: Any, **kwargs: Any) -> None:
    assert is_array1d(data, **kwargs)
    assert is_vector(data, **kwargs)


def assert_not_is_array1d(data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    assert not is_array1d(data, **kwargs)

    if kind == "vector":
        assert is_vector(data, **kwargs)
    else:
        assert not is_vector(data, **kwargs)


# pylint: enable=duplicate-code


def test_is_array1d() -> None:
    assert_is_array1d(np.array([0]))
    assert_is_array1d(np.array([0], dtype="bool"), dtype="bool")

    assert_not_is_array1d(np.array([0], dtype="bool"), dtype="int16")
    assert_not_is_array1d(np.array([[0], [1]]))
    assert_not_is_array1d(pd.Series([0]), kind="vector")

    for sparse_format in ("csr", "csc"):
        np.random.seed(123456)
        sparse = sp.random(10, 10, density=0.5, format=sparse_format)
        assert_not_is_array1d(sparse[0, :])
        assert_not_is_array1d(sparse[:, 0])

    dense = np.zeros((10, 10))
    assert_is_array1d(dense[0, :])
    assert_is_array1d(dense[:, 0])

    table = pd.DataFrame(dense)
    assert_not_is_array1d(table.iloc[0, :], kind="vector")
    assert_not_is_array1d(table.iloc[:, 0], kind="vector")


# pylint: disable=duplicate-code
def assert_be_array1d(data: Any, **kwargs: Any) -> None:
    assert id(be_array1d(data, **kwargs)) == id(data), "be_array1d returned a different object"
    assert id(be_vector(data, **kwargs)) == id(data), "be_vector returned a different object"


def assert_not_be_array1d(message: str, data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    with expect_raise(message):
        be_array1d(data, **kwargs)

    if kind == "vector":
        assert id(be_vector(data, **kwargs)) == id(data), "be_vector returned a different object"
    else:
        with expect_raise(message.replace("expected 1D numpy.ndarray", "expected vector")):
            be_vector(data, **kwargs)


# pylint: disable=duplicate-code


def test_be_array1d() -> None:
    assert_be_array1d(np.array([0]))
    assert_be_array1d(np.array([0], dtype="bool"), dtype="bool")

    assert_not_be_array1d(
        "expected 1D numpy.ndarray of int16, got 1D numpy.ndarray of 1 of bool",
        np.array([0], dtype="bool"),
        dtype="int16",
    )
    assert_not_be_array1d(
        "expected 1D numpy.ndarray of any reasonable type, got both-major numpy.ndarray of 2x1 of int64",
        np.array([[0], [1]]),
    )
    assert_not_be_array1d(
        "expected 1D numpy.ndarray of any reasonable type, got pandas.Series of 1 of int64",
        data=pd.Series([0]),
        kind="vector",
    )

    for sparse_format in ("csr", "csc"):
        np.random.seed(123456)
        sparse = sp.random(10, 10, density=0.5, format=sparse_format)
        assert_not_be_array1d(
            "expected 1D numpy.ndarray of any reasonable type, "
            f"got scipy.sparse.{sparse_format}_matrix of 1x10 of float64 with 40.00% nnz",
            sparse[0, :],
        )
        assert_not_be_array1d(
            "expected 1D numpy.ndarray of any reasonable type, "
            f"got scipy.sparse.{sparse_format}_matrix of 10x1 of float64 with 60.00% nnz",
            sparse[:, 0],
        )

    dense = np.zeros((1, 10))
    assert_not_be_array1d(
        "expected 1D numpy.ndarray of any reasonable type, got both-major numpy.ndarray of 1x10 of float64",
        dense,
    )
    assert_not_be_array1d(
        "expected 1D numpy.ndarray of any reasonable type, got both-major numpy.ndarray of 10x1 of float64",
        dense.T,
    )

    dense = np.zeros((10, 10))
    assert_be_array1d(dense[0, :])
    assert_be_array1d(dense[:, 0])

    table = pd.DataFrame(dense)
    assert_not_be_array1d(
        "expected 1D numpy.ndarray of any reasonable type, got pandas.Series of 10 of float64",
        table.iloc[0, :],
        kind="vector",
    )
    assert_not_be_array1d(
        "expected 1D numpy.ndarray of any reasonable type, got pandas.Series of 10 of float64",
        table.iloc[:, 0],
        kind="vector",
    )

    allow_np_matrix()

    assert_not_be_array1d(
        "expected 1D numpy.ndarray of any reasonable type, got numpy.matrix of 2x1 of int64",
        np.matrix([[0], [1]]),
    )
    assert_not_be_array1d(
        "expected 1D numpy.ndarray of any reasonable type, got numpy.matrix of 1x2 of int64",
        np.matrix([[0, 1]]),
    )


# pylint: disable=duplicate-code


def assert_as_is_array1d(data: Any, expected_id: Optional[int] = None) -> None:
    if expected_id is None:
        expected_id = id(data)

    assert id(as_array1d(data)) == expected_id, "as_array1d returned a different object"
    assert id(as_array1d(data, force_copy=True)) != expected_id, "as_array1d did not force a copy"


def assert_as_can_be_array1d(data: Any) -> None:
    assert id(as_array1d(data)) != id(data), "as_array1d returned same object"


# pylint: enable=duplicate-code


def test_as_array1d() -> None:
    assert_as_is_array1d(np.array([0]))

    series = pd.Series(np.array([0]))
    assert_as_is_array1d(series, id(series.values))

    np.random.seed(123456)
    for sparse_format in ("csr", "csc", "coo"):
        for shape in ((1, 10), (10, 1)):
            sparse = sp.random(*shape, density=0.5, format=sparse_format)
            assert_as_can_be_array1d(sparse)

    dense = np.zeros((10, 10))
    assert_as_is_array1d(dense[0, :])
    assert_as_is_array1d(dense[:, 0])

    series = pd.Series(dense[0, :])
    assert_as_is_array1d(series, id(series.values))

    table = pd.DataFrame(dense)
    assert_as_can_be_array1d(table.iloc[0, :])
    assert_as_can_be_array1d(table.iloc[:, 0])

    allow_np_matrix()

    assert_as_can_be_array1d(np.matrix([[0], [1]]))
    assert_as_can_be_array1d(np.matrix([[0, 1]]))

    assert_as_can_be_array1d([1, 2, 3])
