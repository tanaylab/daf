"""
Test ``daf.typing.array2d``.
"""

# pylint: disable=duplicate-code

from typing import Any
from typing import Optional

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.array2d import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.dense import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.grids import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.layouts import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.matrices import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.sparse import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import allow_np_matrix
from . import expect_raise

# pylint: disable=missing-function-docstring


def assert_is_array2d(data: Any, **kwargs: Any) -> None:
    assert is_array2d(data, **kwargs)
    assert is_matrix(data, **kwargs)
    assert is_grid(data, **kwargs)
    assert is_dense(data, **kwargs)


def assert_is_array_in_rows(data: Any, **kwargs: Any) -> None:
    assert is_array_in_rows(data, **kwargs)
    assert is_matrix_in_rows(data, **kwargs)
    assert is_grid_in_rows(data, **kwargs)
    assert is_dense_in_rows(data, **kwargs)


def assert_is_array_in_columns(data: Any, **kwargs: Any) -> None:
    assert is_array_in_columns(data, **kwargs)
    assert is_matrix_in_columns(data, **kwargs)
    assert is_grid_in_columns(data, **kwargs)
    assert is_dense_in_columns(data, **kwargs)


def assert_not_is_array2d(data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    assert not is_array2d(data, **kwargs)

    if kind in ("dense", "grid"):
        assert is_matrix(data, **kwargs)
    else:
        assert not is_matrix(data, **kwargs)

    if kind == "grid":
        assert is_grid(data, **kwargs)
    else:
        assert not is_grid(data, **kwargs)

    if kind == "dense":
        assert is_dense(data, **kwargs)
    else:
        assert not is_dense(data, **kwargs)


def assert_not_is_array_in_rows(data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    assert not is_array_in_rows(data, **kwargs)

    if kind in ("dense", "grid"):
        assert is_matrix_in_rows(data, **kwargs)
    else:
        assert not is_matrix_in_rows(data, **kwargs)

    if kind == "grid":
        assert is_grid_in_rows(data, **kwargs)
    else:
        assert not is_grid_in_rows(data, **kwargs)

    if kind == "dense":
        assert is_dense_in_rows(data, **kwargs)
    else:
        assert not is_dense_in_rows(data, **kwargs)


def assert_not_is_array_in_columns(data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    assert not is_array_in_columns(data, **kwargs)

    if kind in ("dense", "grid"):
        assert is_matrix_in_columns(data, **kwargs)
    else:
        assert not is_matrix_in_columns(data, **kwargs)

    if kind == "grid":
        assert is_grid_in_columns(data, **kwargs)
    else:
        assert not is_grid_in_columns(data, **kwargs)

    if kind == "dense":
        assert is_dense_in_columns(data, **kwargs)
    else:
        assert not is_dense_in_columns(data, **kwargs)


# pylint: enable=duplicate-code


def test_is_array2d() -> None:
    assert_is_array2d(np.array([[0]]))
    assert_is_array2d(np.array([[0]], dtype="bool"), dtype="bool")
    assert_is_array_in_rows(np.array([[0]]))
    assert_is_array_in_rows(np.array([[0]], dtype="bool"), dtype="bool")
    assert_is_array_in_columns(np.array([[0]]))
    assert_is_array_in_columns(np.array([[0]], dtype="bool"), dtype="bool")

    row_major = be_array2d(np.zeros((2, 2)))
    assert_is_array2d(row_major, layout=ROW_MAJOR)
    assert_not_is_array2d(row_major, layout=COLUMN_MAJOR)

    assert_is_array_in_rows(row_major)
    assert_is_array_in_rows(matrix_copy(row_major))
    assert_not_is_array_in_columns(row_major)

    assert_is_array_in_columns(row_major.T)
    assert_is_array_in_columns(matrix_copy(row_major.T))
    assert_not_is_array_in_rows(row_major.T)

    assert_not_is_array2d(np.array([[0]], dtype="bool"), dtype="int16")
    assert_not_is_array2d(np.array([0]))
    assert_not_is_array2d(pd.DataFrame([[0]]), kind="dense")

    for sparse_format in ("csr", "csc"):
        np.random.seed(123456)
        sparse = be_sparse(sp.random(10, 10, density=0.5, format=sparse_format))
        assert_not_is_array2d(sparse[0, :], kind="grid")
        assert_not_is_array2d(sparse[:, 0], kind="grid")

    assert_not_is_array2d(pd.Series([0.0]))
    assert_not_is_array2d(pd.DataFrame([[0]]), kind="dense")


# pylint: disable=duplicate-code


def assert_be_array2d(data: Any, **kwargs: Any) -> None:
    assert id(be_array2d(data, **kwargs)) == id(data), "be_array2d returned a different object"
    assert id(be_matrix(data, **kwargs)) == id(data), "be_matrix returned a different object"
    assert id(be_grid(data, **kwargs)) == id(data), "be_grid returned a different object"
    assert id(be_dense(data, **kwargs)) == id(data), "be_dense returned a different object"


def assert_be_array_in_rows(data: Any, **kwargs: Any) -> None:
    assert id(be_array_in_rows(data, **kwargs)) == id(data), "be_array_in_rows returned a different object"
    assert id(be_matrix_in_rows(data, **kwargs)) == id(data), "be_matrix_in_rows returned a different object"
    assert id(be_grid_in_rows(data, **kwargs)) == id(data), "be_grid_in_rows returned a different object"
    assert id(be_dense_in_rows(data, **kwargs)) == id(data), "be_dense_in_rows returned a different object"


def assert_be_array_in_columns(data: Any, **kwargs: Any) -> None:
    assert id(be_array_in_columns(data, **kwargs)) == id(data), "be_array_in_columns returned a different object"
    assert id(be_matrix_in_columns(data, **kwargs)) == id(data), "be_matrix_in_columns returned a different object"
    assert id(be_grid_in_columns(data, **kwargs)) == id(data), "be_grid_in_columns returned a different object"
    assert id(be_dense_in_columns(data, **kwargs)) == id(data), "be_dense_in_columns returned a different object"


def assert_not_be_array2d(message: str, data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    with expect_raise(message):
        be_array2d(data, **kwargs)

    if kind in ("grid", "dense"):
        assert id(be_matrix(data, **kwargs)) == id(data), "be_matrix returned a different object"
    else:
        with expect_raise(message.replace(r"numpy.ndarray", "matrix", 1)):
            be_matrix(data, **kwargs)

    if kind == "grid":
        assert id(be_grid(data, **kwargs)) == id(data), "be_grid returned a different object"
    else:
        with expect_raise(message.replace("numpy.ndarray", "grid", 1)):
            be_grid(data, **kwargs)

    if kind == "dense":
        assert id(be_dense(data, **kwargs)) == id(data), "be_dense returned a different object"
    else:
        with expect_raise(message.replace("numpy.ndarray", "dense matrix", 1)):
            be_dense(data, **kwargs)


def assert_not_be_array_in_rows(message: str, data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    with expect_raise(message):
        be_array_in_rows(data, **kwargs)

    if kind in ("grid", "dense"):
        assert id(be_matrix_in_rows(data, **kwargs)) == id(data), "be_matrix_in_rows returned a different object"
    else:
        with expect_raise(message.replace(r"numpy.ndarray", "matrix", 1)):
            be_matrix_in_rows(data, **kwargs)

    if kind == "grid":
        assert id(be_grid_in_rows(data, **kwargs)) == id(data), "be_grid_in_rows returned a different object"
    else:
        with expect_raise(message.replace("numpy.ndarray", "grid", 1)):
            be_grid_in_rows(data, **kwargs)

    if kind == "dense":
        assert id(be_dense_in_rows(data, **kwargs)) == id(data), "be_dense_in_rows returned a different object"
    else:
        with expect_raise(message.replace("numpy.ndarray", "dense matrix", 1)):
            be_dense_in_rows(data, **kwargs)


def assert_not_be_array_in_columns(message: str, data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    with expect_raise(message):
        be_array_in_columns(data, **kwargs)

    if kind in ("grid", "dense"):
        assert id(be_matrix_in_columns(data, **kwargs)) == id(data), "be_matrix_in_columns returned a different object"
    else:
        with expect_raise(message.replace(r"numpy.ndarray", "matrix", 1)):
            be_matrix_in_columns(data, **kwargs)

    if kind == "grid":
        assert id(be_grid_in_columns(data, **kwargs)) == id(data), "be_grid_in_columns returned a different object"
    else:
        with expect_raise(message.replace("numpy.ndarray", "grid", 1)):
            be_grid_in_columns(data, **kwargs)

    if kind == "dense":
        assert id(be_dense_in_columns(data, **kwargs)) == id(data), "be_dense_in_columns returned a different object"
    else:
        with expect_raise(message.replace("numpy.ndarray", "dense matrix", 1)):
            be_dense_in_columns(data, **kwargs)


# pylint: enable=duplicate-code


def test_be_array2d() -> None:
    assert_be_array2d(np.array([[0]]))
    assert_be_array2d(np.array([[0]], dtype="bool"), dtype="bool")
    assert_be_array_in_rows(np.array([[0]]))
    assert_be_array_in_rows(np.array([[0]], dtype="bool"), dtype="bool")
    assert_be_array_in_columns(np.array([[0]]))
    assert_be_array_in_columns(np.array([[0]], dtype="bool"), dtype="bool")

    row_major = be_array2d(np.zeros((2, 2)))
    assert_be_array2d(row_major, layout=ROW_MAJOR)
    assert_not_be_array2d(
        "expected column-major numpy.ndarray of any reasonable type, got row-major numpy.ndarray of 2x2 of float64",
        row_major,
        layout=COLUMN_MAJOR,
    )

    assert_be_array_in_rows(row_major)
    assert_be_array_in_rows(matrix_copy(row_major))
    assert_not_be_array_in_columns(
        "expected column-major numpy.ndarray of any reasonable type, got row-major numpy.ndarray of 2x2 of float64",
        row_major,
    )

    assert_be_array_in_columns(row_major.T)
    assert_be_array_in_columns(matrix_copy(row_major.T))
    assert_not_be_array_in_rows(
        "expected row-major numpy.ndarray of any reasonable type, got column-major numpy.ndarray of 2x2 of float64",
        row_major.T,
    )

    assert_not_be_array2d(
        "expected any-major numpy.ndarray of int16, got both-major numpy.ndarray of 1x1 of bool",
        np.array([[0]], dtype="bool"),
        dtype="int16",
    )
    assert_not_be_array2d(
        "expected any-major numpy.ndarray of any reasonable type, got both-major pandas Table of 1x1 of int64",
        data=pd.DataFrame([[0]]),
        kind="dense",
    )

    for sparse_format in ("csr", "csc"):
        np.random.seed(123456)
        sparse = sp.random(10, 10, density=0.5, format=sparse_format)
        assert_not_be_array2d(
            "expected any-major numpy.ndarray of any reasonable type, "
            f"got scipy.sparse.{sparse_format}_matrix of 1x10 of float64 with 40.00% nnz",
            sparse[0, :],
            kind="grid",
        )
        assert_not_be_array2d(
            "expected any-major numpy.ndarray of any reasonable type, "
            f"got scipy.sparse.{sparse_format}_matrix of 10x1 of float64 with 60.00% nnz",
            sparse[:, 0],
            kind="grid",
        )

    array1d = np.array([0.0])
    assert_not_be_array2d(
        "expected any-major numpy.ndarray of any reasonable type, got 1D numpy.ndarray of 1 of float64",
        array1d,
    )

    assert_not_be_array2d(
        "expected any-major numpy.ndarray of any reasonable type, got pandas.Series of 1 of float64", pd.Series([0.0])
    )

    allow_np_matrix()

    assert_not_be_array2d(
        "expected any-major numpy.ndarray of any reasonable type, got numpy.matrix of 2x1 of int64",
        np.matrix([[0], [1]]),
    )


# pylint: disable=duplicate-code


def assert_as_is_array2d(data: Any, expected_id: Optional[int] = None) -> None:
    if expected_id is None:
        expected_id = id(data)

    assert id(as_array2d(data)) == expected_id, "as_array2d returned a different object"
    assert id(as_array2d(data, force_copy=True)) != expected_id, "as_array2d did not force a copy"


def assert_as_can_be_array2d(data: Any) -> None:
    assert id(as_array2d(data)) != id(data), "as_array2d returned same object"


# pylint: enable=duplicate-code


def test_as_array2d() -> None:
    assert_as_is_array2d(np.array([[0]]))

    table = pd.DataFrame(np.array([[0]]))
    assert_as_is_array2d(table, id(table.values))

    np.random.seed(123456)
    for sparse_format in ("csr", "csc", "coo"):
        sparse = sp.random(10, 10, density=0.5, format=sparse_format)
        assert_as_can_be_array2d(sparse)

    allow_np_matrix()

    assert_as_can_be_array2d(np.matrix([[0], [1]]))

    assert_as_can_be_array2d([[1, 2], [3, 4]])
