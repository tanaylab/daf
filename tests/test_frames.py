"""
Test ``daf.typing.frames``.
"""

# pylint: disable=duplicate-code

from typing import Any
from typing import Optional

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.dense import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.frames import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.grids import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.layouts import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.matrices import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import allow_np_matrix
from . import expect_raise

# pylint: disable=missing-function-docstring


def assert_is_frame(data: Any, **kwargs: Any) -> None:
    assert is_frame(data, **kwargs)
    assert is_matrix(data, **kwargs)
    assert is_dense(data, **kwargs)


def assert_is_frame_in_rows(data: Any, **kwargs: Any) -> None:
    assert is_frame_in_rows(data, **kwargs)
    assert is_matrix_in_rows(data, **kwargs)
    assert is_dense_in_rows(data, **kwargs)


def assert_is_frame_in_columns(data: Any, **kwargs: Any) -> None:
    assert is_frame_in_columns(data, **kwargs)
    assert is_matrix_in_columns(data, **kwargs)
    assert is_dense_in_columns(data, **kwargs)


def assert_not_is_frame(data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    assert not is_frame(data, **kwargs)

    if kind in ("grid", "dense"):
        assert is_matrix(data, **kwargs)
    else:
        assert not is_matrix(data, **kwargs)

    if kind in ("grid", "dense"):
        assert is_grid(data, **kwargs)
    else:
        assert not is_grid(data, **kwargs)

    if kind == "dense":
        assert is_dense(data, **kwargs)
    else:
        assert not is_dense(data, **kwargs)


def assert_not_is_frame_in_rows(data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    assert not is_frame_in_rows(data, **kwargs)

    if kind in ("grid", "dense"):
        assert is_matrix_in_rows(data, **kwargs)
    else:
        assert not is_matrix_in_rows(data, **kwargs)

    if kind in ("grid", "dense"):
        assert is_grid_in_rows(data, **kwargs)
    else:
        assert not is_grid_in_rows(data, **kwargs)

    if kind == "dense":
        assert is_dense_in_rows(data, **kwargs)
    else:
        assert not is_dense_in_rows(data, **kwargs)


def assert_not_is_frame_in_columns(data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    assert not is_frame_in_columns(data, **kwargs)

    if kind in ("grid", "dense"):
        assert is_matrix_in_columns(data, **kwargs)
    else:
        assert not is_matrix_in_columns(data, **kwargs)

    if kind in ("grid", "dense"):
        assert is_grid_in_columns(data, **kwargs)
    else:
        assert not is_grid_in_columns(data, **kwargs)

    if kind == "dense":
        assert is_dense_in_columns(data, **kwargs)
    else:
        assert not is_dense_in_columns(data, **kwargs)


# pylint: enable=duplicate-code


def test_is_frame() -> None:
    assert_is_frame(pd.DataFrame([[0]]))
    assert_is_frame(pd.DataFrame([[0]], dtype="bool"), dtype="bool")
    assert_is_frame_in_rows(pd.DataFrame([[0]]))
    assert_is_frame_in_rows(pd.DataFrame([[0]], dtype="bool"), dtype="bool")
    assert_is_frame_in_columns(pd.DataFrame([[0]]))
    assert_is_frame_in_columns(pd.DataFrame([[0]], dtype="bool"), dtype="bool")

    row_major = pd.DataFrame(np.zeros((2, 2)))
    assert_is_frame(row_major, layout=ROW_MAJOR)
    assert_not_is_frame(row_major, layout=COLUMN_MAJOR)

    assert_is_frame_in_rows(row_major)
    assert_is_frame_in_rows(matrix_copy(row_major))
    assert_not_is_frame_in_columns(row_major)

    assert_is_frame_in_columns(row_major.T)
    assert_is_frame_in_columns(matrix_copy(row_major.T))
    assert_not_is_frame_in_rows(row_major.T)

    assert_not_is_frame(pd.DataFrame([[0]], dtype="bool"), dtype="int16")
    assert_not_is_frame(np.array([[0]]), kind="dense")

    for sparse_format in ("csr", "csc"):
        np.random.seed(123456)
        sparse = sp.random(10, 10, density=0.5, format=sparse_format)
        assert_not_is_frame(sparse[0, :], kind="grid")
        assert_not_is_frame(sparse[:, 0], kind="grid")

    assert_not_is_frame(pd.Series([0.0]))

    allow_np_matrix()

    assert_not_is_frame(np.matrix([[0], [1]]))

    assert_not_is_frame(pd.DataFrame(dict(a=[0], b=["1"])))


# pylint: disable=duplicate-code


def assert_be_frame(data: Any, **kwargs: Any) -> None:
    assert id(be_frame(data, **kwargs)) == id(data), "be_frame returned a different object"
    assert id(be_matrix(data, **kwargs)) == id(data), "be_matrix returned a different object"
    assert id(be_dense(data, **kwargs)) == id(data), "be_dense returned a different object"


def assert_be_frame_in_rows(data: Any, **kwargs: Any) -> None:
    assert id(be_frame_in_rows(data, **kwargs)) == id(data), "be_frame_in_rows returned a different object"
    assert id(be_matrix_in_rows(data, **kwargs)) == id(data), "be_matrix_in_rows returned a different object"
    assert id(be_dense_in_rows(data, **kwargs)) == id(data), "be_dense_in_rows returned a different object"


def assert_be_frame_in_columns(data: Any, **kwargs: Any) -> None:
    assert id(be_frame_in_columns(data, **kwargs)) == id(data), "be_frame_in_columns returned a different object"
    assert id(be_matrix_in_columns(data, **kwargs)) == id(data), "be_matrix_in_columns returned a different object"
    assert id(be_dense_in_columns(data, **kwargs)) == id(data), "be_dense_in_columns returned a different object"


def assert_not_be_frame(message: str, data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    with expect_raise(message):
        be_frame(data, **kwargs)

    if kind in ("grid", "dense"):
        assert id(be_matrix(data, **kwargs)) == id(data), "be_matrix returned a different object"
    else:
        with expect_raise(message.replace(r"pandas.DataFrame", "matrix", 1)):
            be_matrix(data, **kwargs)

    if kind == "dense":
        assert id(be_dense(data, **kwargs)) == id(data), "be_dense returned a different object"
    else:
        with expect_raise(message.replace("pandas.DataFrame", "dense matrix", 1)):
            be_dense(data, **kwargs)


def assert_not_be_frame_in_rows(message: str, data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    with expect_raise(message):
        be_frame_in_rows(data, **kwargs)

    if kind in ("grid", "dense"):
        assert id(be_matrix_in_rows(data, **kwargs)) == id(data), "be_matrix_in_rows returned a different object"
    else:
        with expect_raise(message.replace(r"pandas.DataFrame", "matrix", 1)):
            be_matrix_in_rows(data, **kwargs)

    if kind == "dense":
        assert id(be_dense_in_rows(data, **kwargs)) == id(data), "be_dense_in_rows returned a different object"
    else:
        with expect_raise(message.replace("pandas.DataFrame", "dense matrix", 1)):
            be_dense_in_rows(data, **kwargs)


def assert_not_be_frame_in_columns(message: str, data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    with expect_raise(message):
        be_frame_in_columns(data, **kwargs)

    if kind in ("grid", "dense"):
        assert id(be_matrix_in_columns(data, **kwargs)) == id(data), "be_matrix_in_columns returned a different object"
    else:
        with expect_raise(message.replace(r"pandas.DataFrame", "matrix", 1)):
            be_matrix_in_columns(data, **kwargs)

    if kind == "dense":
        assert id(be_dense_in_columns(data, **kwargs)) == id(data), "be_dense_in_columns returned a different object"
    else:
        with expect_raise(message.replace("pandas.DataFrame", "dense matrix", 1)):
            be_dense_in_columns(data, **kwargs)


# pylint: enable=duplicate-code


def test_be_frame() -> None:
    assert_be_frame(pd.DataFrame([[0]]))
    assert_be_frame(pd.DataFrame([[0]], dtype="bool"), dtype="bool")
    assert_be_frame_in_rows(pd.DataFrame([[0]]))
    assert_be_frame_in_rows(pd.DataFrame([[0]], dtype="bool"), dtype="bool")
    assert_be_frame_in_columns(pd.DataFrame([[0]]))
    assert_be_frame_in_columns(pd.DataFrame([[0]], dtype="bool"), dtype="bool")

    row_major = pd.DataFrame(np.zeros((2, 2)))
    assert_be_frame(row_major, layout=ROW_MAJOR)
    assert_not_be_frame(
        "expected column-major pandas.DataFrame of any reasonable type, "
        "got row-major pandas.DataFrame of 2x2 of float64",
        row_major,
        layout=COLUMN_MAJOR,
    )

    assert_be_frame_in_rows(row_major)
    assert_be_frame_in_rows(matrix_copy(row_major))
    assert_not_be_frame_in_columns(
        "expected column-major pandas.DataFrame of any reasonable type, "
        "got row-major pandas.DataFrame of 2x2 of float64",
        row_major,
    )

    assert_be_frame_in_columns(row_major.T)
    assert_be_frame_in_columns(matrix_copy(row_major.T))
    assert_not_be_frame_in_rows(
        "expected row-major pandas.DataFrame of any reasonable type, "
        "got column-major pandas.DataFrame of 2x2 of float64",
        row_major.T,
    )

    assert_not_be_frame(
        "expected any-major pandas.DataFrame of int16, got both-major pandas.DataFrame of 1x1 of bool",
        pd.DataFrame([[0]], dtype="bool"),
        dtype="int16",
    )
    assert_not_be_frame(
        "expected any-major pandas.DataFrame of any reasonable type, got both-major numpy.ndarray of 1x1 of int64",
        data=np.array([[0]]),
        kind="dense",
    )

    for sparse_format in ("csr", "csc"):
        np.random.seed(123456)
        sparse = sp.random(10, 10, density=0.5, format=sparse_format)
        assert_not_be_frame(
            "expected any-major pandas.DataFrame of any reasonable type, "
            f"got scipy.sparse.{sparse_format}_matrix of 1x10 of float64 with 40.00% nnz",
            sparse[0, :],
            kind="grid",
        )
        assert_not_be_frame(
            "expected any-major pandas.DataFrame of any reasonable type, "
            f"got scipy.sparse.{sparse_format}_matrix of 10x1 of float64 with 60.00% nnz",
            sparse[:, 0],
            kind="grid",
        )

    assert_not_be_frame(
        "expected any-major pandas.DataFrame of any reasonable type, got pandas.Series of 1 of float64",
        pd.Series([0.0]),
    )

    allow_np_matrix()

    assert_not_be_frame(
        "expected any-major pandas.DataFrame of any reasonable type, got numpy.matrix of 2x1 of int64",
        np.matrix([[0], [1]]),
    )

    assert_not_be_frame(
        "expected any-major pandas.DataFrame of any reasonable type, got pandas.DataFrame of 1x2 of mixed types",
        pd.DataFrame(dict(a=[0], b=["1"])),
    )
