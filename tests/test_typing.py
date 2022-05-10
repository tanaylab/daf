"""
Test typing.
"""

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-function-docstring,too-many-locals,self-assigning-variable,too-many-statements


def test_array1d() -> None:
    array1d_of_bool: Array1D = be_array1d(np.zeros(1, dtype="bool"))
    be_array1d(array1d_of_bool, dtype="bool")
    be_vector(array1d_of_bool)
    assert not is_array1d(array1d_of_bool, dtype="int")


def test_series() -> None:
    series_of_bool: Series = be_series(pd.Series(np.zeros(1, dtype="bool")))
    be_series(series_of_bool, dtype="bool")
    assert not is_series(series_of_bool, dtype="int")
    be_array1d(as_array1d(series_of_bool))
    be_vector(series_of_bool, dtype="bool")

    series_of_str: Series = be_series(pd.Series([""], dtype="string"))
    be_series(series_of_str, dtype="str")
    be_array1d(as_array1d(series_of_str), dtype="str")
    be_vector(series_of_str)

    series_of_cat: Series = be_series(pd.Series([""], dtype="category"))
    be_series(series_of_cat, dtype="str")
    be_array1d(as_array1d(series_of_cat), dtype="str")
    be_vector(series_of_cat)


def test_array2d() -> None:
    array2d_of_int: Array2D = be_array2d(np.zeros((2, 3), dtype="int64"))
    be_array2d(array2d_of_int, dtype="int64")
    assert not is_array2d(array2d_of_int, dtype="bool")

    be_array_rows(array2d_of_int, dtype="int64")
    be_array_columns(array2d_of_int.T)
    be_matrix(array2d_of_int)
    be_matrix_rows(array2d_of_int)

    be_array1d(array2d_of_int[:, 0])
    be_array1d(array2d_of_int[0, :])

    array2d_of_str: Array2D = be_array2d(np.zeros((2, 3), dtype="U"))
    be_array2d(array2d_of_str, dtype="str")

    be_array_rows(array2d_of_str)
    be_array_columns(array2d_of_str.T, dtype="str")
    be_matrix(array2d_of_str)
    be_matrix_columns(array2d_of_str.T)

    be_array1d(array2d_of_str[:, 0])
    be_array1d(array2d_of_str[0, :])

    row_of_int: Array2D = be_array2d(np.zeros((1, 3), dtype="int64"))
    be_array1d(as_array1d(row_of_int))

    column_of_str: Array2D = be_array2d(np.zeros((2, 1), dtype="U"))
    be_array1d(as_array1d(column_of_str))


def test_frame() -> None:
    frame_of_int: Frame = be_frame(
        pd.DataFrame(np.zeros((2, 3), dtype="int64"), index=["a", "b"], columns=["x", "y", "z"])
    )
    be_frame(frame_of_int, dtype="int64")
    assert not is_frame(frame_of_int, dtype="bool")
    be_matrix(frame_of_int)
    be_matrix_rows(frame_of_int)

    array2d_of_int: Array2D = as_array2d(frame_of_int)
    be_array2d(array2d_of_int)

    be_frame_rows(frame_of_int, dtype="int64")
    be_frame_columns(frame_of_int.T)

    be_series(frame_of_int["x"])
    be_series(frame_of_int.loc["a", :])
    be_series(frame_of_int.iloc[0, :])
    be_series(frame_of_int.iloc[:, 0])

    frame_of_str: Frame = be_frame(pd.DataFrame(np.zeros((2, 3), dtype="U"), index=["a", "b"], columns=["x", "y", "z"]))
    be_frame(frame_of_str, dtype="str")

    be_frame_rows(frame_of_str)
    be_frame_columns(frame_of_str.T, dtype="str")

    array2d_of_str: Array2D = as_array2d(frame_of_str)
    be_array2d(array2d_of_str)
    be_matrix(frame_of_str)
    be_matrix_columns(frame_of_str.T)

    be_series(frame_of_str["x"])
    be_series(frame_of_str.loc["a", :])
    be_series(frame_of_str.iloc[0, :])
    be_series(frame_of_str.iloc[:, 0])

    row_of_int: Frame = be_frame(pd.DataFrame(np.zeros((1, 3), dtype="int64"), index=["a"], columns=["x", "y", "z"]))
    be_array1d(as_array1d(row_of_int))

    column_of_str: Frame = be_frame(pd.DataFrame(np.zeros((2, 1), dtype="U"), index=["a", "b"], columns=["x"]))
    be_array1d(as_array1d(column_of_str))

    frame_of_mix: Frame = be_frame(pd.DataFrame(dict(a=[1, 2, 3], b=["X", "Y", "Z"]), index=["x", "y", "z"]))

    be_series(frame_of_mix["a"], dtype="int64")
    be_series(frame_of_mix["b"], dtype="str")
    be_frame_columns(frame_of_mix)


def test_sparse() -> None:
    sparse_of_float32: Sparse = be_sparse(sp.csr_matrix([[1, 0], [0, 1]], dtype="float32"))
    be_sparse(sparse_of_float32, dtype="float32")
    assert not is_sparse(sparse_of_float32, dtype="float16")
    be_matrix(sparse_of_float32, dtype="float32")
    be_matrix_rows(sparse_of_float32, dtype="float32")

    be_array1d(as_array1d(sparse_of_float32[0, :]))
    be_array1d(as_array1d(sparse_of_float32[:, 0]))

    be_sparse_rows(sparse_of_float32, dtype="float32")
    be_sparse_rows(sparse_of_float32)
    be_sparse_columns(sparse_of_float32.T)
    be_sparse_columns(sparse_of_float32.T, dtype="float32")
    be_matrix(sparse_of_float32.T)
    be_matrix_columns(sparse_of_float32.T)
