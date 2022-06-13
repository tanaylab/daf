"""
Test ``daf.typing.dtypes``.
"""

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.dense import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.dtypes import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.vectors import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-function-docstring


def test_bool_dtype() -> None:
    for data in (np.array([0], dtype="bool"), pd.Series([0], dtype="bool")):
        assert is_dtype(data.dtype)
        assert is_dtype(data.dtype, "bool")
        assert not is_dtype(data.dtype, INT_DTYPES)
        assert not is_dtype(data.dtype, FLOAT_DTYPES)
        assert not is_dtype(data.dtype, NUM_DTYPES)
        assert not is_dtype(data.dtype, STR_DTYPE)


def test_int_dtypes() -> None:
    for dtype in INT_DTYPES:
        for data in (np.array([0], dtype=dtype), pd.Series([0], dtype=dtype)):
            assert is_dtype(data.dtype)
            assert not is_dtype(data.dtype, "bool")
            assert is_dtype(data.dtype, dtype)
            assert is_dtype(data.dtype, INT_DTYPES)
            assert not is_dtype(data.dtype, FLOAT_DTYPES)
            assert is_dtype(data.dtype, NUM_DTYPES)
            assert not is_dtype(data.dtype, STR_DTYPE)


def test_float_dtypes() -> None:
    for dtype in FLOAT_DTYPES:
        for data in (np.array([0.5], dtype=dtype), pd.Series([0.5], dtype=dtype)):
            assert is_dtype(data.dtype)
            assert not is_dtype(data.dtype, "bool")
            assert is_dtype(data.dtype, dtype)
            assert not is_dtype(data.dtype, INT_DTYPES)
            assert is_dtype(data.dtype, FLOAT_DTYPES)
            assert is_dtype(data.dtype, NUM_DTYPES)
            assert not is_dtype(data.dtype, STR_DTYPE)


def test_str_dtype() -> None:
    for values in (["a", "a"], ["a", "ab"]):
        for data in (
            np.array(values),
            np.array(values, dtype="U"),
            np.array(["a", "bc"], dtype="object"),
            pd.Series(values),
            pd.Series(values, dtype="U"),
            pd.Series(values, dtype="object"),
            pd.Series(values, dtype="category"),
        ):
            assert is_dtype(data.dtype)
            assert not is_dtype(data.dtype, "bool")
            assert not is_dtype(data.dtype, INT_DTYPES)
            assert not is_dtype(data.dtype, FLOAT_DTYPES)
            assert not is_dtype(data.dtype, NUM_DTYPES)
            assert is_dtype(data.dtype, STR_DTYPE)


def test_dtype_of() -> None:
    assert str(dtype_of(be_vector(np.array([0], dtype="int8")))) == "int8"
    assert str(dtype_of(be_dense(np.array([[0]], dtype="int8")))) == "int8"
    assert str(dtype_of(sp.csr_matrix([[0]], dtype="int8"))) == "int8"
    assert str(dtype_of(sp.csc_matrix([[0]], dtype="int8"))) == "int8"
    assert str(dtype_of(pd.DataFrame([[0]], dtype="int8"))) == "int8"
    assert str(dtype_of(pd.Series([0], dtype="int8"))) == "int8"
