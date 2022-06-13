"""
Test ``daf.typing.series``.
"""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore

from daf.typing.series import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.unions import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_raise

# pylint: disable=missing-function-docstring


def assert_is_series(data: Any, **kwargs: Any) -> None:
    assert is_series(data, **kwargs)
    assert is_proper1d(data, **kwargs)
    assert is_proper(data, **kwargs)
    assert is_known1d(data)
    assert is_known(data)


def test_is_series() -> None:
    assert_is_series(pd.Series([0]))
    assert_is_series(pd.Series([0], dtype="bool"), dtype="bool")

    assert not is_series(pd.Series([0], dtype="bool"), dtype="int16")
    assert not is_series(pd.DataFrame([[0], [1]]))
    assert not is_series(pd.DataFrame([[0, 1]]))
    assert not is_series(np.array([0]))


def assert_be_series(data: Any, **kwargs: Any) -> None:
    assert id(be_series(data, **kwargs)) == id(data)
    assert id(be_proper1d(data, **kwargs)) == id(data)
    assert id(be_proper(data, **kwargs)) == id(data)
    assert id(be_known1d(data)) == id(data)
    assert id(be_known(data)) == id(data)


def assert_not_be_series(message: str, data: Any, **kwargs: Any) -> None:
    with expect_raise(message):
        be_series(data, **kwargs)


def test_be_series() -> None:
    assert_be_series(pd.Series([0]))
    assert_be_series(pd.Series([0], dtype="bool"), dtype="bool")

    assert_not_be_series(
        "expected: pandas.Series of int16, got: pandas.Series of 1 of bool",
        pd.Series([0], dtype="bool"),
        dtype="int16",
    )
    assert_not_be_series(
        "expected: pandas.Series of any reasonable type, got: both-major pandas.DataFrame of 2x1 of int64",
        pd.DataFrame([[0], [1]]),
    )
    assert_not_be_series(
        "expected: pandas.Series of any reasonable type, got: 1D numpy.ndarray of 1 of int64",
        np.array([0]),
    )
