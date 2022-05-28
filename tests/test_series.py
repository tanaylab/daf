"""
Test ``daf.typing.series``.
"""

from typing import Any
from typing import Optional

import numpy as np
import pandas as pd  # type: ignore

from daf.typing.series import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.vectors import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_raise

# pylint: disable=missing-function-docstring

# pylint: disable=duplicate-code


def assert_is_series(data: Any, **kwargs: Any) -> None:
    assert is_series(data, **kwargs)
    assert is_vector(data, **kwargs)


def assert_not_is_series(data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    assert not is_series(data, **kwargs)

    if kind == "vector":
        assert is_vector(data, **kwargs)
    else:
        assert not is_vector(data, **kwargs)


# pylint: enable=duplicate-code


def test_is_series() -> None:
    assert_is_series(pd.Series([0]))
    assert_is_series(pd.Series([0], dtype="bool"), dtype="bool")

    assert_not_is_series(pd.Series([0], dtype="bool"), dtype="int16")
    assert_not_is_series(pd.DataFrame([[0], [1]]))
    assert_not_is_series(pd.DataFrame([[0, 1]]))
    assert_not_is_series(np.array([0]), kind="vector")


# pylint: disable=duplicate-code


def assert_be_series(data: Any, **kwargs: Any) -> None:
    assert id(be_series(data, **kwargs)) == id(data), "be_series returned a different object"
    assert id(be_vector(data, **kwargs)) == id(data), "be_vector returned a different object"


def assert_not_be_series(message: str, data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    with expect_raise(message):
        be_series(data, **kwargs)

    if kind == "vector":
        assert id(be_vector(data, **kwargs)) == id(data), "be_vector returned a different object"
    else:
        with expect_raise(message.replace("expected pandas.Series", "expected vector")):
            be_vector(data, **kwargs)


# pylint: enable=duplicate-code


def test_be_series() -> None:
    assert_be_series(pd.Series([0]))
    assert_be_series(pd.Series([0], dtype="bool"), dtype="bool")

    assert_not_be_series(
        "expected pandas.Series of int16, got pandas.Series of 1 of bool",
        pd.Series([0], dtype="bool"),
        dtype="int16",
    )
    assert_not_be_series(
        "expected pandas.Series of any reasonable type, got both-major pandas.DataFrame of 2x1 of int64",
        pd.DataFrame([[0], [1]]),
    )
    assert_not_be_series(
        "expected pandas.Series of any reasonable type, got 1D numpy.ndarray of 1 of int64",
        np.array([0]),
        kind="vector",
    )
