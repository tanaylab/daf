"""
Test ``daf.typing.frames``.
"""

from typing import Any
from typing import Optional

import numpy as np
import pandas as pd  # type: ignore

from daf.typing.frames import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.matrices import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_raise

# pylint: disable=missing-function-docstring

# pylint: disable=duplicate-code


def assert_is_frame(data: Any, kind: Optional[str] = None, **kwargs: Any) -> None:
    assert is_frame(data, **kwargs)
    if kind == "matrix":
        assert is_matrix(data, **kwargs)
    else:
        assert not is_matrix(data, **kwargs)


def assert_not_is_frame(data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    assert not is_frame(data, **kwargs)
    if kind == "matrix":
        assert is_matrix(data, **kwargs)
    else:
        assert not is_matrix(data, **kwargs)


# pylint: enable=duplicate-code


def test_is_frame() -> None:
    assert_is_frame(pd.DataFrame(dict(a=[0], b=["1"])))
    assert_is_frame(pd.DataFrame([[0], [1]]), kind="matrix")

    assert_not_is_frame(np.array([0]))
    assert_not_is_frame(np.array([[0]]), kind="matrix")


# pylint: disable=duplicate-code


def assert_be_frame(data: Any, kind: Optional[str] = None, **kwargs: Any) -> None:
    assert id(be_frame(data, **kwargs)) == id(data), "be_frame returned a different object"

    if kind == "matrix":
        assert id(be_matrix(data, **kwargs)) == id(data), "be_matrix returned a different object"


def assert_not_be_frame(message: str, data: Any, *, kind: Optional[str] = None, **kwargs: Any) -> None:
    with expect_raise(message):
        be_frame(data, **kwargs)

    if kind == "matrix":
        assert id(be_matrix(data, **kwargs)) == id(data), "be_matrix returned a different object"
    else:
        with expect_raise(
            message.replace("expected pandas.DataFrame", "expected any-major matrix of any reasonable type")
        ):
            be_matrix(data, **kwargs)


# pylint: enable=duplicate-code


def test_be_frame() -> None:
    assert_be_frame(pd.DataFrame(dict(a=[0], b=["1"])))
    assert_be_frame(pd.DataFrame([[0], [1]]))

    assert_not_be_frame("expected pandas.DataFrame, got 1D numpy.ndarray of 1 of int64", np.array([0]))
    assert_not_be_frame(
        "expected pandas.DataFrame, got both-major numpy.ndarray of 1x1 of int64", np.array([[0]]), kind="matrix"
    )
