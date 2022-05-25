"""
Test ``daf.typing.optimization``.
"""

# pylint: disable=duplicate-code

from typing import Any

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.array1d import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.array2d import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.layouts import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.optimization import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import allow_np_matrix

# pylint: enable=duplicate-code

# pylint: disable=missing-function-docstring


def assert_is_optimal(data: Any) -> None:
    assert is_optimal(data)
    assert id(optimize(data)) == id(data)


def assert_not_is_optimal(data: Any, preferred_layout: AnyMajor = ROW_MAJOR) -> None:
    assert not is_optimal(data)
    optimized = be_optimal(optimize(data, preferred_layout=preferred_layout))
    assert id(optimized) != id(data)

    if data.ndim == 1:
        assert np.all(as_array1d(data) == as_array1d(optimized))
    else:
        assert np.all(as_array2d(data) == as_array2d(optimized))


def test_optimize() -> None:
    np.random.seed(123456)
    assert_is_optimal(np.random.rand(10))
    assert_is_optimal(pd.Series(np.random.rand(10)))

    assert_not_is_optimal(np.random.rand(10)[0:10:2])
    assert_not_is_optimal(pd.Series(np.random.rand(10)[0:10:2]))

    assert_is_optimal(np.random.rand(10, 20))
    assert_is_optimal(pd.DataFrame(np.random.rand(10, 20)))

    assert_is_optimal(pd.Series(["a", "b"]))
    assert_not_is_optimal(pd.Series(["a", "b"], dtype="category"))

    assert_is_optimal(pd.DataFrame([["a", "b"], ["c", "d"]]))
    assert_not_is_optimal(pd.DataFrame([["a", "b"], ["c", "d"]], dtype="category"))

    assert_not_is_optimal(np.random.rand(10, 20)[0:10:2, 0:20:5])
    assert_not_is_optimal(pd.DataFrame(np.random.rand(10, 20)[0:10:2, 0:20:5]))

    allow_np_matrix()

    assert_not_is_optimal(np.matrix([[0, 1]]))

    assert_is_optimal(sp.random(10, 20, density=0.5, format="csr"))
    assert_is_optimal(sp.random(10, 20, density=0.5, format="csc"))
    assert_not_is_optimal(sp.random(10, 20, density=0.5, format="coo"))
    assert_not_is_optimal(sp.random(10, 20, density=0.5, format="coo"), preferred_layout=COLUMN_MAJOR)
