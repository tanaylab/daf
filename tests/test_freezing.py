"""
Test ``daf.typing.freezing``.
"""

# pylint: disable=duplicate-code

from typing import Union

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.dense import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.fake_pandas import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.frames import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.freezing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.matrices import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.vectors import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_raise

# pylint: disable=duplicate-code

# pylint: disable=missing-function-docstring


def check_data1d(data1d: Union[Vector, Series]) -> None:
    assert not is_frozen(data1d)
    data1d[1] = 1
    assert data1d[1] == 1

    data1d = freeze(data1d)

    assert is_frozen(data1d)
    with expect_raise("assignment destination is read-only"):
        data1d[1] = 2
    assert data1d[1] == 1

    with unfrozen(data1d) as melted:
        assert not is_frozen(melted)
        melted[1] = 2

    assert is_frozen(data1d)
    assert data1d[1] == 2
    with expect_raise("assignment destination is read-only"):
        data1d[1] = 3
    assert data1d[1] == 2

    data1d = unfreeze(data1d)

    assert not is_frozen(data1d)
    data1d[1] = 3
    assert data1d[1] == 3


def test_data1d() -> None:
    check_data1d(be_vector(np.zeros(10)))
    check_data1d(pd.Series(np.zeros(10)))


def check_matrix(matrix: Matrix) -> None:
    assert not is_frozen(matrix)
    matrix[1, 1] = 1
    assert matrix[1, 1] == 1

    matrix = freeze(matrix)
    assert is_frozen(matrix)
    with expect_raise("assignment destination is read-only"):
        matrix[1, 1] = 2
    assert matrix[1, 1] == 1

    with unfrozen(matrix) as melted:
        assert not is_frozen(melted)
        melted[1, 1] = 2

    assert is_frozen(matrix)
    assert matrix[1, 1] == 2
    with expect_raise("assignment destination is read-only"):
        matrix[1, 1] = 3
    assert matrix[1, 1] == 2

    matrix = unfreeze(matrix)

    assert not is_frozen(matrix)
    matrix[1, 1] = 3
    assert matrix[1, 1] == 3


def test_matrix() -> None:
    check_matrix(be_dense(np.zeros((10, 10))))

    data = np.zeros((10, 10))
    data[1, 1] = -1
    check_matrix(sp.csr_matrix(data))

    data = np.zeros((10, 10))
    data[1, 1] = -1
    check_matrix(sp.csc_matrix(data))


def test_frame() -> None:
    frame = pd.DataFrame(np.zeros((10, 10)))

    assert not is_frozen(frame)
    frame.iloc[1, 1] = 1
    assert frame.iloc[1, 1] == 1

    frame = freeze(frame)
    assert is_frozen(frame)
    with expect_raise("assignment destination is read-only"):
        frame.iloc[1, 1] = 2
    assert frame.iloc[1, 1] == 1

    with unfrozen(frame) as melted:
        assert not is_frozen(melted)
        melted.iloc[1, 1] = 2

    assert is_frozen(frame)
    assert frame.iloc[1, 1] == 2
    with expect_raise("assignment destination is read-only"):
        frame.iloc[1, 1] = 3
    assert frame.iloc[1, 1] == 2

    frame = unfreeze(frame)

    assert not is_frozen(frame)
    frame.iloc[1, 1] = 3
    assert frame.iloc[1, 1] == 3
