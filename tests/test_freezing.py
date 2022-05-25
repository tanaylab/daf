"""
Test ``daf.typing.freezing``.
"""

# pylint: disable=duplicate-code

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.typing.array1d import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.array2d import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.freezing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.grids import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.tables import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing.vectors import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_raise

# pylint: disable=duplicate-code

# pylint: disable=missing-function-docstring


def check_vector(vector: Vector) -> None:
    assert not is_frozen(vector)
    vector[1] = 1
    assert vector[1] == 1

    vector = freeze(vector)

    assert is_frozen(vector)
    with expect_raise("assignment destination is read-only"):
        vector[1] = 2
    assert vector[1] == 1

    with unfrozen(vector) as melted:
        assert not is_frozen(melted)
        melted[1] = 2

    assert is_frozen(vector)
    assert vector[1] == 2
    with expect_raise("assignment destination is read-only"):
        vector[1] = 3
    assert vector[1] == 2

    vector = unfreeze(vector)

    assert not is_frozen(vector)
    vector[1] = 3
    assert vector[1] == 3


def test_array1d() -> None:
    check_vector(be_array1d(np.zeros(10)))
    check_vector(pd.Series(np.zeros(10)))


def check_grid(grid: Grid) -> None:
    assert not is_frozen(grid)
    grid[1, 1] = 1
    assert grid[1, 1] == 1

    grid = freeze(grid)
    assert is_frozen(grid)
    with expect_raise("assignment destination is read-only"):
        grid[1, 1] = 2
    assert grid[1, 1] == 1

    with unfrozen(grid) as melted:
        assert not is_frozen(melted)
        melted[1, 1] = 2

    assert is_frozen(grid)
    assert grid[1, 1] == 2
    with expect_raise("assignment destination is read-only"):
        grid[1, 1] = 3
    assert grid[1, 1] == 2

    grid = unfreeze(grid)

    assert not is_frozen(grid)
    grid[1, 1] = 3
    assert grid[1, 1] == 3


def test_grid() -> None:
    check_grid(be_array2d(np.zeros((10, 10))))

    data = np.zeros((10, 10))
    data[1, 1] = -1
    check_grid(sp.csr_matrix(data))

    data = np.zeros((10, 10))
    data[1, 1] = -1
    check_grid(sp.csc_matrix(data))


def test_table() -> None:
    table = pd.DataFrame(np.zeros((10, 10)))

    assert not is_frozen(table)
    table.iloc[1, 1] = 1
    assert table.iloc[1, 1] == 1

    table = freeze(table)
    assert is_frozen(table)
    with expect_raise("assignment destination is read-only"):
        table.iloc[1, 1] = 2
    assert table.iloc[1, 1] == 1

    with unfrozen(table) as melted:
        assert not is_frozen(melted)
        melted.iloc[1, 1] = 2

    assert is_frozen(table)
    assert table.iloc[1, 1] == 2
    with expect_raise("assignment destination is read-only"):
        table.iloc[1, 1] = 3
    assert table.iloc[1, 1] == 2

    table = unfreeze(table)

    assert not is_frozen(table)
    table.iloc[1, 1] = 3
    assert table.iloc[1, 1] == 3
