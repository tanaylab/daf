"""
Test ``daf.storage.chain``.
"""

import numpy as np

from daf.storage.chains import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.memory import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.none import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-function-docstring


def test_chain_datum() -> None:
    base = MemoryStorage(name="base")
    base.set_datum("base_datum", "base value")
    base.set_datum("both_datum", "both base value")

    delta = MemoryStorage(name="delta")
    delta.set_datum("delta_datum", "delta value")
    delta.set_datum("both_datum", "both delta value")

    chain = StorageChain([delta, NO_STORAGE, base.as_reader()], name="chain")

    assert set(chain.datum_names()) == set(["base_datum", "both_datum", "delta_datum"])

    assert chain.get_datum("base_datum") == "base value"
    assert chain.get_datum("both_datum") == "both delta value"
    assert chain.get_datum("delta_datum") == "delta value"

    assert not chain.has_datum("neither_datum")


def test_chain_axis() -> None:
    base = MemoryStorage(name="base")
    base_names = freeze(as_array1d(["base0", "base1"]))
    both_names = freeze(as_array1d(["both0", "both1", "both2"]))
    base.create_axis("base_axis", base_names)
    base.create_axis("both_axis", both_names)

    delta = MemoryStorage(name="delta")
    delta_names = freeze(as_array1d(["both0", "both1", "both2"]))
    delta.create_axis("both_axis", both_names)
    delta.create_axis("delta_axis", delta_names)

    chain = StorageChain([delta, base], name="chain")

    assert set(chain.axis_names()) == set(["base_axis", "both_axis", "delta_axis"])

    assert np.all(chain.axis_size("base_axis") == len(base_names))
    assert np.all(chain.axis_size("both_axis") == len(both_names))
    assert np.all(chain.axis_size("delta_axis") == len(delta_names))

    assert np.all(chain.axis_entries("base_axis") == base_names)
    assert np.all(chain.axis_entries("both_axis") == both_names)
    assert np.all(chain.axis_entries("delta_axis") == delta_names)

    assert not chain.has_axis("neither_axis")


def test_chain_array1d() -> None:
    base = MemoryStorage(name="base")
    axis_names = freeze(as_array1d(["entry0", "entry1"]))
    base.create_axis("axis", axis_names)

    base_values = freeze(as_array1d([0, 1]))
    base.set_array1d("axis;base_array1d", base_values)
    base_both_values = freeze(as_array1d([0, 1]))
    base.set_array1d("axis;both_array1d", base_both_values)

    delta = MemoryStorage(name="delta")
    delta.create_axis("axis", axis_names)

    delta_both_values = freeze(as_array1d([2, 3]))
    delta.set_array1d("axis;both_array1d", delta_both_values)

    delta_values = freeze(as_array1d([4, 5]))
    delta.set_array1d("axis;delta_array1d", delta_values)

    chain = StorageChain([delta.as_reader(), base, NO_STORAGE], name="chain")

    assert set(chain.array1d_names("axis")) == set(["axis;base_array1d", "axis;both_array1d", "axis;delta_array1d"])

    assert fast_all_close(chain.get_array1d("axis;base_array1d"), base_values)
    assert fast_all_close(chain.get_array1d("axis;both_array1d"), delta_both_values)
    assert fast_all_close(chain.get_array1d("axis;delta_array1d"), delta_values)

    assert not chain.has_array1d("axis;neither_array1d")


def test_chain_data2d() -> None:
    base = MemoryStorage(name="base")
    row_names = freeze(as_array1d(["row0", "row1"]))
    base.create_axis("row", row_names)
    column_names = freeze(as_array1d(["column0", "column1", "column2"]))
    base.create_axis("column", column_names)

    base_values = freeze(be_array_in_rows(as_array2d([[0, 1, 2], [3, 4, 5]])))
    base.set_grid("row,column;base_data2d", base_values)
    base_both_values = freeze(be_array_in_rows(as_array2d([[6, 7, 8], [9, 10, 11]])))
    base.set_grid("row,column;both_data2d", base_both_values)

    delta = MemoryStorage(name="delta")
    delta.create_axis("row", row_names)
    delta.create_axis("column", column_names)

    delta_both_values = freeze(be_grid_in_rows(as_array2d([[12, 13, 14], [15, 16, 17]])))
    delta.set_grid("row,column;both_data2d", delta_both_values)

    delta_values = freeze(be_grid_in_rows(as_array2d([[18, 19, 20], [21, 22, 23]])))
    delta.set_grid("row,column;delta_data2d", delta_values)

    chain = StorageChain([delta, delta, base], name="chain")

    assert set(chain.data2d_names(("row", "column"))) == set(
        ["row,column;base_data2d", "row,column;both_data2d", "row,column;delta_data2d"]
    )

    assert fast_all_close(chain.get_data2d("row,column;base_data2d"), base_values)
    assert fast_all_close(chain.get_data2d("row,column;both_data2d"), delta_both_values)
    assert fast_all_close(chain.get_data2d("row,column;delta_data2d"), delta_values)

    assert not chain.has_data2d("row,column;neither_data2d")
