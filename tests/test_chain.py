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
    base_names = as_array1d(["base0", "base1"])
    both_names = as_array1d(["both0", "both1", "both2"])
    base.create_axis("base_axis", base_names)
    base.create_axis("both_axis", both_names)

    delta = MemoryStorage(name="delta")
    delta_names = as_array1d(["both0", "both1", "both2"])
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


def test_chain_vector() -> None:
    base = MemoryStorage(name="base")
    axis_names = as_array1d(["entry0", "entry1"])
    base.create_axis("axis", axis_names)

    base_values = as_array1d([0, 1])
    base.set_vector("axis:base_vector", base_values)
    base_both_values = as_array1d([0, 1])
    base.set_vector("axis:both_vector", base_both_values)

    delta = MemoryStorage(name="delta")
    delta.create_axis("axis", axis_names)

    delta_both_values = as_array1d([2, 3])
    delta.set_vector("axis:both_vector", delta_both_values)

    delta_values = as_array1d([4, 5])
    delta.set_vector("axis:delta_vector", delta_values)

    chain = StorageChain([delta.as_reader(), base, NO_STORAGE], name="chain")

    assert set(chain.vector_names("axis")) == set(["axis:base_vector", "axis:both_vector", "axis:delta_vector"])

    assert fast_all_close(chain.get_array1d("axis:base_vector"), base_values)
    assert fast_all_close(chain.get_array1d("axis:both_vector"), delta_both_values)
    assert fast_all_close(chain.get_array1d("axis:delta_vector"), delta_values)

    assert fast_all_close(chain.get_series("axis:base_vector"), base_values)
    assert fast_all_close(chain.get_series("axis:both_vector"), delta_both_values)
    assert fast_all_close(chain.get_series("axis:delta_vector"), delta_values)

    assert not chain.has_vector("axis:neither_vector")


def test_chain_matrix() -> None:
    base = MemoryStorage(name="base")
    row_names = as_array1d(["row0", "row1"])
    base.create_axis("rows", row_names)
    column_names = as_array1d(["column0", "column1", "column2"])
    base.create_axis("columns", column_names)

    base_values = as_array2d([[0, 1, 2], [3, 4, 5]])
    base.set_matrix("rows,columns:base_matrix", base_values)
    base_both_values = as_array2d([[6, 7, 8], [9, 10, 11]])
    base.set_matrix("rows,columns:both_matrix", base_both_values)

    delta = MemoryStorage(name="delta")
    delta.create_axis("rows", row_names)
    delta.create_axis("columns", column_names)

    delta_both_values = as_array2d([[12, 13, 14], [15, 16, 17]])
    delta.set_matrix("rows,columns:both_matrix", delta_both_values)

    delta_values = as_array2d([[18, 19, 20], [21, 22, 23]])
    delta.set_matrix("rows,columns:delta_matrix", delta_values)

    chain = StorageChain([delta, delta, base], name="chain")

    assert set(chain.matrix_names(("rows", "columns"))) == set(
        ["rows,columns:base_matrix", "rows,columns:both_matrix", "rows,columns:delta_matrix"]
    )

    assert fast_all_close(chain.get_grid("rows,columns:base_matrix"), base_values)
    assert fast_all_close(chain.get_grid("rows,columns:both_matrix"), delta_both_values)
    assert fast_all_close(chain.get_grid("rows,columns:delta_matrix"), delta_values)

    assert fast_all_close(chain.get_table("rows,columns:base_matrix"), base_values)
    assert fast_all_close(chain.get_table("rows,columns:both_matrix"), delta_both_values)
    assert fast_all_close(chain.get_table("rows,columns:delta_matrix"), delta_values)

    assert not chain.has_matrix("rows,columns:neither_matrix")
