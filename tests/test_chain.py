"""
Test ``daf.storage.chain``.
"""

import numpy as np

from daf.storage.chains import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.memory import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.none import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_description

# pylint: disable=missing-function-docstring


def test_chain_item() -> None:
    base = MemoryStorage(name="base")
    base.set_item("base_item", "base value")
    base.set_item("both_item", "both base value")

    delta = MemoryStorage(name="delta")
    delta.set_item("delta_item", "delta value")
    delta.set_item("both_item", "both delta value")

    chain = StorageChain([delta, NO_STORAGE, base.as_reader()], name="chain")

    assert set(chain.item_names()) == set(["base_item", "both_item", "delta_item"])

    assert chain.get_item("base_item") == "base value"
    assert chain.get_item("both_item") == "both delta value"
    assert chain.get_item("delta_item") == "delta value"

    assert not chain.has_item("neither_item")

    expect_description(
        chain,
        deep=True,
        expected="""
        chain:
          class: daf.storage.chains.StorageChain
          chain:
          - delta
          - base
          axes: {}
          data:
          - base_item
          - both_item
          - delta_item
        delta:
          class: daf.storage.memory.MemoryStorage
          axes: {}
          data:
          - both_item
          - delta_item
        base:
          class: daf.storage.memory.MemoryStorage
          axes: {}
          data:
          - base_item
          - both_item
    """,
    )


def test_chain_axis() -> None:
    base = MemoryStorage(name="base")
    base_names = freeze(as_vector(["base0", "base1"]))
    both_names = freeze(as_vector(["both0", "both1", "both2"]))
    base.create_axis("base_axis", base_names)
    base.create_axis("both_axis", both_names)

    delta = MemoryStorage(name="delta")
    delta_names = freeze(as_vector(["both0", "both1", "both2"]))
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

    expect_description(
        chain,
        deep=True,
        expected="""
        chain:
          class: daf.storage.chains.StorageChain
          chain:
          - delta
          - base
          axes:
            base_axis: 2 entries
            both_axis: 3 entries
            delta_axis: 3 entries
          data: []
        delta:
          class: daf.storage.memory.MemoryStorage
          axes:
            both_axis: 3 entries
            delta_axis: 3 entries
          data: []
        base:
          class: daf.storage.memory.MemoryStorage
          axes:
            base_axis: 2 entries
            both_axis: 3 entries
          data: []
        """,
    )


def test_chain_vector() -> None:
    base = MemoryStorage(name="base")
    axis_names = freeze(as_vector(["entry0", "entry1"]))
    base.create_axis("axis", axis_names)

    base_values = freeze(as_vector([0, 1]))
    base.set_vector("axis;base_vector", base_values)
    base_both_values = freeze(as_vector([0, 1]))
    base.set_vector("axis;both_vector", base_both_values)

    delta = MemoryStorage(name="delta")
    delta.create_axis("axis", axis_names)

    delta_both_values = freeze(as_vector([2, 3]))
    delta.set_vector("axis;both_vector", delta_both_values)

    delta_values = freeze(as_vector([4, 5]))
    delta.set_vector("axis;delta_vector", delta_values)

    chain = StorageChain([delta.as_reader(), base, NO_STORAGE], name="chain")

    assert set(chain.data1d_names("axis")) == set(["axis;base_vector", "axis;both_vector", "axis;delta_vector"])

    assert fast_all_close(chain.get_data1d("axis;base_vector"), base_values)
    assert fast_all_close(chain.get_data1d("axis;both_vector"), delta_both_values)
    assert fast_all_close(chain.get_data1d("axis;delta_vector"), delta_values)

    assert not chain.has_data1d("axis;neither_vector")

    expect_description(
        chain,
        deep=True,
        expected="""
        chain:
          class: daf.storage.chains.StorageChain
          chain:
          - delta
          - base
          axes:
            axis: 2 entries
          data:
          - axis;base_vector
          - axis;both_vector
          - axis;delta_vector
        delta:
          class: daf.storage.memory.MemoryStorage
          axes:
            axis: 2 entries
          data:
          - axis;both_vector
          - axis;delta_vector
        base:
          class: daf.storage.memory.MemoryStorage
          axes:
            axis: 2 entries
          data:
          - axis;base_vector
          - axis;both_vector
    """,
    )


def test_chain_data2d() -> None:
    base = MemoryStorage(name="base")
    row_names = freeze(as_vector(["row0", "row1"]))
    base.create_axis("row", row_names)
    column_names = freeze(as_vector(["column0", "column1", "column2"]))
    base.create_axis("column", column_names)

    base_values = freeze(be_dense_in_rows(as_dense([[0, 1, 2], [3, 4, 5]])))
    base.set_matrix("row,column;base_data2d", base_values)
    base_both_values = freeze(be_dense_in_rows(as_dense([[6, 7, 8], [9, 10, 11]])))
    base.set_matrix("row,column;both_data2d", base_both_values)

    delta = MemoryStorage(name="delta")
    delta.create_axis("row", row_names)
    delta.create_axis("column", column_names)

    delta_both_values = freeze(be_matrix_in_rows(as_dense([[12, 13, 14], [15, 16, 17]])))
    delta.set_matrix("row,column;both_data2d", delta_both_values)

    delta_values = freeze(be_matrix_in_rows(as_dense([[18, 19, 20], [21, 22, 23]])))
    delta.set_matrix("row,column;delta_data2d", delta_values)

    chain = StorageChain([delta, delta, base], name="chain")

    assert set(chain.data2d_names(("row", "column"))) == set(
        ["row,column;base_data2d", "row,column;both_data2d", "row,column;delta_data2d"]
    )

    assert fast_all_close(chain.get_data2d("row,column;base_data2d"), base_values)
    assert fast_all_close(chain.get_data2d("row,column;both_data2d"), delta_both_values)
    assert fast_all_close(chain.get_data2d("row,column;delta_data2d"), delta_values)

    assert not chain.has_data2d("row,column;neither_data2d")

    expect_description(
        chain,
        deep=True,
        expected="""
        chain:
          class: daf.storage.chains.StorageChain
          chain:
          - delta
          - base
          axes:
            column: 3 entries
            row: 2 entries
          data:
          - row,column;base_data2d
          - row,column;both_data2d
          - row,column;delta_data2d
        delta:
          class: daf.storage.memory.MemoryStorage
          axes:
            column: 3 entries
            row: 2 entries
          data:
          - row,column;both_data2d
          - row,column;delta_data2d
        base:
          class: daf.storage.memory.MemoryStorage
          axes:
            column: 3 entries
            row: 2 entries
          data:
          - row,column;base_data2d
          - row,column;both_data2d
    """,
    )
