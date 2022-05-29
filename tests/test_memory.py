"""
Test ``daf.storage.memory``.
"""

import numpy as np
import scipy.sparse as sp  # type: ignore

from daf.storage.memory import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_raise

# pylint: disable=missing-function-docstring


def test_memory_datum() -> None:
    memory = MemoryStorage(name="test")

    assert not memory.has_datum("description")
    assert len(memory.datum_names()) == 0
    with expect_raise("missing datum: description in the storage: test"):
        memory.get_datum("description")

    memory.set_datum("description", "test memory storage")

    assert set(memory.datum_names()) == set(["description"])
    assert memory.has_datum("description")
    assert memory.get_datum("description") == "test memory storage"

    with expect_raise("refuse to overwrite the datum: description in the storage: test"):
        memory.set_datum("description", "retest memory storage")
    assert memory.get_datum("description") == "test memory storage"
    memory.set_datum("description", "retest memory storage", overwrite=True)
    assert memory.get_datum("description") == "retest memory storage"


def test_memory_axis() -> None:
    memory = MemoryStorage(name="test")

    assert not memory.has_axis("cells")
    assert len(memory.axis_names()) == 0
    with expect_raise("missing axis: cells in the storage: test"):
        memory.axis_size("cells")
    with expect_raise("missing axis: cells in the storage: test"):
        memory.axis_entries("cells")

    cell_names = freeze(as_array1d(["cell0", "cell1"]))
    memory.create_axis("cells", cell_names)

    assert memory.has_axis("cells")
    assert set(memory.axis_names()) == set(["cells"])
    assert is_frozen(memory.axis_entries("cells"))
    assert len(memory.axis_entries("cells")) == len(cell_names)
    assert np.all(memory.axis_entries("cells") == cell_names)

    with expect_raise("refuse to recreate the axis: cells in the storage: test"):
        memory.create_axis("cells", cell_names)
    assert np.all(memory.axis_entries("cells") == cell_names)


def test_memory_array1d() -> None:
    memory = MemoryStorage(name="test")

    assert not memory.has_array1d("cells:type")
    with expect_raise("missing axis: cells in the storage: test"):
        memory.array1d_names("cells")
    with expect_raise("missing axis: cells in the storage: test"):
        memory.get_array1d("cells:type")

    cell_names = freeze(as_array1d(["cell0", "cell1"]))
    memory.create_axis("cells", cell_names)

    assert not memory.has_array1d("cells:type")
    assert len(memory.array1d_names("cells")) == 0
    with expect_raise("missing 1D data: cells:type in the storage: test"):
        memory.get_array1d("cells:type")

    cell_types = freeze(as_array1d(["T", "B"]))
    memory.set_array1d("cells:type", cell_types)

    assert memory.has_array1d("cells:type")
    assert set(memory.array1d_names("cells")) == set(["cells:type"])
    assert is_array1d(memory.get_array1d("cells:type"))
    assert is_frozen(memory.get_array1d("cells:type"))
    assert np.all(memory.get_array1d("cells:type") == cell_types)


def test_memory_data2d() -> None:
    memory = MemoryStorage(name="test")

    assert not memory.has_data2d("cells,genes:UMIs")
    with expect_raise("missing rows axis: cells in the storage: test"):
        memory.data2d_names("cells,genes")
    with expect_raise("missing rows axis: cells in the storage: test"):
        memory.get_data2d("cells,genes:UMIs")

    cell_names = freeze(as_array1d(["cell0", "cell1"]))
    memory.create_axis("cells", cell_names)

    assert not memory.has_data2d("cells,genes:UMIs")
    with expect_raise("missing columns axis: genes in the storage: test"):
        memory.data2d_names("cells,genes")
    with expect_raise("missing columns axis: genes in the storage: test"):
        memory.get_data2d("cells,genes:UMIs")

    gene_names = freeze(as_array1d(["gene0", "gene1", "gene2"]))
    memory.create_axis("genes", gene_names)

    assert not memory.has_data2d("cells,genes:UMIs")
    assert len(memory.data2d_names("cells,genes")) == 0
    with expect_raise("missing 2D data: cells,genes:UMIs in the storage: test"):
        memory.get_data2d("cells,genes:UMIs")

    umis = freeze(be_array_in_rows(as_array2d([[0, 10, 90], [190, 10, 0]])))
    memory.set_grid("cells,genes:UMIs", umis)

    assert memory.has_data2d("cells,genes:UMIs")
    assert set(memory.data2d_names("cells,genes")) == set(["cells,genes:UMIs"])
    assert is_array_in_rows(memory.get_data2d("cells,genes:UMIs"))
    assert is_frozen(be_grid(memory.get_data2d("cells,genes:UMIs")))
    assert fast_all_close(memory.get_data2d("cells,genes:UMIs"), umis)

    new_umis = freeze(sp.csr_matrix([[90, 0, 10], [10, 0, 190]]))
    memory.set_grid("cells,genes:UMIs", new_umis, overwrite=True)
    assert fast_all_close(memory.get_data2d("cells,genes:UMIs"), new_umis)
