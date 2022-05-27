"""
Test ``daf.storage.memory``.
"""

import numpy as np
import pandas as pd  # type: ignore
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

    cell_names = as_array1d(["cell0", "cell1"])
    memory.create_axis("cells", cell_names)

    assert memory.has_axis("cells")
    assert set(memory.axis_names()) == set(["cells"])
    assert is_frozen(memory.axis_entries("cells"))
    assert len(memory.axis_entries("cells")) == len(cell_names)
    assert np.all(memory.axis_entries("cells") == cell_names)

    with expect_raise("refuse to recreate the axis: cells in the storage: test"):
        memory.create_axis("cells", cell_names)
    assert np.all(memory.axis_entries("cells") == cell_names)


def test_memory_vector() -> None:
    memory = MemoryStorage(name="test")

    assert not memory.has_vector("cells:type")
    with expect_raise("missing axis: cells in the storage: test"):
        memory.vector_names("cells")
    with expect_raise("missing axis: cells in the storage: test"):
        memory.get_array1d("cells:type")
    with expect_raise("missing axis: cells in the storage: test"):
        memory.get_series("cells:type")

    cell_names = as_array1d(["cell0", "cell1"])
    memory.create_axis("cells", cell_names)

    assert not memory.has_vector("cells:type")
    assert len(memory.vector_names("cells")) == 0
    with expect_raise("missing vector: cells:type in the storage: test"):
        memory.get_array1d("cells:type")
    with expect_raise("missing vector: cells:type in the storage: test"):
        memory.get_series("cells:type")

    cell_types = as_array1d(["T", "B"])
    memory.set_vector("cells:type", cell_types)

    assert memory.has_vector("cells:type")
    assert set(memory.vector_names("cells")) == set(["cells:type"])
    assert is_array1d(memory.get_array1d("cells:type"))
    assert is_frozen(memory.get_array1d("cells:type"))
    assert np.all(memory.get_array1d("cells:type") == cell_types)
    assert is_series(memory.get_series("cells:type"))
    assert is_frozen(memory.get_series("cells:type"))
    assert np.all(memory.get_series("cells:type").index == cell_names)
    assert np.all(memory.get_series("cells:type").values == cell_types)

    new_cell_types = pd.Series(["B", "T"], index=cell_names)
    with expect_raise("refuse to overwrite the vector: cells:type in the storage: test"):
        memory.set_vector("cells:type", new_cell_types)
    assert np.all(memory.get_array1d("cells:type") == cell_types)
    memory.set_vector("cells:type", new_cell_types, overwrite=True)
    assert np.all(memory.get_array1d("cells:type") == new_cell_types)


def test_memory_matrix() -> None:
    memory = MemoryStorage(name="test")

    assert not memory.has_matrix("cells,genes:UMIs")
    with expect_raise("missing rows axis: cells in the storage: test"):
        memory.matrix_names(("cells", "genes"))
    with expect_raise("missing rows axis: cells in the storage: test"):
        memory.get_grid("cells,genes:UMIs")
    with expect_raise("missing rows axis: cells in the storage: test"):
        memory.get_table("cells,genes:UMIs")

    cell_names = as_array1d(["cell0", "cell1"])
    memory.create_axis("cells", cell_names)

    assert not memory.has_matrix("cells,genes:UMIs")
    with expect_raise("missing columns axis: genes in the storage: test"):
        memory.matrix_names(("cells", "genes"))
    with expect_raise("missing columns axis: genes in the storage: test"):
        memory.get_grid("cells,genes:UMIs")
    with expect_raise("missing columns axis: genes in the storage: test"):
        memory.get_table("cells,genes:UMIs")

    gene_names = as_array1d(["gene0", "gene1", "gene2"])
    memory.create_axis("genes", gene_names)

    assert not memory.has_matrix("cells,genes:UMIs")
    assert len(memory.matrix_names(("cells", "genes"))) == 0
    with expect_raise("missing matrix: cells,genes:UMIs in the storage: test"):
        memory.get_grid("cells,genes:UMIs")
    with expect_raise("missing matrix: cells,genes:UMIs in the storage: test"):
        memory.get_table("cells,genes:UMIs")

    umis = as_array2d([[0, 10, 90], [190, 10, 0]])
    memory.set_matrix("cells,genes:UMIs", umis)

    assert memory.has_matrix("cells,genes:UMIs")
    assert set(memory.matrix_names(("cells", "genes"))) == set(["cells,genes:UMIs"])
    assert is_array_in_rows(memory.get_grid("cells,genes:UMIs"))
    assert is_frozen(memory.get_grid("cells,genes:UMIs"))
    assert fast_all_close(memory.get_grid("cells,genes:UMIs"), umis)
    assert is_table_in_rows(memory.get_table("cells,genes:UMIs"))
    assert is_frozen(memory.get_table("cells,genes:UMIs"))
    assert np.all(memory.get_table("cells,genes:UMIs").index == cell_names)
    assert np.all(memory.get_table("cells,genes:UMIs").columns == gene_names)
    assert fast_all_close(memory.get_table("cells,genes:UMIs").values, umis)

    new_umis = pd.DataFrame(as_array2d([[90, 10, 0], [0, 10, 190]]), index=cell_names, columns=gene_names)

    with expect_raise("refuse to overwrite the matrix: cells,genes:UMIs in the storage: test"):
        memory.set_matrix("cells,genes:UMIs", new_umis)
    assert fast_all_close(memory.get_grid("cells,genes:UMIs"), umis)
    assert is_array_in_rows(memory.get_grid("cells,genes:UMIs"))
    memory.set_matrix("cells,genes:UMIs", new_umis, overwrite=True)
    assert fast_all_close(memory.get_grid("cells,genes:UMIs"), new_umis)

    newer_umis = sp.csr_matrix([[90, 0, 10], [10, 0, 190]])
    memory.set_matrix("cells,genes:UMIs", newer_umis, overwrite=True)
    assert fast_all_close(memory.get_grid("cells,genes:UMIs"), newer_umis)
