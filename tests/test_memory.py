"""
Test ``daf.storage.memory``.
"""

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from daf.storage.chain import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.interface import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.memory import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-function-docstring


def check_memory(memory: StorageReader, name: str) -> None:
    assert memory.name == name

    assert set(memory.datum_names()) == set(["description"])
    assert set(memory.axis_names()) == set(["cells", "genes"])
    assert set(memory.vector_names("cells")) == set(["cells:type"])
    assert set(memory.matrix_names(("cells", "genes"))) == set(["cells,genes:UMIs", "cells,genes:fraction_of_cell"])
    assert set(memory.matrix_names(("genes", "cells"))) == set(["genes,cells:fraction_of_gene"])

    assert memory.get_datum("description") == "test memory storage"

    assert memory.has_datum("description")
    assert not memory.has_datum("name")

    assert memory.axis_size("cells") == 2
    assert memory.axis_size("genes") == 3

    assert np.all(memory.axis_entries("cells") == np.array(["cell0", "cell1"]))
    assert np.all(memory.axis_entries("genes") == np.array(["gene0", "gene1", "gene2"]))

    assert memory.has_vector("cells:type")
    assert not memory.has_vector("genes:type")

    assert np.all(memory.get_array1d("cells:type") == np.array(["T", "B"]))

    assert np.all(memory.get_series("cells:type").values == np.array(["T", "B"]))
    assert np.all(memory.get_series("cells:type").index == np.array(["cell0", "cell1"]))

    assert memory.has_matrix("cells,genes:UMIs")
    assert not memory.has_matrix("genes,cells:UMIs")

    assert np.all(memory.get_grid("cells,genes:UMIs") == np.array([[0, 10, 90], [190, 10, 0]]))
    assert np.all(memory.get_table("cells,genes:UMIs").values == np.array([[0, 10, 90], [190, 10, 0]]))
    assert np.all(memory.get_table("cells,genes:UMIs").index == np.array(["cell0", "cell1"]))
    assert np.all(memory.get_table("cells,genes:UMIs").columns == np.array(["gene0", "gene1", "gene2"]))

    grid = memory.get_grid("cells,genes:fraction_of_cell")
    assert is_sparse(grid)
    assert np.all(grid.toarray() == np.array([[0, 0.1, 0.9], [0.95, 0.05, 0]]))
    assert np.all(memory.get_table("cells,genes:fraction_of_cell").values == np.array([[0, 0.1, 0.9], [0.95, 0.05, 0]]))
    assert np.all(memory.get_table("cells,genes:fraction_of_cell").index == np.array(["cell0", "cell1"]))
    assert np.all(memory.get_table("cells,genes:fraction_of_cell").columns == np.array(["gene0", "gene1", "gene2"]))

    assert np.all(memory.get_grid("genes,cells:fraction_of_gene") == np.array([[0, 1], [0.5, 0.5], [1, 0]]))
    assert np.all(memory.get_table("genes,cells:fraction_of_gene").values == np.array([[0, 1], [0.5, 0.5], [1, 0]]))
    assert np.all(memory.get_table("genes,cells:fraction_of_gene").index == np.array(["gene0", "gene1", "gene2"]))
    assert np.all(memory.get_table("genes,cells:fraction_of_gene").columns == np.array(["cell0", "cell1"]))


def test_memory() -> None:
    memory = MemoryStorage(name="test")

    memory.set_datum("description", "test memory storage")

    memory.create_axis("cells", as_array1d(["cell0", "cell1"]))
    memory.create_axis("genes", as_array1d(["gene0", "gene1", "gene2"]))

    memory.set_vector("cells:type", pd.Series(["T", "B"], index=["cell0", "cell1"]))

    with memory.create_array2d("cells,genes:UMIs", dtype="uint8") as umis:
        umis[:, :] = np.array([[0, 10, 90], [190, 10, 0]])

    memory.set_matrix("cells,genes:fraction_of_cell", sp.csr_matrix([[0, 0.1, 0.9], [0.95, 0.05, 0]]))

    memory.set_matrix(
        "genes,cells:fraction_of_gene",
        pd.DataFrame([[0, 0.5, 1], [1, 0.5, 0]], columns=["gene0", "gene1", "gene2"], index=["cell0", "cell1"]).T,
    )

    check_memory(memory, "test")
    reader = memory.as_reader()
    assert id(reader.as_reader()) == id(reader)
    check_memory(reader, "test")

    copy = MemoryStorage(name="copy")
    copy.update(reader)
    check_memory(copy, "copy")
    copy.update(reader, overwrite=True)
    check_memory(StorageChain([memory, copy.as_reader()], name="chain"), "chain")
