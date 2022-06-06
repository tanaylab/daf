"""
Test ``daf.storage.anndata``.
"""

import numpy as np
import scipy.sparse as sp  # type: ignore
from anndata import AnnData  # type: ignore

from daf.storage.anndata import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.memory import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.views import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_raise

# pylint: disable=missing-function-docstring


def simple_anndata() -> AnnData:
    adata = AnnData(as_array2d([[0, 10, 90], [190, 10, 0]]), dtype="int64")
    adata.obs_names = ["cell0", "cell1"]
    adata.var_names = ["gene0", "gene1", "gene2"]
    return adata


def test_anndata_datum() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert not adata.has_datum("description")
    assert len(adata.datum_names()) == 0
    with expect_raise("missing datum: description in the storage: test"):
        adata.get_datum("description")

    adata.set_datum("description", "test AnnData storage")

    assert set(adata.datum_names()) == set(["description"])
    assert adata.has_datum("description")
    assert adata.get_datum("description") == "test AnnData storage"

    adata.set_datum("description", "retest AnnData storage", overwrite=True)
    assert adata.get_datum("description") == "retest AnnData storage"


def test_anndata_axis() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert adata.has_axis("obs")
    assert adata.axis_size("obs") == 2
    assert list(adata.axis_entries("obs")) == ["cell0", "cell1"]

    assert adata.has_axis("var")
    assert adata.axis_size("var") == 3
    assert list(adata.axis_entries("var")) == ["gene0", "gene1", "gene2"]

    assert adata.has_axis("var")

    assert not adata.has_axis("type")
    assert set(adata.axis_names()) == set(["obs", "var"])
    with expect_raise("missing axis: type in the storage: test"):
        adata.axis_size("type")
    with expect_raise("missing axis: type in the storage: test"):
        adata.axis_entries("type")

    type_names = freeze(as_array1d(["type0", "type1"]))
    adata.create_axis("type", type_names)

    assert adata.has_axis("type")
    assert adata.axis_size("type") == 2
    assert set(adata.axis_names()) == set(["obs", "var", "type"])
    assert len(adata.axis_entries("type")) == len(type_names)
    assert np.all(adata.axis_entries("type") == type_names)

    with expect_raise("refuse to recreate the axis: type in the storage: test"):
        adata.create_axis("type", freeze(as_array1d(["type1", "type0"])))
    assert np.all(adata.axis_entries("type") == type_names)


def test_anndata_array1d_of_obs() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert not adata.has_array1d("obs;type")
    assert len(adata.array1d_names("obs")) == 0
    with expect_raise("missing 1D data: obs;type in the storage: test"):
        adata.get_array1d("obs;type")

    cell_types = freeze(as_array1d(["T", "B"]))
    adata.set_array1d("obs;type", cell_types)

    assert adata.has_array1d("obs;type")
    assert set(adata.array1d_names("obs")) == set(["obs;type"])
    assert is_array1d(adata.get_array1d("obs;type"), dtype=STR_DTYPE)
    assert np.all(adata.get_array1d("obs;type") == cell_types)

    new_cell_types = freeze(as_array1d(["B", "T"]))
    adata.set_array1d("obs;type", new_cell_types, overwrite=True)
    assert np.all(adata.get_array1d("obs;type") == new_cell_types)


def test_anndata_array1d_of_var() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert not adata.has_array1d("var;significant")
    assert len(adata.array1d_names("var")) == 0
    with expect_raise("missing 1D data: var;significant in the storage: test"):
        adata.get_array1d("var;significant")

    significant_genes_mask = freeze(as_array1d([True, False, True]))
    adata.set_array1d("var;significant", significant_genes_mask)

    assert adata.has_array1d("var;significant")
    assert set(adata.array1d_names("var")) == set(["var;significant"])
    assert is_array1d(adata.get_array1d("var;significant"), dtype="bool")
    assert np.all(adata.get_array1d("var;significant") == significant_genes_mask)

    new_significant_genes_mask = freeze(as_array1d([False, True, True]))
    adata.set_array1d("var;significant", new_significant_genes_mask, overwrite=True)
    assert np.all(adata.get_array1d("var;significant") == new_significant_genes_mask)


def test_anndata_array1d_of_other() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    type_names = freeze(as_array1d(["type0", "type1"]))
    adata.create_axis("type", type_names)

    assert not adata.has_array1d("type;color")
    assert len(adata.array1d_names("type")) == 0
    with expect_raise("missing 1D data: type;color in the storage: test"):
        adata.get_array1d("type;color")

    type_colors = freeze(as_array1d(["red", "green"]))
    adata.set_array1d("type;color", type_colors)

    assert adata.has_array1d("type;color")
    assert set(adata.array1d_names("type")) == set(["type;color"])
    assert is_array1d(adata.get_array1d("type;color"), dtype=STR_DTYPE)
    assert np.all(adata.get_array1d("type;color") == type_colors)

    new_type_colors = freeze(as_array1d(["black", "white"]))
    adata.set_array1d("type;color", new_type_colors, overwrite=True)
    assert np.all(adata.get_array1d("type;color") == new_type_colors)


def test_anndata_x() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert adata.has_data2d("obs,var;X")
    assert not adata.has_data2d("var,obs;X")
    assert set(adata.data2d_names("obs,var")) == set(["obs,var;X"])
    assert len(adata.data2d_names("var,obs")) == 0

    assert np.all(adata.get_data2d("obs,var;X") == as_array2d([[0, 10, 90], [190, 10, 0]]))

    new_x = freeze(sp.csr_matrix([[90, 0, 10], [10, 0, 190]]))
    adata.set_grid("obs,var;X", new_x, overwrite=True)
    assert fast_all_close(adata.get_data2d("obs,var;X"), new_x)


def test_anndata_layer() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert not adata.has_data2d("obs,var;Y")

    layer_y = freeze(sp.csr_matrix([[90, 0, 10], [10, 0, 190]]))
    adata.set_grid("obs,var;Y", layer_y)

    assert adata.has_data2d("obs,var;Y")
    assert set(adata.data2d_names("obs,var")) == set(["obs,var;X", "obs,var;Y"])
    assert len(adata.data2d_names("var,obs")) == 0
    assert fast_all_close(adata.get_data2d("obs,var;Y"), layer_y)

    new_layer_y = freeze(as_array2d([[0, 190], [10, 10], [90, 0]]))
    adata.set_grid("var,obs;Y", be_array_in_rows(new_layer_y), overwrite=True)
    assert fast_all_close(adata.get_data2d("obs,var;Y"), new_layer_y.transpose())


def test_anndata_obs_other() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    ages = freeze(as_array1d(["young", "mid", "old"]))
    adata.create_axis("age", ages)

    assert len(adata.data2d_names("obs,age")) == 0
    assert len(adata.data2d_names("age,obs")) == 0
    assert not adata.has_data2d("obs,age;fractions")
    assert not adata.has_data2d("age,obs;fractions")

    age_metacell_fractions = freeze(as_array2d([[0.75, 0.1], [0.15, 0.2], [0.1, 0.6]]))
    adata.set_grid("age,obs;fraction", be_array_in_rows(age_metacell_fractions))

    assert set(adata.data2d_names("obs,age")) == set(["obs,age;fraction"])
    assert len(adata.data2d_names("age,obs")) == 0
    assert adata.has_data2d("obs,age;fraction")
    assert not adata.has_data2d("age,obs;fraction")
    assert fast_all_close(adata.get_data2d("obs,age;fraction"), age_metacell_fractions.transpose())


def test_anndata_var_other() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    type_names = freeze(as_array1d(["type0", "type1"]))
    adata.create_axis("type", type_names)

    assert len(adata.data2d_names("var,type")) == 0
    assert len(adata.data2d_names("type,var")) == 0
    assert not adata.has_data2d("var,type;essential")
    assert not adata.has_data2d("type,var;essential")

    essential_type_genes = freeze(as_array2d([[False, True, False], [True, False, False]]))
    adata.set_grid("type,var;essential", be_array_in_rows(essential_type_genes))

    assert set(adata.data2d_names("var,type")) == set(["var,type;essential"])
    assert len(adata.data2d_names("type,var")) == 0
    assert adata.has_data2d("var,type;essential")
    assert not adata.has_data2d("type,var;essential")
    assert fast_all_close(adata.get_data2d("var,type;essential"), essential_type_genes.transpose())


def test_anndata_obs_obs() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert len(adata.data2d_names("obs,obs")) == 0
    assert not adata.has_data2d("obs,obs;edges")

    edges = freeze(as_array2d([[0.0, 0.1], [0.2, 0.0]]))
    adata.set_grid("obs,obs;edges", be_array_in_rows(edges))

    assert set(adata.data2d_names("obs,obs")) == set(["obs,obs;edges"])
    assert adata.has_data2d("obs,obs;edges")
    assert fast_all_close(adata.get_data2d("obs,obs;edges"), edges)


def test_anndata_var_var() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert len(adata.data2d_names("var,var")) == 0
    assert not adata.has_data2d("var,var;edges")

    edges = freeze(as_array2d([[0.0, 0.1, 0.0], [0.2, 0.0, 0.1], [0.3, 0.0, 0.0]]))
    adata.set_grid("var,var;edges", be_array_in_rows(edges))

    assert set(adata.data2d_names("var,var")) == set(["var,var;edges"])
    assert adata.has_data2d("var,var;edges")
    assert fast_all_close(adata.get_data2d("var,var;edges"), edges)


def test_anndata_other_2d() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    type_names = freeze(as_array1d(["type0", "type1"]))
    adata.create_axis("type", type_names)

    ages = freeze(as_array1d(["young", "mid", "old"]))
    adata.create_axis("age", ages)

    assert len(adata.data2d_names("type,age")) == 0
    assert not adata.has_data2d("type,age;fraction")

    type_age_fractions = freeze(as_array2d([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]))
    adata.set_grid("type,age;fraction", be_array_in_rows(type_age_fractions))

    assert set(adata.data2d_names("type,age")) == set(["type,age;fraction"])
    assert adata.has_data2d("type,age;fraction")
    assert fast_all_close(adata.get_data2d("type,age;fraction"), type_age_fractions)


def test_anndata_as_daf() -> None:
    cdata = AnnData(as_array2d([[0, 10, 90], [190, 10, 0], [20, 160, 20]]), dtype="int64")
    cdata.obs_names = ["cell0", "cell1", "cell2"]
    cdata.var_names = ["gene0", "gene1", "gene2"]
    cdata.obs["metacell"] = as_array1d([0, 1, 1])

    cells = anndata_as_daf(AnnDataView(cdata, obs="cell", var="gene", X="UMIs"), name="cells")

    assert np.all(cells.get_array1d("cell;metacell") == as_array1d([0, 1, 1]))
    assert np.all(cells.get_data2d("cell,gene;UMIs") == as_array2d([[0, 10, 90], [190, 10, 0], [20, 160, 20]]))

    mdata = AnnData(as_array2d([[0, 10, 90], [210, 170, 20]]), dtype="int64")
    mdata.obs_names = ["metacell0", "metacell1"]
    mdata.var_names = ["gene0", "gene1", "gene2"]
    mdata.var["significant"] = as_array1d([True, False, True])
    mdata.obs["cells"] = as_array1d([1, 2])

    both = anndata_as_daf(
        [
            AnnDataView(cdata, obs="cell", var="gene", X="UMIs", name="cells"),
            AnnDataView(mdata, obs="metacell", var="gene", X="UMIs", name="metacells"),
        ],
        name="both",
    )

    assert np.all(both.get_array1d("cell;metacell") == as_array1d([0, 1, 1]))
    assert np.all(both.get_array1d("metacell;cells") == as_array1d([1, 2]))
    assert np.all(both.get_array1d("gene;significant") == as_array1d([True, False, True]))

    assert np.all(both.get_data2d("cell,gene;UMIs") == as_array2d([[0, 10, 90], [190, 10, 0], [20, 160, 20]]))
    assert np.all(both.get_data2d("metacell,gene;UMIs") == as_array2d([[0, 10, 90], [210, 170, 20]]))


def test_daf_as_anndata() -> None:
    both = MemoryStorage(name="both")

    both.create_axis("cell", freeze(as_array1d(["cell0", "cell1", "cell2"])))
    both.create_axis("gene", freeze(as_array1d(["gene0", "gene1", "gene2"])))
    both.create_axis("metacell", freeze(as_array1d(["metacell0", "metacell1"])))

    both.set_array1d("cell;metacell", freeze(as_array1d([0, 1, 1])))
    both.set_array1d("metacell;cells", freeze(as_array1d([1, 2])))
    both.set_array1d("gene;significant", freeze(as_array1d([True, False, True])))
    both.set_grid("cell,gene;UMIs", be_array_in_rows(freeze(as_array2d([[0, 10, 90], [190, 10, 0], [20, 160, 20]]))))
    both.set_grid("metacell,gene;UMIs", be_array_in_rows(freeze(as_array2d([[0, 10, 90], [210, 170, 20]]))))

    cells = StorageView(both, axes=dict(cell="obs", gene="var", metacell=None), data={"cell,gene;UMIs": "X"})
    cdata = daf_as_anndata(cells)

    assert np.all(cdata.X == as_array2d([[0, 10, 90], [190, 10, 0], [20, 160, 20]]))
    assert np.all(cdata.obs["metacell"] == as_array1d([0, 1, 1]))
    assert np.all(cdata.var["significant"] == as_array1d([True, False, True]))
