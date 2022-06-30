"""
Test ``daf.storage.anndata``.
"""

import warnings

import numpy as np
import scipy.sparse as sp  # type: ignore
from anndata import AnnData  # type: ignore

from daf.storage.anndata import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.memory import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.views import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_description
from . import expect_raise

warnings.simplefilter("error")


# pylint: disable=missing-function-docstring


def simple_anndata() -> AnnData:
    adata = AnnData(as_dense([[0, 10, 90], [190, 10, 0]]), dtype="int64")
    adata.obs_names = ["cell0", "cell1"]
    adata.var_names = ["gene0", "gene1", "gene2"]
    return adata


def test_anndata_description() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    expect_description(
        adata,
        expected="""
            test:
              class: daf.storage.anndata.AnnDataWriter
              axes:
                obs: 2 entries
                var: 3 entries
              data:
              - obs,var#X
            """,
    )

    expect_description(
        adata,
        detail=True,
        expected="""
            test:
              class: daf.storage.anndata.AnnDataWriter
              axes:
                obs: pandas.core.indexes.base.Index
                var: pandas.core.indexes.base.Index
              data:
                obs,var#X: row-major numpy.ndarray of 2x3 of int64
            """,
    )

    expect_description(
        adata,
        deep=True,
        expected="""
            test:
              class: daf.storage.anndata.AnnDataWriter
              axes:
                obs: 2 entries
                var: 3 entries
              data:
              - obs,var#X
            """,
    )


def test_anndata_item() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert not adata.has_item("description")
    assert len(adata.item_names()) == 0
    with expect_raise("missing item: description in the storage: test"):
        adata.get_item("description")

    adata.set_item("description", "test AnnData storage")

    assert set(adata.item_names()) == set(["description"])
    assert adata.has_item("description")
    assert adata.get_item("description") == "test AnnData storage"

    adata.set_item("description", "retest AnnData storage", overwrite=True)
    assert adata.get_item("description") == "retest AnnData storage"


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

    type_names = freeze(as_vector(["type0", "type1"]))
    adata.create_axis("type", type_names)

    assert adata.has_axis("type")
    assert adata.axis_size("type") == 2
    assert set(adata.axis_names()) == set(["obs", "var", "type"])
    assert len(adata.axis_entries("type")) == len(type_names)
    assert np.all(adata.axis_entries("type") == type_names)

    with expect_raise("refuse to recreate the axis: type in the storage: test"):
        adata.create_axis("type", freeze(as_vector(["type1", "type0"])))
    assert np.all(adata.axis_entries("type") == type_names)


def test_anndata_vector_of_obs() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert not adata.has_data1d("obs#type")
    assert len(adata.data1d_names("obs")) == 0
    with expect_raise("missing 1D data: obs#type in the storage: test"):
        adata.get_data1d("obs#type")

    cell_types = freeze(as_vector(["T", "B"]))
    adata.set_vector("obs#type", cell_types)

    assert adata.has_data1d("obs#type")
    assert set(adata.data1d_names("obs")) == set(["obs#type"])
    assert is_series(adata.get_data1d("obs#type"), dtype=STR_DTYPE)
    assert np.all(adata.get_data1d("obs#type") == cell_types)

    new_cell_types = freeze(as_vector(["B", "T"]))
    adata.set_vector("obs#type", new_cell_types, overwrite=True)
    assert np.all(adata.get_data1d("obs#type") == new_cell_types)


def test_anndata_vector_of_var() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert not adata.has_data1d("var#significant")
    assert len(adata.data1d_names("var")) == 0
    with expect_raise("missing 1D data: var#significant in the storage: test"):
        adata.get_data1d("var#significant")

    significant_genes_mask = freeze(as_vector([True, False, True]))
    adata.set_vector("var#significant", significant_genes_mask)

    assert adata.has_data1d("var#significant")
    assert set(adata.data1d_names("var")) == set(["var#significant"])
    assert is_series(adata.get_data1d("var#significant"), dtype="bool")
    assert np.all(adata.get_data1d("var#significant") == significant_genes_mask)

    new_significant_genes_mask = freeze(as_vector([False, True, True]))
    adata.set_vector("var#significant", new_significant_genes_mask, overwrite=True)
    assert np.all(adata.get_data1d("var#significant") == new_significant_genes_mask)


def test_anndata_vector_of_other() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    type_names = freeze(as_vector(["type0", "type1"]))
    adata.create_axis("type", type_names)

    assert not adata.has_data1d("type#color")
    assert len(adata.data1d_names("type")) == 0
    with expect_raise("missing 1D data: type#color in the storage: test"):
        adata.get_data1d("type#color")

    type_colors = freeze(as_vector(["red", "green"]))
    adata.set_vector("type#color", type_colors)

    assert adata.has_data1d("type#color")
    assert set(adata.data1d_names("type")) == set(["type#color"])
    assert is_vector(adata.get_data1d("type#color"), dtype=STR_DTYPE)
    assert np.all(adata.get_data1d("type#color") == type_colors)

    new_type_colors = freeze(as_vector(["black", "white"]))
    adata.set_vector("type#color", new_type_colors, overwrite=True)
    assert np.all(adata.get_data1d("type#color") == new_type_colors)


def test_anndata_x() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert adata.has_data2d("obs,var#X")
    assert not adata.has_data2d("var,obs#X")
    assert set(adata.data2d_names("obs,var")) == set(["obs,var#X"])
    assert len(adata.data2d_names("var,obs")) == 0

    assert np.all(adata.get_data2d("obs,var#X") == as_dense([[0, 10, 90], [190, 10, 0]]))

    new_x = freeze(sp.csr_matrix([[90, 0, 10], [10, 0, 190]]))
    adata.set_matrix("obs,var#X", new_x, overwrite=True)
    assert fast_all_close(adata.get_data2d("obs,var#X"), new_x)


def test_anndata_layer() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert not adata.has_data2d("obs,var#Y")

    layer_y = freeze(sp.csr_matrix([[90, 0, 10], [10, 0, 190]]))
    adata.set_matrix("obs,var#Y", layer_y)

    assert adata.has_data2d("obs,var#Y")
    assert set(adata.data2d_names("obs,var")) == set(["obs,var#X", "obs,var#Y"])
    assert len(adata.data2d_names("var,obs")) == 0
    assert fast_all_close(adata.get_data2d("obs,var#Y"), layer_y)

    new_layer_y = freeze(as_dense([[0, 190], [10, 10], [90, 0]]))
    adata.set_matrix("var,obs#Y", be_dense_in_rows(new_layer_y), overwrite=True)
    assert fast_all_close(adata.get_data2d("obs,var#Y"), new_layer_y.transpose())


def test_anndata_obs_other() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    ages = freeze(as_vector(["young", "mid", "old"]))
    adata.create_axis("age", ages)

    assert len(adata.data2d_names("obs,age")) == 0
    assert len(adata.data2d_names("age,obs")) == 0
    assert not adata.has_data2d("obs,age#fraction_in_obs")
    assert not adata.has_data2d("age,obs#fraction_in_obs")

    age_metacell_fractions = freeze(as_dense([[0.75, 0.1], [0.15, 0.2], [0.1, 0.6]]))
    adata.set_matrix("age,obs#fraction_in_obs", be_dense_in_rows(age_metacell_fractions))

    assert set(adata.data2d_names("obs,age")) == set(["obs,age#fraction_in_obs"])
    assert len(adata.data2d_names("age,obs")) == 0
    assert adata.has_data2d("obs,age#fraction_in_obs")
    assert not adata.has_data2d("age,obs#fraction_in_obs")
    assert fast_all_close(adata.get_data2d("obs,age#fraction_in_obs"), age_metacell_fractions.transpose())


def test_anndata_var_other() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    type_names = freeze(as_vector(["type0", "type1"]))
    adata.create_axis("type", type_names)

    assert len(adata.data2d_names("var,type")) == 0
    assert len(adata.data2d_names("type,var")) == 0
    assert not adata.has_data2d("var,type#essential")
    assert not adata.has_data2d("type,var#essential")

    essential_type_genes = freeze(as_dense([[False, True, False], [True, False, False]]))
    adata.set_matrix("type,var#essential", be_dense_in_rows(essential_type_genes))

    assert set(adata.data2d_names("var,type")) == set(["var,type#essential"])
    assert len(adata.data2d_names("type,var")) == 0
    assert adata.has_data2d("var,type#essential")
    assert not adata.has_data2d("type,var#essential")
    assert fast_all_close(adata.get_data2d("var,type#essential"), essential_type_genes.transpose())


def test_anndata_obs_obs() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert len(adata.data2d_names("obs,obs")) == 0
    assert not adata.has_data2d("obs,obs#edges")

    edges = freeze(as_dense([[0.0, 0.1], [0.2, 0.0]]))
    adata.set_matrix("obs,obs#edges", be_dense_in_rows(edges))

    assert set(adata.data2d_names("obs,obs")) == set(["obs,obs#edges"])
    assert adata.has_data2d("obs,obs#edges")
    assert fast_all_close(adata.get_data2d("obs,obs#edges"), edges)


def test_anndata_var_var() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    assert len(adata.data2d_names("var,var")) == 0
    assert not adata.has_data2d("var,var#edges")

    edges = freeze(as_dense([[0.0, 0.1, 0.0], [0.2, 0.0, 0.1], [0.3, 0.0, 0.0]]))
    adata.set_matrix("var,var#edges", be_dense_in_rows(edges))

    assert set(adata.data2d_names("var,var")) == set(["var,var#edges"])
    assert adata.has_data2d("var,var#edges")
    assert fast_all_close(adata.get_data2d("var,var#edges"), edges)


def test_anndata_other_2d() -> None:
    adata = AnnDataWriter(simple_anndata(), name="test")

    type_names = freeze(as_vector(["type0", "type1"]))
    adata.create_axis("type", type_names)

    ages = freeze(as_vector(["young", "mid", "old"]))
    adata.create_axis("age", ages)

    assert len(adata.data2d_names("type,age")) == 0
    assert not adata.has_data2d("type,age#fraction_in_type")

    type_age_fractions = freeze(as_dense([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]))
    adata.set_matrix("type,age#fraction_in_type", be_dense_in_rows(type_age_fractions))

    assert set(adata.data2d_names("type,age")) == set(["type,age#fraction_in_type"])
    assert adata.has_data2d("type,age#fraction_in_type")
    assert fast_all_close(adata.get_data2d("type,age#fraction_in_type"), type_age_fractions)


def test_anndata_as_storage() -> None:
    cdata = AnnData(as_dense([[0, 10, 90], [190, 10, 0], [20, 160, 20]]), dtype="int64")
    cdata.obs_names = ["cell0", "cell1", "cell2"]
    cdata.var_names = ["gene0", "gene1", "gene2"]
    cdata.obs["metacell"] = as_vector([0, 1, 1])

    cells = anndata_as_storage(cdata, obs="cell", var="gene", X="UMIs", name="cells")

    assert np.all(cells.get_data1d("cell#metacell") == as_vector([0, 1, 1]))
    assert np.all(cells.get_data2d("cell,gene#UMIs") == as_dense([[0, 10, 90], [190, 10, 0], [20, 160, 20]]))


def test_storage_as_anndata() -> None:
    both = MemoryStorage(name="both")

    both.create_axis("cell", freeze(as_vector(["cell0", "cell1", "cell2"])))
    both.create_axis("gene", freeze(as_vector(["gene0", "gene1", "gene2"])))
    both.create_axis("metacell", freeze(as_vector(["metacell0", "metacell1"])))

    both.set_vector("cell#metacell", freeze(as_vector([0, 1, 1])))
    both.set_vector("metacell#cells", freeze(as_vector([1, 2])))
    both.set_vector("gene#significant", freeze(as_vector([True, False, True])))
    both.set_matrix("cell,gene#UMIs", be_dense_in_rows(freeze(as_dense([[0, 10, 90], [190, 10, 0], [20, 160, 20]]))))
    both.set_matrix("metacell,gene#UMIs", be_dense_in_rows(freeze(as_dense([[0, 10, 90], [210, 170, 20]]))))

    cells = StorageView(both, axes=dict(cell="obs", gene="var", metacell=None), data={"cell,gene#UMIs": "X"})
    cdata = storage_as_anndata(cells)

    assert np.all(cdata.X == as_dense([[0, 10, 90], [190, 10, 0], [20, 160, 20]]))
    assert np.all(cdata.obs["metacell"] == as_vector([0, 1, 1]))
    assert np.all(cdata.var["significant"] == as_vector([True, False, True]))
