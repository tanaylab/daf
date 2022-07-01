"""
Test ``daf.access.readers`` and ``daf.access.writers``.
"""

from textwrap import dedent

import numpy as np
import scipy.sparse as sp  # type: ignore

from daf import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_description
from . import expect_raise

# pylint: disable=missing-function-docstring


def test_daf_item() -> None:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    assert not data.as_reader().has_item("description")
    assert len(data.item_names()) == 0
    with expect_raise("missing item: description in the data set: test.daf"):
        data.get_item("description")

    data.set_item("description", "test daf storage")

    assert data.item_names() == ["description"]
    assert data.has_item("description")
    assert data.get_item("description") == "test daf storage"

    with expect_raise("refuse to overwrite the item: description in the data set: test.daf"):
        data.set_item("description", "retest daf storage")
    assert data.get_item("description") == "test daf storage"
    data.set_item("description", "retest daf storage", overwrite=True)
    assert data.get_item("description") == "retest daf storage"


def test_daf_axis() -> None:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    assert not data.has_axis("cell")
    assert len(data.axis_names()) == 0
    with expect_raise("missing axis: cell in the data set: test.daf"):
        data.axis_size("cell")
    with expect_raise("missing axis: cell in the data set: test.daf"):
        data.axis_entries("cell")

    cell_names = as_vector(["cell0", "cell1"])
    data.create_axis("cell", cell_names)

    assert data.has_axis("cell")
    assert data.axis_size("cell") == 2
    assert data.axis_index("cell", "cell1") == 1
    assert data.axis_names() == ["cell"]
    assert is_frozen(be_vector(data.axis_entries("cell")))
    assert len(data.axis_entries("cell")) == len(cell_names)
    assert np.all(data.axis_entries("cell") == cell_names)

    with expect_raise("refuse to recreate the axis: cell in the data set: test.daf"):
        data.create_axis("cell", cell_names)
    assert np.all(data.axis_entries("cell") == cell_names)


def test_daf_data1d() -> None:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    assert not data.has_data1d("cell#type")
    with expect_raise("missing axis: cell in the data set: test.daf"):
        data.data1d_names("cell")
    with expect_raise("missing axis: cell in the data set: test.daf"):
        data.get_vector("cell#type")

    cell_names = as_vector(["cell0", "cell1"])
    data.create_axis("cell", cell_names)

    assert not data.has_data1d("cell#type")
    assert len(data.data1d_names("cell")) == 0
    with expect_raise("missing 1D data: cell#type in the data set: test.daf"):
        data.get_series("cell#type")

    cell_types = as_vector(["T", "B"])
    data.create_axis("type", ["B", "T"])
    data.set_data1d("cell#type", ["T", "B"])
    data.set_data1d("type#color", ["red", "green"])

    assert data.has_data1d("cell#type")
    assert data.data1d_names("cell") == ["cell#type"]
    assert data.data1d_names("cell", full=False) == ["type"]
    assert is_vector(data.get_vector("cell#type"))
    assert is_frozen(data.get_vector("cell#type"))
    assert list(data.get_vector("cell#type")) == ["T", "B"]
    assert data.get_item("cell=cell0,type") == "T"
    assert list(data.get_vector("type#")) == ["B", "T"]
    assert list(data.get_vector("cell#type#")) == ["T", "B"]
    assert list(data.get_vector("cell#type#color")) == ["green", "red"]

    assert is_series(data.get_series("cell#type"))
    assert is_frozen(data.get_series("cell#type"))
    assert np.all(as_vector(data.get_series("cell#type")) == cell_types)
    assert np.all(as_vector(data.get_series("cell#type").index) == cell_names)

    new_cell_types = as_vector(["B", "T"])

    with expect_raise("refuse to overwrite the 1D data: cell#type in the data set: test.daf"):
        data.set_data1d("cell#type", new_cell_types)
    assert np.all(data.get_vector("cell#type") == cell_types)
    data.set_data1d("cell#type", new_cell_types, overwrite=True)
    assert np.all(data.get_vector("cell#type") == new_cell_types)


def test_daf_data2d() -> None:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    assert not data.has_data2d("cell,gene#UMIs")
    with expect_raise("missing axis: cell in the data set: test.daf"):
        data.data2d_names("cell,gene")
    with expect_raise("missing axis: cell in the data set: test.daf"):
        data.get_matrix("cell,gene#UMIs")

    cell_names = as_vector(["cell0", "cell1"])
    data.create_axis("cell", cell_names)

    assert not data.has_data2d("cell,gene#UMIs")
    with expect_raise("missing axis: gene in the data set: test.daf"):
        data.data2d_names("cell,gene")
    with expect_raise("missing axis: gene in the data set: test.daf"):
        data.get_matrix("cell,gene#UMIs")

    gene_names = as_vector(["gene0", "gene1", "gene2"])
    data.create_axis("gene", gene_names)

    assert not data.has_data2d("cell,gene#UMIs")
    assert len(data.data2d_names("cell,gene")) == 0
    with expect_raise("missing 2D data: cell,gene#UMIs in the data set: test.daf"):
        data.get_matrix("cell,gene#UMIs")

    umis = be_dense_in_rows(as_dense([[0, 10, 90], [190, 10, 0]]))
    data.set_data2d("cell,gene#UMIs", [[0, 10, 90], [190, 10, 0]])

    assert data.has_data2d("cell,gene#UMIs")
    assert data.data2d_names("cell,gene") == ["cell,gene#UMIs"]
    assert data.data2d_names("cell,gene", full=False) == ["UMIs"]
    assert is_dense_in_rows(data.get_matrix("cell,gene#UMIs"))
    assert is_frozen(data.get_matrix("cell,gene#UMIs"))
    assert fast_all_close(data.get_matrix("cell,gene#UMIs"), umis)
    assert data.get_item("cell=cell0,gene=gene1,UMIs") == 10
    assert data.get_item("gene=gene1,cell=cell0,UMIs") == 10
    assert list(data.get_vector("gene#cell=cell0,UMIs")) == [0, 10, 90]

    assert data.has_data2d("gene,cell#UMIs")
    assert data.data2d_names("gene,cell") == ["gene,cell#UMIs"]
    assert data.data2d_names("gene,cell", full=False) == ["UMIs"]
    assert is_dense_in_rows(data.get_matrix("gene,cell#UMIs"))
    assert is_frozen(data.get_matrix("gene,cell#UMIs"))
    assert np.allclose(data.get_matrix("gene,cell#UMIs"), umis.transpose())

    new_umis = freeze(sp.csr_matrix([[90, 0, 10], [10, 0, 190]]))
    data.set_data2d("gene,cell#UMIs", new_umis.transpose(), overwrite=True)
    assert fast_all_close(data.get_matrix("cell,gene#UMIs"), new_umis)

    fractions = np.array([[0, 0.1, 0.9], [0.95, 0.05, 0]])
    with data.create_dense_in_rows("cell,gene#fraction_in_cell", dtype="float32") as fractions_buffer:
        fractions_buffer[:] = fractions[:]

    assert is_frame_in_rows(data.get_frame("cell,gene#fraction_in_cell"))
    assert is_frozen(data.get_frame("cell,gene#fraction_in_cell"))
    assert fast_all_close(as_matrix(data.get_frame("cell,gene#fraction_in_cell")), fractions)
    assert np.all(as_vector(data.get_frame("cell,gene#fraction_in_cell").index) == cell_names)
    assert np.all(as_vector(data.get_frame("cell,gene#fraction_in_cell").columns) == gene_names)


def test_daf_columns() -> None:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    cell_names = as_vector(["cell0", "cell1"])
    data.create_axis("cell", cell_names)

    cell_types = as_vector(["T", "B"])
    data.set_data1d("cell#type", ["T", "B"])

    cell_ages = as_vector([6.5, 7.5])
    data.set_data1d("cell#age", cell_ages)

    with expect_raise("missing axis: gene in the data set: test.daf"):
        data.get_columns("gene", ["type", "age"])

    with expect_raise("missing 1D data: cell#cluster in the data set: test.daf"):
        data.get_columns("cell", ["type", "cluster"])

    assert np.all(as_vector(data.get_columns("cell", ["type", "age"]).index) == cell_names)
    assert np.all(as_vector(data.get_columns("cell", ["type", "age"]).columns) == as_vector(["type", "age"]))
    assert np.all(as_vector(data.get_columns("cell", ["type", "age"])["type"]) == cell_types)
    assert np.all(as_vector(data.get_columns("cell", ["type", "age"])["age"]) == cell_ages)


def test_daf_view() -> None:
    memory = MemoryStorage(name="base")

    memory.set_item("zero", 0.0)
    memory.set_item("one", 1.0)

    memory.create_axis("cell", freeze(as_vector(["cell0", "cell1"])))
    memory.create_axis("gene", freeze(as_vector(["gene0", "gene1", "gene2"])))

    umis = freeze(be_dense_in_rows(as_dense([[0, 10, 90], [190, 10, 0]])))
    memory.set_matrix("cell,gene#UMIs", umis)

    base = DafWriter(memory, name=".daf")
    view = base.view(axes=dict(cell="profile"), data={"cell,gene#UMIs": "count"})
    assert fast_all_close(view.as_reader().get_matrix("profile,gene#count"), umis)


@computation(
    required_inputs={"row,column#value": "Arbitrary 2D data."},
    assured_outputs={
        "row#sum": """
            The sum of the values in each row.
            Assumes no ``NaN`` values are in the data.
            """
    },
)
def row_sums(data: DafWriter, *, overwrite: bool = False) -> None:  # pylint: disable=unused-argument
    """
    Contrived computation step. You are better of writing ``.get_vector("row#column,value|RowSums")``.

    __DAF__
    """
    expect_description(
        data,
        deep=True,
        detail=True,
        expected="""
            test.daf.adapter#<id>.tests.test_access.row_sums#<id>:
              class: daf.access.writers.DafWriter
              axes:
                column: frozen 1D numpy.ndarray of 3 of <U5
                row: frozen 1D numpy.ndarray of 2 of <U5
              data:
                row,column#value: frozen row-major numpy.ndarray of 2x3 of int64
              chain: test.daf.adapter#<id>.tests.test_access.row_sums#<id>.chain
              derived: test.daf.adapter#<id>.tests.test_access.row_sums#<id>.derived
              storage: test.daf.adapter#<id>.tests.test_access.row_sums#<id>.storage
              base: test.daf.adapter#<id>.tests.test_access.row_sums#<id>.base
            test.daf.adapter#<id>.tests.test_access.row_sums#<id>.chain:
              class: daf.storage.chains.StorageChain
              chain:
              - test.daf.adapter#<id>.tests.test_access.row_sums#<id>.derived
              - test.daf.adapter#<id>.tests.test_access.row_sums#<id>.storage
              - test.daf.adapter#<id>.tests.test_access.row_sums#<id>.base
              axes:
                column: frozen 1D numpy.ndarray of 3 of <U5
                row: frozen 1D numpy.ndarray of 2 of <U5
              data:
                row,column#value: frozen row-major numpy.ndarray of 2x3 of int64
            test.daf.adapter#<id>.tests.test_access.row_sums#<id>.derived:
              class: daf.storage.memory.MemoryStorage
              axes:
                column: frozen 1D numpy.ndarray of 3 of <U5
                row: frozen 1D numpy.ndarray of 2 of <U5
              data: {}
            test.daf.adapter#<id>.tests.test_access.row_sums#<id>.storage:
              class: daf.storage.memory.MemoryStorage
              axes:
                column: frozen 1D numpy.ndarray of 3 of <U5
                row: frozen 1D numpy.ndarray of 2 of <U5
              data: {}
            test.daf.adapter#<id>.tests.test_access.row_sums#<id>.base:
              class: daf.storage.views.StorageView
              cache: test.daf.adapter#<id>.tests.test_access.row_sums#<id>.base.cache
              base: test.daf.adapter#<id>.chain
              axes:
                column: all 3 entries of column in frozen 1D numpy.ndarray of 3 of <U5
                row: all 2 entries of row in frozen 1D numpy.ndarray of 2 of <U5
              data:
                row,column#value: from row,column#value in frozen row-major numpy.ndarray of 2x3 of int64
            """,
    )
    data.set_item("tmp", 0)
    data.set_data1d("row#sum", data.get_matrix("row,column#value").sum(axis=1))


def test_computation_doc() -> None:
    assert row_sums.__doc__ == dedent(
        """
        Contrived computation step. You are better of writing ``.get_vector("row#column,value|RowSums")``.

        **Required Inputs**

        ``row,column#value``
            Arbitrary 2D data.

        **Assured Outputs**

        ``row#sum``
            The sum of the values in each row.
            Assumes no ``NaN`` values are in the data.

        If ``overwrite``, will overwrite existing data.
        """
    )


def test_computation_call() -> None:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    cell_names = as_vector(["cell0", "cell1"])
    data.create_axis("cell", cell_names)

    gene_names = as_vector(["gene0", "gene1", "gene2"])
    data.create_axis("gene", gene_names)

    data.set_data2d("cell,gene#UMIs", [[0, 10, 90], [190, 10, 0]])

    expect_description(
        data,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell,gene#UMIs
            """,
    )

    with data.adapter(
        axes=dict(cell="row", gene="column"),
        data={"cell,gene#UMIs": "value"},
        back_data=["row#sum"],
    ) as work:
        expect_description(
            work,
            # pylint: disable=duplicate-code
            expected="""
                test.daf.adapter#<id>:
                  class: daf.access.writers.DafWriter
                  axes:
                    column: 3 entries
                    row: 2 entries
                  data:
                  - row,column#value
                """,
            # pylint: enable=duplicate-code
        )
        row_sums(work)

    expect_description(
        data,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell#sum
              - cell,gene#UMIs
            """,
    )

    assert not data.has_item("tmp")
    assert data.has_data1d("cell#sum")
    assert fast_all_close(data.get_vector("cell#sum"), as_vector([100, 200]))


@computation(
    required_inputs={"cell,gene#UMIs": "Cells RNA data."},
    assured_outputs={
        "quality": "How good is the clustering.",
        "gene#mean": "The mean fraction of each gene in the total of all cells",
        "cluster#": "A new axis of clusters, each containing cells.",
        "cell#cluster": "The cluster each cell belongs to.",
        "cluster,gene#mean": "The mean fraction of each gene out of the total of the cells of some cluster.",
    },
    optional_outputs={"details": "Useless details."},
)
def cluster_cells(data: DafWriter, *, overwrite: bool = False) -> None:  # pylint: disable=unused-argument
    """
    Fake computation step.

    __DAF__
    """
    data.set_item("quality", "fake")
    data.set_item("details", "useless")
    data.create_axis("cluster", ["cluster0", "cluster1"])
    data.set_data1d("cell#cluster", [0, 0, 1])
    data.set_data1d("gene#mean", [240 / 400, 70 / 400, 90 / 400])
    data.set_data2d("cluster,gene#mean", [[190 / 300, 20 / 300, 90 / 300], [50 / 100, 50 / 100, 0]])

    expect_description(
        data,
        expected="""
            test.daf.adapter#<id>.tests.test_access.cluster_cells#<id>:
              class: daf.access.writers.DafWriter
              axes:
                cell: 3 entries
                cluster: 2 entries
                gene: 3 entries
              data:
              - details
              - quality
              - cell#cluster
              - gene#mean
              - cell,gene#UMIs
              - cluster,gene#mean
            """,
    )


def test_computation_back() -> None:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    profile_names = as_vector(["profile0", "profile1", "profile2"])
    data.create_axis("profile", profile_names)

    gene_names = as_vector(["gene0", "gene1", "gene2"])
    data.create_axis("gene", gene_names)

    data.set_data2d("profile,gene#UMIs", [[0, 10, 90], [190, 10, 0], [50, 50, 0]])

    expect_description(
        data,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                gene: 3 entries
                profile: 3 entries
              data:
              - profile,gene#UMIs
            """,
    )

    with data.adapter(
        axes=dict(profile="cell"),
        back_axes={"cluster": BackAxis("type")},
        back_data={
            "quality": BackData("quality"),
            "gene#mean": BackData("fraction"),
            "cell#cluster": BackData("type"),
            "cluster,gene#mean": BackData("fraction"),
        },
    ) as work:
        cluster_cells(work)

    expect_description(
        data,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                gene: 3 entries
                profile: 3 entries
                type: 2 entries
              data:
              - quality
              - gene#fraction
              - profile#type
              - profile,gene#UMIs
              - type,gene#fraction
            """,
    )

    assert data.has_data1d("profile#type")
    assert fast_all_close(data.get_vector("profile#type"), as_vector([0, 0, 1]))


@computation(
    required_inputs={"row,column#value": "Arbitrary 2D data."},
    assured_outputs={"row,column#absolute": "Absolute values.", "row#sum_abs": "Sum or absolute values per row."},
)
def row_sum_absolute(data: DafWriter, *, overwrite: bool = False) -> None:  # pylint: disable=unused-argument
    """
    Fake computation on a slice.

    __DAF__
    """
    expect_description(
        data,
        expected="""
            test.daf.adapter#<id>.tests.test_access.row_sum_absolute#<id>:
              class: daf.access.writers.DafWriter
              axes:
                column: 2 entries
                row: 2 entries
              data:
              - row,column#value
            """,
    )
    values = data.get_matrix("row,column#value")

    # Yes, `np.abs` will work on sparse matrices, but will return a dense matrix. Isn't it great?
    if is_sparse(values):
        absolute = sp.csr_matrix((np.abs(values.data), values.indices, values.indptr))
    else:
        absolute = np.abs(values)

    data.set_data2d("row,column#absolute", absolute)
    data.set_data1d("row#sum_abs", absolute.sum(axis=1))


def test_computation_slice_dense() -> None:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    cell_names = as_vector(["cell0", "cell1"])
    data.create_axis("cell", cell_names)

    gene_names = as_vector(["gene0", "gene1", "gene2"])
    data.create_axis("gene", gene_names)

    data.set_data2d("cell,gene#UMIs", [[0, -10.0, 90], [190, -10.0, 0]])

    expect_description(
        data,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell,gene#UMIs
            """,
    )

    with data.adapter(
        axes=dict(cell="row", gene=AxisView(name="column", entries=[0, 1])),
        data={"cell,gene#UMIs": "value"},
        back_data={"row#sum_abs": BackData(default=None), "row,column#absolute": BackData(default=None)},
    ) as work:
        expect_description(
            work,
            expected="""
                test.daf.adapter#<id>:
                  class: daf.access.writers.DafWriter
                  axes:
                    column: 2 entries
                    row: 2 entries
                  data:
                  - row,column#value
                """,
        )
        row_sum_absolute(work)

    expect_description(
        data,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell#sum_abs
              - cell,gene#UMIs
              - cell,gene#absolute
            """,
    )

    assert fast_all_close(data.get_vector("cell#sum_abs"), as_vector([10, 200]))
    assert fast_all_close(
        data.get_matrix("cell,gene#absolute"),
        np.array([[0, 10, None], [190, 10, None]], dtype="float32"),
        equal_nan=True,
    )


def test_computation_slice_sparse() -> None:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    cell_names = as_vector(["cell0", "cell1"])
    data.create_axis("cell", cell_names)

    gene_names = as_vector(["gene0", "gene1", "gene2"])
    data.create_axis("gene", gene_names)

    data.set_data2d("cell,gene#UMIs", sp.csr_matrix([[0, -10.0, 90], [190, -10.0, 0]]))

    expect_description(
        data,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell,gene#UMIs
            """,
    )

    with data.adapter(
        axes=dict(cell="row", gene=AxisView(name="column", entries=[0, 1])),
        data={"cell,gene#UMIs": "value"},
        back_data={"row#sum_abs": BackData(default=None), "row,column#absolute": BackData(default=None)},
    ) as work:
        expect_description(
            work,
            expected="""
                test.daf.adapter#<id>:
                  class: daf.access.writers.DafWriter
                  axes:
                    column: 2 entries
                    row: 2 entries
                  data:
                  - row,column#value
                """,
        )
        row_sum_absolute(work)

    expect_description(
        data,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell#sum_abs
              - cell,gene#UMIs
              - cell,gene#absolute
            """,
    )

    assert fast_all_close(data.get_vector("cell#sum_abs"), as_vector([10, 200]))
    assert fast_all_close(
        data.get_matrix("cell,gene#absolute"),
        np.array([[0, 10, None], [190, 10, None]], dtype="float32"),
        equal_nan=True,
    )


def test_computation_slice_very_sparse() -> None:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    cell_names = as_vector(["cell0", "cell1"])
    data.create_axis("cell", cell_names)

    gene_names = as_vector(["gene0", "gene1", "gene2"])
    data.create_axis("gene", gene_names)

    data.set_data2d("cell,gene#UMIs", sp.csr_matrix([[0, -10.0, 90], [190, -10.0, 0]]))

    expect_description(
        data,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell,gene#UMIs
            """,
    )

    with data.adapter(
        axes=dict(cell="row", gene=AxisView(name="column", entries=[0, 1])),
        data={"cell,gene#UMIs": "value"},
        back_data={"row#sum_abs": BackData(default=None), "row,column#absolute": BackData(default=0)},
    ) as work:
        expect_description(
            work,
            expected="""
                test.daf.adapter#<id>:
                  class: daf.access.writers.DafWriter
                  axes:
                    column: 2 entries
                    row: 2 entries
                  data:
                  - row,column#value
                """,
        )
        row_sum_absolute(work)

    expect_description(
        data,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell#sum_abs
              - cell,gene#UMIs
              - cell,gene#absolute
            """,
    )

    assert fast_all_close(data.get_vector("cell#sum_abs"), as_vector([10, 200]))
    assert fast_all_close(
        data.get_matrix("cell,gene#absolute"), sp.csr_matrix([[0, 10, 0], [190, 10, 0]], dtype="float32")
    )


def test_pipeline() -> None:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    cell_names = as_vector(["cell0", "cell1"])
    data.create_axis("cell", cell_names)

    gene_names = as_vector(["gene0", "gene1", "gene2"])
    data.create_axis("gene", gene_names)

    data.set_data1d("cell#age", np.array([1, 2], dtype="uint8"))

    data.set_data2d("cell,gene#UMIs", [[85, 10, 5], [170, 20, 10]])
    data.set_data2d("cell,gene#ratios", sp.csr_matrix([[0.25, 0.5, 1], [4, 2, 0]]))
    data.set_data2d("cell,gene#folds", sp.csr_matrix([[-2.0, -1.0, 0], [2, 1, 0]]))

    expect_description(
        data,
        deep=True,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell#age
              - cell,gene#UMIs
              - cell,gene#folds
              - cell,gene#ratios
              chain: test.daf.chain
              derived: test.daf.derived
              storage: test
              base: test.as_reader
            test.daf.chain:
              class: daf.storage.chains.StorageChain
              chain:
              - test.daf.derived
              - test
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell#age
              - cell,gene#UMIs
              - cell,gene#folds
              - cell,gene#ratios
            test.daf.derived:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data: []
            test:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell#age
              - cell,gene#UMIs
              - cell,gene#folds
              - cell,gene#ratios
            """,
    )

    for _ in range(2):
        assert list(data.get_vector("cell#age|Abs")) == [1, 2]
        assert data.get_item("cell,age|Mean") == 1.5
        assert list(data.get_vector("cell#age|Clip,min=0.5,max=1.5")) == [1, 1.5]

        assert list(data.get_vector("cell#gene,UMIs|!Sum")) == [100, 200]
        assert data.get_item("gene,cell=cell0,UMIs|Sum") == 100
        assert fast_all_close(data.get_matrix("cell,gene#UMIs|!Abs"), np.array([[85, 10, 5], [170, 20, 10]]))
        assert list(data.get_vector("gene#cell=cell0,UMIs|Abs")) == [85, 10, 5]
        assert fast_all_close(
            data.get_matrix("cell,gene#folds|!Densify|Abs"),
            np.array([[2, 1, 0], [2, 1, 0]]),
        )
        assert fast_all_close(
            data.get_matrix("cell,gene#folds|Significant,low=1.25,high=1.75"),
            sp.csr_matrix([[-2, 0, 0], [2, 0, 0]]),
        )
        assert fast_all_close(
            data.get_matrix("cell,gene#folds|!Densify|Significant,low=1.25,high=1.75"),
            sp.csr_matrix([[-2, 0, 0], [2, 0, 0]]),
        )
        assert fast_all_close(
            data.get_matrix("cell,gene#folds|Significant,low=1.25,high=1.75|Densify"),
            np.array([[-2, 0, 0], [2, 0, 0]]),
        )

    expect_description(
        data,
        deep=True,
        expected="""
            test.daf:
              class: daf.access.writers.DafWriter
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell,age|Mean,dtype=float32
              - cell#age
              - cell#age|Clip,min=0.5,max=1.5,dtype=float32
              - cell,gene#UMIs
              - cell,gene#folds
              - cell,gene#folds|Densify,dtype=float64|Abs,dtype=float64
              - cell,gene#folds|Densify,dtype=float64|Significant,low=1.25,high=1.75,abs=True,dtype=float64
              - cell,gene#folds|Significant,low=1.25,high=1.75,abs=True,dtype=float64
              - cell,gene#folds|Significant,low=1.25,high=1.75,abs=True,dtype=float64|Densify,dtype=float64
              - cell,gene#ratios
              chain: test.daf.chain
              derived: test.daf.derived
              storage: test
              base: test.as_reader
            test.daf.chain:
              class: daf.storage.chains.StorageChain
              chain:
              - test.daf.derived
              - test
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell,age|Mean,dtype=float32
              - cell#age
              - cell#age|Clip,min=0.5,max=1.5,dtype=float32
              - cell,gene#UMIs
              - cell,gene#folds
              - cell,gene#folds|Densify,dtype=float64|Abs,dtype=float64
              - cell,gene#folds|Densify,dtype=float64|Significant,low=1.25,high=1.75,abs=True,dtype=float64
              - cell,gene#folds|Significant,low=1.25,high=1.75,abs=True,dtype=float64
              - cell,gene#folds|Significant,low=1.25,high=1.75,abs=True,dtype=float64|Densify,dtype=float64
              - cell,gene#ratios
            test.daf.derived:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell,age|Mean,dtype=float32
              - cell#age|Clip,min=0.5,max=1.5,dtype=float32
              - cell,gene#folds|Densify,dtype=float64|Abs,dtype=float64
              - cell,gene#folds|Densify,dtype=float64|Significant,low=1.25,high=1.75,abs=True,dtype=float64
              - cell,gene#folds|Significant,low=1.25,high=1.75,abs=True,dtype=float64
              - cell,gene#folds|Significant,low=1.25,high=1.75,abs=True,dtype=float64|Densify,dtype=float64
            test:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - cell#age
              - cell,gene#UMIs
              - cell,gene#folds
              - cell,gene#ratios
            """,
    )
