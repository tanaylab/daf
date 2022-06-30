"""
Test ``daf.storage.view``.
"""

from typing import List
from typing import Optional
from typing import Union

import numpy as np

from daf.storage.interface import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.memory import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.views import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_description

# pylint: disable=missing-function-docstring


def make_base() -> MemoryStorage:
    memory = MemoryStorage(name="base")

    memory.set_item("zero", 0.0)
    memory.set_item("one", 1.0)

    cell_names = freeze(as_vector(["cell0", "cell1"]))
    memory.create_axis("cell", cell_names)

    gene_names = freeze(as_vector(["gene0", "gene1", "gene2"]))
    memory.create_axis("gene", gene_names)

    cell_types = freeze(as_vector(["T", "B"]))
    memory.set_vector("cell#type", cell_types)

    umis = freeze(be_dense_in_rows(as_dense([[0, 10, 90], [190, 10, 0]])))
    memory.set_matrix("cell,gene#UMIs", umis)

    return memory


def check_item_names(view: StorageReader, expected: Union[StorageReader, List[str]]) -> None:
    if isinstance(expected, StorageReader):
        assert set(view.item_names()) == set(expected.item_names())
    else:
        assert set(view.item_names()) == set(expected)


def check_axis_names(view: StorageReader, expected: Union[StorageReader, List[str]]) -> None:
    if isinstance(expected, StorageReader):
        assert set(view.axis_names()) == set(expected.axis_names())
    else:
        assert set(view.axis_names()) == set(expected)


def check_axis_entries(
    view: StorageReader, axis: str, expected: Union[StorageReader, str, Vector], base: Optional[StorageReader] = None
) -> None:
    if isinstance(expected, StorageReader):
        assert base is None
        assert np.all(view.axis_entries(axis) == expected.axis_entries(axis))
    elif isinstance(expected, str):
        assert base is not None
        assert np.all(view.axis_entries(axis) == base.axis_entries(expected))
    else:
        assert base is None
        assert np.all(view.axis_entries(axis) == expected)


def check_axis_size(
    view: StorageReader, axis: str, expected: Union[StorageReader, str, int], base: Optional[StorageReader] = None
) -> None:
    if isinstance(expected, StorageReader):
        assert base is None
        assert view.axis_size(axis) == expected.axis_size(axis)
    elif isinstance(expected, str):
        assert base is not None
        assert view.axis_size(axis) == base.axis_size(expected)
    else:
        assert base is None
        assert view.axis_size(axis) == expected


def check_data1d_names(
    view: StorageReader,
    axis: str,
    expected: Union[None, StorageReader, str, List[str]],
    base: Optional[StorageReader] = None,
) -> None:
    if expected is None:
        assert base is None
        assert not view.has_axis(axis)
    elif isinstance(expected, StorageReader):
        assert base is None
        assert set(view.data1d_names(axis)) == set(expected.data1d_names(axis))
    elif isinstance(expected, str):
        assert base is not None
        assert set(view.data1d_names(axis)) == set(base.data1d_names(expected))
    else:
        assert base is None
        assert set(view.data1d_names(axis)) == set(expected)


def check_data2d_names(
    view: StorageReader,
    axes: str,
    expected: Union[None, StorageReader, str, List[str]],
    base: Optional[StorageReader] = None,
) -> None:
    if expected is None:
        assert base is None
        assert not view.has_axis(axes.split(",")[0]) or not view.has_axis(axes.split(",")[1])
    elif isinstance(expected, StorageReader):
        assert set(view.data2d_names(axes)) == set(expected.data2d_names(axes))
    elif isinstance(expected, str):
        assert base is not None
        assert set(view.data2d_names(axes)) == set(base.data2d_names(expected))
    else:
        assert base is None
        assert set(view.data2d_names(axes)) == set(expected)


def check_item_value(
    view: StorageReader, name: str, expected: Union[None, StorageReader, str], base: Optional[StorageReader] = None
) -> None:
    if expected is None:
        assert base is None
        assert not view.has_item(name)
    elif isinstance(expected, StorageReader):
        assert base is None
        assert view.get_item(name) == expected.get_item(name)
    elif isinstance(expected, str):
        assert base is not None
        assert view.get_item(name) == base.get_item(expected)
    else:
        assert base is None
        assert view.get_item(name) == expected


def check_vector_value(
    view: StorageReader,
    name: str,
    expected: Union[None, StorageReader, str, Vector],
    base: Optional[StorageReader] = None,
) -> None:
    if expected is None:
        assert base is None
        assert not view.has_data1d(name)
    elif isinstance(expected, StorageReader):
        assert base is None
        assert np.all(view.get_data1d(name) == expected.get_data1d(name))
    elif isinstance(expected, str):
        assert base is not None
        assert np.all(view.get_data1d(name) == base.get_data1d(expected))
    else:
        assert base is None
        assert np.all(view.get_data1d(name) == expected)


def check_data2d_value(
    view: StorageReader,
    name: str,
    expected: Union[None, StorageReader, str, Dense],
    base: Optional[StorageReader] = None,
) -> None:
    if expected is None:
        assert base is None
        assert not view.has_data2d(name)
    elif isinstance(expected, StorageReader):
        assert base is None
        assert fast_all_close(view.get_data2d(name), expected.get_data2d(name))
    elif isinstance(expected, str):
        assert base is not None
        assert fast_all_close(view.get_data2d(name), base.get_data2d(expected))
    else:
        assert base is None
        assert fast_all_close(view.get_data2d(name), expected)


def check_same_data(view: StorageView, base: StorageReader) -> None:
    check_item_names(view, base)
    assert view.base_item("zero") == "zero"
    assert view.base_item("one") == "one"
    assert view.exposed_item("zero") == "zero"
    assert view.exposed_item("one") == "one"
    check_item_value(view, "zero", base)
    check_item_value(view, "one", base)


# pylint: disable=duplicate-code
def check_same_axes(view: StorageView, base: StorageReader) -> None:
    check_axis_names(view, base)
    check_axis_size(view, "cell", base)
    check_axis_size(view, "gene", base)
    check_axis_entries(view, "cell", base)
    check_axis_entries(view, "gene", base)
    assert view.base_axis("cell") == "cell"
    assert view.base_axis("gene") == "gene"
    assert view.exposed_axis("cell") == "cell"
    assert view.exposed_axis("gene") == "gene"


def check_same_vector(view: StorageView, base: StorageReader) -> None:
    check_data1d_names(view, "cell", base)
    assert view.base_data1d("cell#type") == "cell#type"
    assert view.exposed_data1d("cell#type") == "cell#type"
    check_vector_value(view, "cell#type", base)


def check_same_data2d(view: StorageView, base: StorageReader) -> None:
    check_data2d_names(view, "cell,gene", base)
    assert view.base_data2d("cell,gene#UMIs") == "cell,gene#UMIs"
    assert view.exposed_data2d("cell,gene#UMIs") == "cell,gene#UMIs"
    check_data2d_value(view, "cell,gene#UMIs", base)


def test_default_view() -> None:
    base = make_base()
    view = StorageView(base, name="view")

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_vector(view, base)
    check_same_data2d(view, base)

    expect_description(
        view,
        detail=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell in frozen 1D numpy.ndarray of 2 of <U5
                gene: all 3 entries of gene in frozen 1D numpy.ndarray of 3 of <U5
              data:
                one: from one in builtins.float = 1.0
                zero: from zero in builtins.float = 0.0
                cell#type: from cell#type in frozen 1D numpy.ndarray of 2 of <U1
                cell,gene#UMIs: from cell,gene#UMIs in frozen row-major numpy.ndarray of 2x3 of int64
            """,
    )

    expect_description(
        view,
        detail=True,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell in frozen 1D numpy.ndarray of 2 of <U5
                gene: all 3 entries of gene in frozen 1D numpy.ndarray of 3 of <U5
              data:
                one: from one in builtins.float = 1.0
                zero: from zero in builtins.float = 0.0
                cell#type: from cell#type in frozen 1D numpy.ndarray of 2 of <U1
                cell,gene#UMIs: from cell,gene#UMIs in frozen row-major numpy.ndarray of 2x3 of int64
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: frozen 1D numpy.ndarray of 2 of <U5
                gene: frozen 1D numpy.ndarray of 3 of <U5
              data: {}
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: frozen 1D numpy.ndarray of 2 of <U5
                gene: frozen 1D numpy.ndarray of 3 of <U5
              data:
                one: builtins.float = 1.0
                zero: builtins.float = 0.0
                cell#type: frozen 1D numpy.ndarray of 2 of <U1
                cell,gene#UMIs: frozen row-major numpy.ndarray of 2x3 of int64
            """,
    )


def test_hide_item() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"one": None})

    check_same_axes(view, base)
    check_same_vector(view, base)
    check_same_data2d(view, base)

    check_item_names(view, ["zero"])
    assert view.base_item("zero") == "zero"
    assert view.exposed_item("zero") == "zero"
    assert view.exposed_item("one") is None
    check_item_value(view, "zero", base)
    check_item_value(view, "one", None)

    expect_description(
        view,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell
                gene: all 3 entries of gene
              data:
                zero: from zero
                cell#type: from cell#type
                cell,gene#UMIs: from cell,gene#UMIs
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data: []
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - one
              - zero
              - cell#type
              - cell,gene#UMIs
            """,
    )


def test_hide_axis() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(gene=None))

    check_same_data(view, base)
    check_same_vector(view, base)

    check_axis_names(view, ["cell"])
    check_axis_size(view, "cell", base)
    check_axis_entries(view, "cell", base)
    assert view.base_axis("cell") == "cell"
    assert view.exposed_axis("cell") == "cell"
    assert view.exposed_axis("gene") is None

    check_data2d_names(view, "cell,gene", None)
    check_data2d_value(view, "cell,gene#UMIs", None)

    expect_description(
        view,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell
              data:
                one: from one
                zero: from zero
                cell#type: from cell#type
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
              data: []
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - one
              - zero
              - cell#type
              - cell,gene#UMIs
            """,
    )


def test_hide_vector() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cell#type": None})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_data2d(view, base)

    check_data1d_names(view, "cell", [])
    assert view.exposed_data1d("cell#type") is None
    check_vector_value(view, "cell#type", None)

    expect_description(
        view,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell
                gene: all 3 entries of gene
              data:
                one: from one
                zero: from zero
                cell,gene#UMIs: from cell,gene#UMIs
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data: []
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - one
              - zero
              - cell#type
              - cell,gene#UMIs
            """,
    )


def test_hide_data2d() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cell,gene#UMIs": None})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_vector(view, base)

    check_data2d_names(view, "cell,gene", [])
    assert view.exposed_data2d("cell,gene#UMIs") is None
    check_data2d_value(view, "cell,gene#UMIs", None)

    expect_description(
        view,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell
                gene: all 3 entries of gene
              data:
                one: from one
                zero: from zero
                cell#type: from cell#type
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data: []
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - one
              - zero
              - cell#type
              - cell,gene#UMIs
            """,
    )


def test_rename_item() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"zero": "none"})

    check_same_axes(view, base)
    check_same_vector(view, base)
    check_same_data2d(view, base)

    check_item_names(view, ["none", "one"])
    assert view.base_item("none") == "zero"
    assert view.base_item("one") == "one"
    assert view.exposed_item("zero") == "none"
    assert view.exposed_item("one") == "one"
    check_item_value(view, "none", "zero", base)
    check_item_value(view, "one", base)

    expect_description(
        view,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell
                gene: all 3 entries of gene
              data:
                none: from zero
                one: from one
                cell#type: from cell#type
                cell,gene#UMIs: from cell,gene#UMIs
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data: []
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - one
              - zero
              - cell#type
              - cell,gene#UMIs
            """,
    )


def test_rename_axis() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(gene="value"))

    check_same_data(view, base)
    check_same_vector(view, base)

    check_axis_names(view, ["cell", "value"])
    check_axis_size(view, "cell", base)
    check_axis_size(view, "value", "gene", base)
    check_axis_entries(view, "cell", base)
    check_axis_entries(view, "value", "gene", base)
    assert view.base_axis("cell") == "cell"
    assert view.base_axis("value") == "gene"
    assert view.exposed_axis("cell") == "cell"
    assert view.exposed_axis("gene") == "value"

    check_data2d_names(view, "cell,value", ["cell,value#UMIs"])
    assert view.base_data2d("cell,value#UMIs") == "cell,gene#UMIs"
    assert view.exposed_data2d("cell,gene#UMIs") == "cell,value#UMIs"
    check_data2d_value(view, "cell,value#UMIs", "cell,gene#UMIs", base)

    expect_description(
        view,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell
                value: all 3 entries of gene
              data:
                one: from one
                zero: from zero
                cell#type: from cell#type
                cell,value#UMIs: from cell,gene#UMIs
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                value: 3 entries
              data: []
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - one
              - zero
              - cell#type
              - cell,gene#UMIs
            """,
    )


def test_rename_vector() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cell#type": "kind"})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_data2d(view, base)

    check_data1d_names(view, "cell", ["cell#kind"])
    assert view.base_data1d("cell#kind") == "cell#type"
    assert view.exposed_data1d("cell#type") == "cell#kind"
    check_vector_value(view, "cell#kind", "cell#type", base)

    expect_description(
        view,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell
                gene: all 3 entries of gene
              data:
                one: from one
                zero: from zero
                cell#kind: from cell#type
                cell,gene#UMIs: from cell,gene#UMIs
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data: []
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - one
              - zero
              - cell#type
              - cell,gene#UMIs
            """,
    )


def test_rename_data2d() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cell,gene#UMIs": "counts"})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_vector(view, base)

    check_data2d_names(view, "cell,gene", ["cell,gene#counts"])
    assert view.base_data2d("cell,gene#counts") == "cell,gene#UMIs"
    assert view.exposed_data2d("cell,gene#UMIs") == "cell,gene#counts"
    check_data2d_value(view, "cell,gene#counts", "cell,gene#UMIs", base)

    expect_description(
        view,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell
                gene: all 3 entries of gene
              data:
                one: from one
                zero: from zero
                cell#type: from cell#type
                cell,gene#counts: from cell,gene#UMIs
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data: []
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - one
              - zero
              - cell#type
              - cell,gene#UMIs
            """,
    )


def test_slice_axis_strings() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(gene=as_vector(["gene2", "gene0"])))

    check_same_data(view, base)
    check_same_vector(view, base)

    check_axis_names(view, base)
    check_axis_size(view, "cell", base)
    check_axis_size(view, "gene", 2)
    check_axis_entries(view, "cell", base)
    check_axis_entries(view, "gene", as_vector(["gene2", "gene0"]))
    assert view.base_axis("cell") == "cell"
    assert view.base_axis("gene") == "gene"
    assert view.exposed_axis("cell") == "cell"
    assert view.exposed_axis("gene") == "gene"

    check_data2d_names(view, "cell,gene", base)
    assert view.base_data2d("cell,gene#UMIs") == "cell,gene#UMIs"
    assert view.exposed_data2d("cell,gene#UMIs") == "cell,gene#UMIs"
    check_data2d_value(view, "cell,gene#UMIs", as_dense([[90, 0], [0, 190]]))
    check_data2d_value(view, "cell,gene#UMIs", as_dense([[90, 0], [0, 190]]))

    expect_description(
        view,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell
                gene: 2 out of 3 entries of gene (66.67%)
              data:
                one: from one
                zero: from zero
                cell#type: from cell#type
                cell,gene#UMIs: from cell,gene#UMIs
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 2 entries
              data:
              - cell,gene#UMIs
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - one
              - zero
              - cell#type
              - cell,gene#UMIs
            """,
    )


def test_slice_axis_mask() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(gene=as_vector([True, False, True])))

    check_same_data(view, base)
    check_same_vector(view, base)

    check_axis_names(view, base)
    check_axis_size(view, "cell", base)
    check_axis_size(view, "gene", 2)
    check_axis_entries(view, "cell", base)
    check_axis_entries(view, "gene", as_vector(["gene0", "gene2"]))
    assert view.base_axis("cell") == "cell"
    assert view.base_axis("gene") == "gene"
    assert view.exposed_axis("cell") == "cell"
    assert view.exposed_axis("gene") == "gene"

    check_data2d_names(view, "cell,gene", base)
    assert view.base_data2d("cell,gene#UMIs") == "cell,gene#UMIs"
    assert view.exposed_data2d("cell,gene#UMIs") == "cell,gene#UMIs"
    check_data2d_value(view, "cell,gene#UMIs", as_dense([[0, 90], [190, 0]]))
    check_data2d_value(view, "cell,gene#UMIs", as_dense([[0, 90], [190, 0]]))

    expect_description(
        view,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                cell: all 2 entries of cell
                gene: 2 out of 3 entries of gene (66.67%)
              data:
                one: from one
                zero: from zero
                cell#type: from cell#type
                cell,gene#UMIs: from cell,gene#UMIs
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 2 entries
              data:
              - cell,gene#UMIs
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - one
              - zero
              - cell#type
              - cell,gene#UMIs
            """,
    )


def test_rename_slice_axis_indices() -> None:
    base = make_base()
    view = StorageView(
        base, name="view", axes=dict(cell=AxisView(name="place", track="full_index", entries=as_vector([1, 0])))
    )

    check_same_data(view, base)

    check_axis_names(view, ["place", "gene"])
    check_axis_size(view, "place", "cell", base)
    check_axis_size(view, "gene", base)
    check_axis_entries(view, "place", as_vector(["cell1", "cell0"]))
    check_axis_entries(view, "gene", base)
    assert view.base_axis("place") == "cell"
    assert view.base_axis("gene") == "gene"
    assert view.exposed_axis("cell") == "place"
    assert view.exposed_axis("gene") == "gene"

    check_data1d_names(view, "place", ["place#type", "place#full_index"])

    assert view.base_data1d("place#type") == "cell#type"
    assert view.exposed_data1d("cell#type") == "place#type"
    check_vector_value(view, "place#type", as_vector(["B", "T"]))
    check_vector_value(view, "place#type", as_vector(["B", "T"]))

    assert view.base_data1d("place#full_index") == "cell#"
    assert view.exposed_data1d("cell#") == "place#full_index"
    check_vector_value(view, "place#full_index", as_vector([1, 0]))
    check_vector_value(view, "place#full_index", as_vector([1, 0]))

    check_data2d_names(view, "place,gene", ["place,gene#UMIs"])
    assert view.base_data2d("place,gene#UMIs") == "cell,gene#UMIs"
    assert view.exposed_data2d("cell,gene#UMIs") == "place,gene#UMIs"
    check_data2d_value(view, "place,gene#UMIs", as_dense([[190, 10, 0], [0, 10, 90]]))
    check_data2d_value(view, "place,gene#UMIs", as_dense([[190, 10, 0], [0, 10, 90]]))

    expect_description(
        view,
        deep=True,
        expected="""
            view:
              class: daf.storage.views.StorageView
              cache: view.cache
              base: base
              axes:
                gene: all 3 entries of gene
                place: 2 out of 2 entries of cell (100.00%)
              data:
                one: from one
                zero: from zero
                place#full_index: from cell#
                place#type: from cell#type
                place,gene#UMIs: from cell,gene#UMIs
            view.cache:
              class: daf.storage.memory.MemoryStorage
              axes:
                gene: 3 entries
                place: 2 entries
              data:
              - place#type
              - place,gene#UMIs
            base:
              class: daf.storage.memory.MemoryStorage
              axes:
                cell: 2 entries
                gene: 3 entries
              data:
              - one
              - zero
              - cell#type
              - cell,gene#UMIs
            """,
    )
