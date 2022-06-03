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

# pylint: disable=missing-function-docstring


def make_base() -> MemoryStorage:
    memory = MemoryStorage(name="base")

    memory.set_datum("zero", 0.0)
    memory.set_datum("one", 1.0)

    cell_names = freeze(as_array1d(["cell0", "cell1"]))
    memory.create_axis("cell", cell_names)

    gene_names = freeze(as_array1d(["gene0", "gene1", "gene2"]))
    memory.create_axis("gene", gene_names)

    cell_types = freeze(as_array1d(["T", "B"]))
    memory.set_array1d("cell;type", cell_types)

    umis = freeze(be_array_in_rows(as_array2d([[0, 10, 90], [190, 10, 0]])))
    memory.set_grid("cell,gene;UMIs", umis)

    return memory


def check_datum_names(view: StorageReader, expected: Union[StorageReader, List[str]]) -> None:
    if isinstance(expected, StorageReader):
        assert set(view.datum_names()) == set(expected.datum_names())
    else:
        assert set(view.datum_names()) == set(expected)


def check_axis_names(view: StorageReader, expected: Union[StorageReader, List[str]]) -> None:
    if isinstance(expected, StorageReader):
        assert set(view.axis_names()) == set(expected.axis_names())
    else:
        assert set(view.axis_names()) == set(expected)


def check_axis_entries(
    view: StorageReader, axis: str, expected: Union[StorageReader, str, Array1D], base: Optional[StorageReader] = None
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


def check_array1d_names(
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
        assert set(view.array1d_names(axis)) == set(expected.array1d_names(axis))
    elif isinstance(expected, str):
        assert base is not None
        assert set(view.array1d_names(axis)) == set(base.array1d_names(expected))
    else:
        assert base is None
        assert set(view.array1d_names(axis)) == set(expected)


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


def check_datum_value(
    view: StorageReader, name: str, expected: Union[None, StorageReader, str], base: Optional[StorageReader] = None
) -> None:
    if expected is None:
        assert base is None
        assert not view.has_datum(name)
    elif isinstance(expected, StorageReader):
        assert base is None
        assert view.get_datum(name) == expected.get_datum(name)
    elif isinstance(expected, str):
        assert base is not None
        assert view.get_datum(name) == base.get_datum(expected)
    else:
        assert base is None
        assert view.get_datum(name) == expected


def check_array1d_value(
    view: StorageReader,
    name: str,
    expected: Union[None, StorageReader, str, Array1D],
    base: Optional[StorageReader] = None,
) -> None:
    if expected is None:
        assert base is None
        assert not view.has_array1d(name)
    elif isinstance(expected, StorageReader):
        assert base is None
        assert np.all(view.get_array1d(name) == expected.get_array1d(name))
    elif isinstance(expected, str):
        assert base is not None
        assert np.all(view.get_array1d(name) == base.get_array1d(expected))
    else:
        assert base is None
        assert np.all(view.get_array1d(name) == expected)


def check_data2d_value(
    view: StorageReader,
    name: str,
    expected: Union[None, StorageReader, str, Array2D],
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
    check_datum_names(view, base)
    assert view.wrapped_datum("zero") == "zero"
    assert view.wrapped_datum("one") == "one"
    assert view.exposed_datum("zero") == "zero"
    assert view.exposed_datum("one") == "one"
    check_datum_value(view, "zero", base)
    check_datum_value(view, "one", base)


def check_same_axes(view: StorageView, base: StorageReader) -> None:
    check_axis_names(view, base)
    check_axis_size(view, "cell", base)
    check_axis_size(view, "gene", base)
    check_axis_entries(view, "cell", base)
    check_axis_entries(view, "gene", base)
    assert view.wrapped_axis("cell") == "cell"
    assert view.wrapped_axis("gene") == "gene"
    assert view.exposed_axis("cell") == "cell"
    assert view.exposed_axis("gene") == "gene"


def check_same_array1d(view: StorageView, base: StorageReader) -> None:
    check_array1d_names(view, "cell", base)
    assert view.wrapped_array1d("cell;type") == "cell;type"
    assert view.exposed_array1d("cell;type") == "cell;type"
    check_array1d_value(view, "cell;type", base)


def check_same_data2d(view: StorageView, base: StorageReader) -> None:
    check_data2d_names(view, "cell,gene", base)
    assert view.wrapped_data2d("cell,gene;UMIs") == "cell,gene;UMIs"
    assert view.exposed_data2d("cell,gene;UMIs") == "cell,gene;UMIs"
    check_data2d_value(view, "cell,gene;UMIs", base)


def test_default_view() -> None:
    base = make_base()
    view = StorageView(base, name="view")

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_array1d(view, base)
    check_same_data2d(view, base)


def test_hide_datum() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"one": None})

    check_same_axes(view, base)
    check_same_array1d(view, base)
    check_same_data2d(view, base)

    check_datum_names(view, ["zero"])
    assert view.wrapped_datum("zero") == "zero"
    assert view.exposed_datum("zero") == "zero"
    assert view.exposed_datum("one") is None
    check_datum_value(view, "zero", base)
    check_datum_value(view, "one", None)


def test_hide_axis() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(gene=None))

    check_same_data(view, base)
    check_same_array1d(view, base)

    check_axis_names(view, ["cell"])
    check_axis_size(view, "cell", base)
    check_axis_entries(view, "cell", base)
    assert view.wrapped_axis("cell") == "cell"
    assert view.exposed_axis("cell") == "cell"
    assert view.exposed_axis("gene") is None

    check_data2d_names(view, "cell,gene", None)
    check_data2d_value(view, "cell,gene;UMIs", None)


def test_hide_array1d() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cell;type": None})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_data2d(view, base)

    check_array1d_names(view, "cell", [])
    assert view.exposed_array1d("cell;type") is None
    check_array1d_value(view, "cell;type", None)


def test_hide_data2d() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cell,gene;UMIs": None})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_array1d(view, base)

    check_data2d_names(view, "cell,gene", [])
    assert view.exposed_data2d("cell,gene;UMIs") is None
    check_data2d_value(view, "cell,gene;UMIs", None)


def test_rename_datum() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"zero": "none"})

    check_same_axes(view, base)
    check_same_array1d(view, base)
    check_same_data2d(view, base)

    check_datum_names(view, ["none", "one"])
    assert view.wrapped_datum("none") == "zero"
    assert view.wrapped_datum("one") == "one"
    assert view.exposed_datum("zero") == "none"
    assert view.exposed_datum("one") == "one"
    check_datum_value(view, "none", "zero", base)
    check_datum_value(view, "one", base)


def test_rename_axis() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(gene="value"))

    check_same_data(view, base)
    check_same_array1d(view, base)

    check_axis_names(view, ["cell", "value"])
    check_axis_size(view, "cell", base)
    check_axis_size(view, "value", "gene", base)
    check_axis_entries(view, "cell", base)
    check_axis_entries(view, "value", "gene", base)
    assert view.wrapped_axis("cell") == "cell"
    assert view.wrapped_axis("value") == "gene"
    assert view.exposed_axis("cell") == "cell"
    assert view.exposed_axis("gene") == "value"

    check_data2d_names(view, "cell,value", ["cell,value;UMIs"])
    assert view.wrapped_data2d("cell,value;UMIs") == "cell,gene;UMIs"
    assert view.exposed_data2d("cell,gene;UMIs") == "cell,value;UMIs"
    check_data2d_value(view, "cell,value;UMIs", "cell,gene;UMIs", base)


def test_rename_array1d() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cell;type": "kind"})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_data2d(view, base)

    check_array1d_names(view, "cell", ["cell;kind"])
    assert view.wrapped_array1d("cell;kind") == "cell;type"
    assert view.exposed_array1d("cell;type") == "cell;kind"
    check_array1d_value(view, "cell;kind", "cell;type", base)


def test_rename_data2d() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cell,gene;UMIs": "counts"})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_array1d(view, base)

    check_data2d_names(view, "cell,gene", ["cell,gene;counts"])
    assert view.wrapped_data2d("cell,gene;counts") == "cell,gene;UMIs"
    assert view.exposed_data2d("cell,gene;UMIs") == "cell,gene;counts"
    check_data2d_value(view, "cell,gene;counts", "cell,gene;UMIs", base)


def test_slice_axis_strings() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(gene=as_array1d(["gene2", "gene0"])))

    check_same_data(view, base)
    check_same_array1d(view, base)

    check_axis_names(view, base)
    check_axis_size(view, "cell", base)
    check_axis_size(view, "gene", 2)
    check_axis_entries(view, "cell", base)
    check_axis_entries(view, "gene", as_array1d(["gene2", "gene0"]))
    assert view.wrapped_axis("cell") == "cell"
    assert view.wrapped_axis("gene") == "gene"
    assert view.exposed_axis("cell") == "cell"
    assert view.exposed_axis("gene") == "gene"

    check_data2d_names(view, "cell,gene", base)
    assert view.wrapped_data2d("cell,gene;UMIs") == "cell,gene;UMIs"
    assert view.exposed_data2d("cell,gene;UMIs") == "cell,gene;UMIs"
    check_data2d_value(view, "cell,gene;UMIs", as_array2d([[90, 0], [0, 190]]))
    check_data2d_value(view, "cell,gene;UMIs", as_array2d([[90, 0], [0, 190]]))


def test_slice_axis_mask() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(gene=as_array1d([True, False, True])))

    check_same_data(view, base)
    check_same_array1d(view, base)

    check_axis_names(view, base)
    check_axis_size(view, "cell", base)
    check_axis_size(view, "gene", 2)
    check_axis_entries(view, "cell", base)
    check_axis_entries(view, "gene", as_array1d(["gene0", "gene2"]))
    assert view.wrapped_axis("cell") == "cell"
    assert view.wrapped_axis("gene") == "gene"
    assert view.exposed_axis("cell") == "cell"
    assert view.exposed_axis("gene") == "gene"

    check_data2d_names(view, "cell,gene", base)
    assert view.wrapped_data2d("cell,gene;UMIs") == "cell,gene;UMIs"
    assert view.exposed_data2d("cell,gene;UMIs") == "cell,gene;UMIs"
    check_data2d_value(view, "cell,gene;UMIs", as_array2d([[0, 90], [190, 0]]))
    check_data2d_value(view, "cell,gene;UMIs", as_array2d([[0, 90], [190, 0]]))


def test_rename_slice_axis_indices() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(cell=("place", as_array1d([1, 0]))))

    check_same_data(view, base)

    check_axis_names(view, ["place", "gene"])
    check_axis_size(view, "place", "cell", base)
    check_axis_size(view, "gene", base)
    check_axis_entries(view, "place", as_array1d(["cell1", "cell0"]))
    check_axis_entries(view, "gene", base)
    assert view.wrapped_axis("place") == "cell"
    assert view.wrapped_axis("gene") == "gene"
    assert view.exposed_axis("cell") == "place"
    assert view.exposed_axis("gene") == "gene"

    check_array1d_names(view, "place", ["place;type"])
    assert view.wrapped_array1d("place;type") == "cell;type"
    assert view.exposed_array1d("cell;type") == "place;type"
    check_array1d_value(view, "place;type", as_array1d(["B", "T"]))
    check_array1d_value(view, "place;type", as_array1d(["B", "T"]))

    check_data2d_names(view, "place,gene", ["place,gene;UMIs"])
    assert view.wrapped_data2d("place,gene;UMIs") == "cell,gene;UMIs"
    assert view.exposed_data2d("cell,gene;UMIs") == "place,gene;UMIs"
    check_data2d_value(view, "place,gene;UMIs", as_array2d([[190, 10, 0], [0, 10, 90]]))
    check_data2d_value(view, "place,gene;UMIs", as_array2d([[190, 10, 0], [0, 10, 90]]))
