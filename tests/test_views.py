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
    memory.create_axis("cells", cell_names)

    gene_names = freeze(as_array1d(["gene0", "gene1", "gene2"]))
    memory.create_axis("genes", gene_names)

    cell_types = freeze(as_array1d(["T", "B"]))
    memory.set_array1d("cells:type", cell_types)

    umis = freeze(be_array_in_rows(as_array2d([[0, 10, 90], [190, 10, 0]])))
    memory.set_grid("cells,genes:UMIs", umis)

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
    check_axis_size(view, "cells", base)
    check_axis_size(view, "genes", base)
    check_axis_entries(view, "cells", base)
    check_axis_entries(view, "genes", base)
    assert view.wrapped_axis("cells") == "cells"
    assert view.wrapped_axis("genes") == "genes"
    assert view.exposed_axis("cells") == "cells"
    assert view.exposed_axis("genes") == "genes"


def check_same_array1d(view: StorageView, base: StorageReader) -> None:
    check_array1d_names(view, "cells", base)
    assert view.wrapped_array1d("cells:type") == "cells:type"
    assert view.exposed_array1d("cells:type") == "cells:type"
    check_array1d_value(view, "cells:type", base)


def check_same_data2d(view: StorageView, base: StorageReader) -> None:
    check_data2d_names(view, "cells,genes", base)
    assert view.wrapped_data2d("cells,genes:UMIs") == "cells,genes:UMIs"
    assert view.exposed_data2d("cells,genes:UMIs") == "cells,genes:UMIs"
    check_data2d_value(view, "cells,genes:UMIs", base)


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
    view = StorageView(base, name="view", axes=dict(genes=None))

    check_same_data(view, base)
    check_same_array1d(view, base)

    check_axis_names(view, ["cells"])
    check_axis_size(view, "cells", base)
    check_axis_entries(view, "cells", base)
    assert view.wrapped_axis("cells") == "cells"
    assert view.exposed_axis("cells") == "cells"
    assert view.exposed_axis("genes") is None

    check_data2d_names(view, "cells,genes", None)
    check_data2d_value(view, "cells,genes:UMIs", None)


def test_hide_array1d() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cells:type": None})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_data2d(view, base)

    check_array1d_names(view, "cells", [])
    assert view.exposed_array1d("cells:type") is None
    check_array1d_value(view, "cells:type", None)


def test_hide_data2d() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cells,genes:UMIs": None})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_array1d(view, base)

    check_data2d_names(view, "cells,genes", [])
    assert view.exposed_data2d("cells,genes:UMIs") is None
    check_data2d_value(view, "cells,genes:UMIs", None)


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
    view = StorageView(base, name="view", axes=dict(genes="values"))

    check_same_data(view, base)
    check_same_array1d(view, base)

    check_axis_names(view, ["cells", "values"])
    check_axis_size(view, "cells", base)
    check_axis_size(view, "values", "genes", base)
    check_axis_entries(view, "cells", base)
    check_axis_entries(view, "values", "genes", base)
    assert view.wrapped_axis("cells") == "cells"
    assert view.wrapped_axis("values") == "genes"
    assert view.exposed_axis("cells") == "cells"
    assert view.exposed_axis("genes") == "values"

    check_data2d_names(view, "cells,values", ["cells,values:UMIs"])
    assert view.wrapped_data2d("cells,values:UMIs") == "cells,genes:UMIs"
    assert view.exposed_data2d("cells,genes:UMIs") == "cells,values:UMIs"
    check_data2d_value(view, "cells,values:UMIs", "cells,genes:UMIs", base)


def test_rename_array1d() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cells:type": "kind"})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_data2d(view, base)

    check_array1d_names(view, "cells", ["cells:kind"])
    assert view.wrapped_array1d("cells:kind") == "cells:type"
    assert view.exposed_array1d("cells:type") == "cells:kind"
    check_array1d_value(view, "cells:kind", "cells:type", base)


def test_rename_data2d() -> None:
    base = make_base()
    view = StorageView(base, name="view", data={"cells,genes:UMIs": "counts"})

    check_same_data(view, base)
    check_same_axes(view, base)
    check_same_array1d(view, base)

    check_data2d_names(view, "cells,genes", ["cells,genes:counts"])
    assert view.wrapped_data2d("cells,genes:counts") == "cells,genes:UMIs"
    assert view.exposed_data2d("cells,genes:UMIs") == "cells,genes:counts"
    check_data2d_value(view, "cells,genes:counts", "cells,genes:UMIs", base)


def test_slice_axis_strings() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(genes=as_array1d(["gene2", "gene0"])))

    check_same_data(view, base)
    check_same_array1d(view, base)

    check_axis_names(view, base)
    check_axis_size(view, "cells", base)
    check_axis_size(view, "genes", 2)
    check_axis_entries(view, "cells", base)
    check_axis_entries(view, "genes", as_array1d(["gene2", "gene0"]))
    assert view.wrapped_axis("cells") == "cells"
    assert view.wrapped_axis("genes") == "genes"
    assert view.exposed_axis("cells") == "cells"
    assert view.exposed_axis("genes") == "genes"

    check_data2d_names(view, "cells,genes", base)
    assert view.wrapped_data2d("cells,genes:UMIs") == "cells,genes:UMIs"
    assert view.exposed_data2d("cells,genes:UMIs") == "cells,genes:UMIs"
    check_data2d_value(view, "cells,genes:UMIs", as_array2d([[90, 0], [0, 190]]))
    check_data2d_value(view, "cells,genes:UMIs", as_array2d([[90, 0], [0, 190]]))


def test_slice_axis_mask() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(genes=as_array1d([True, False, True])))

    check_same_data(view, base)
    check_same_array1d(view, base)

    check_axis_names(view, base)
    check_axis_size(view, "cells", base)
    check_axis_size(view, "genes", 2)
    check_axis_entries(view, "cells", base)
    check_axis_entries(view, "genes", as_array1d(["gene0", "gene2"]))
    assert view.wrapped_axis("cells") == "cells"
    assert view.wrapped_axis("genes") == "genes"
    assert view.exposed_axis("cells") == "cells"
    assert view.exposed_axis("genes") == "genes"

    check_data2d_names(view, "cells,genes", base)
    assert view.wrapped_data2d("cells,genes:UMIs") == "cells,genes:UMIs"
    assert view.exposed_data2d("cells,genes:UMIs") == "cells,genes:UMIs"
    check_data2d_value(view, "cells,genes:UMIs", as_array2d([[0, 90], [190, 0]]))
    check_data2d_value(view, "cells,genes:UMIs", as_array2d([[0, 90], [190, 0]]))


def test_rename_slice_axis_indices() -> None:
    base = make_base()
    view = StorageView(base, name="view", axes=dict(cells=("places", as_array1d([1, 0]))))

    check_same_data(view, base)

    check_axis_names(view, ["places", "genes"])
    check_axis_size(view, "places", "cells", base)
    check_axis_size(view, "genes", base)
    check_axis_entries(view, "places", as_array1d(["cell1", "cell0"]))
    check_axis_entries(view, "genes", base)
    assert view.wrapped_axis("places") == "cells"
    assert view.wrapped_axis("genes") == "genes"
    assert view.exposed_axis("cells") == "places"
    assert view.exposed_axis("genes") == "genes"

    check_array1d_names(view, "places", ["places:type"])
    assert view.wrapped_array1d("places:type") == "cells:type"
    assert view.exposed_array1d("cells:type") == "places:type"
    check_array1d_value(view, "places:type", as_array1d(["B", "T"]))
    check_array1d_value(view, "places:type", as_array1d(["B", "T"]))

    check_data2d_names(view, "places,genes", ["places,genes:UMIs"])
    assert view.wrapped_data2d("places,genes:UMIs") == "cells,genes:UMIs"
    assert view.exposed_data2d("cells,genes:UMIs") == "places,genes:UMIs"
    check_data2d_value(view, "places,genes:UMIs", as_array2d([[190, 10, 0], [0, 10, 90]]))
    check_data2d_value(view, "places,genes:UMIs", as_array2d([[190, 10, 0], [0, 10, 90]]))
