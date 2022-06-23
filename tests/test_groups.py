"""
Test ``daf.groups``.
"""

import numpy as np
import scipy.sparse as sp  # type: ignore

from daf import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-function-docstring


def sample_data(*, gap: bool = False, sparse: bool = False) -> DafWriter:
    data = DafWriter(MemoryStorage(name="test"), name=".daf")

    data.create_axis("cell", ["cell0", "cell1", "cell2", "cell3", "cell4"])
    data.create_axis("gene", ["gene0", "gene1"])
    data.create_axis("age", ["1.0", "1.5", "2.0"])
    if gap:
        data.set_data1d("cell;metacell", [2, -1, 0, 2, 2])
    else:
        data.set_data1d("cell;metacell", [1, -1, 0, 1, 1])
    if sparse:
        data.set_data2d("cell,gene;UMIs", sp.csr_matrix([[1, 2], [3, 4], [5, 6], [7, 8], [0, 10]]))
    else:
        data.set_data2d("cell,gene;UMIs", np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 10]]))
    data.set_data1d("cell;age", [1, 2, 1, 1.5, 1])

    with data.adapter(
        axes=dict(cell="member"), data={"cell;metacell": "group"}, back_axes=dict(group="metacell")
    ) as work:
        create_group_axis(work, format="metacell%s")
    if gap:
        assert list(data.axis_entries("metacell")) == ["metacell0", "metacell1", "metacell2"]
        data.set_data1d("metacell;type", ["type0", "type2", "type1"])
    else:
        assert list(data.axis_entries("metacell")) == ["metacell0", "metacell1"]
        data.set_data1d("metacell;type", ["type0", "type1"])

    return data


def test_count_group_members() -> None:
    data = sample_data()
    with data.adapter(
        axes=dict(cell="member", metacell="group"),
        data={"cell;metacell": "group"},
        back_data={"group;members": "cells"},
    ) as work:
        count_group_members(work)
    assert list(data.get_vector("metacell;cells")) == [1, 3]


def test_gap_count_group_members() -> None:
    data = sample_data(gap=True)
    with data.adapter(
        axes=dict(cell="member", metacell="group"),
        data={"cell;metacell": "group"},
        back_data={"group;members": "cells"},
    ) as work:
        count_group_members(work)
    assert list(data.get_vector("metacell;cells")) == [1, 0, 3]


def test_assign_group_values() -> None:
    data = sample_data()
    with data.adapter(
        axes=dict(cell="member", metacell="group"),
        data={"cell;metacell": "group", "metacell;type": "value"},
        back_data={"member;value": "type"},
    ) as work:
        assign_group_values(work, default="outlier")
    assert list(data.get_vector("cell;type")) == ["type1", "outlier", "type0", "type1", "type1"]


def test_assign_gap_group_values() -> None:
    data = sample_data(gap=True)
    with data.adapter(
        axes=dict(cell="member", metacell="group"),
        data={"cell;metacell": "group", "metacell;type": "value"},
        back_data={"member;value": "type"},
    ) as work:
        assign_group_values(work, default="outlier")
    assert list(data.get_vector("cell;type")) == ["type1", "outlier", "type0", "type1", "type1"]


def test_count_group_values() -> None:
    data = sample_data()
    with data.adapter(
        axes=dict(cell="member", metacell="group", age="value"),
        data={"cell;metacell": "group", "cell;age": "value"},
        back_data={"group,value;members": "cells"},
    ) as work:
        count_group_values(work)
    assert np.allclose(as_dense(data.get_matrix("metacell,age;cells")), np.array([[1, 0, 0], [2, 1, 0]]))


def test_count_group_dense_values() -> None:
    data = sample_data()
    with data.adapter(
        axes=dict(cell="member", metacell="group", age="value"),
        data={"cell;metacell": "group", "cell;age": "value"},
        back_data={"group,value;members": "cells"},
    ) as work:
        count_group_values(work, dense=True)
    assert np.allclose(data.get_matrix("metacell,age;cells"), np.array([[1, 0, 0], [2, 1, 0]]))


def test_count_gap_group_values() -> None:
    data = sample_data(gap=True)
    with data.adapter(
        axes=dict(cell="member", metacell="group", age="value"),
        data={"cell;metacell": "group", "cell;age": "value"},
        back_data={"group,value;members": "cells"},
    ) as work:
        count_group_values(work)
    assert np.allclose(as_dense(data.get_matrix("metacell,age;cells")), np.array([[1, 0, 0], [0, 0, 0], [2, 1, 0]]))


def test_aggregate_group_data1d() -> None:
    data = sample_data()

    with data.adapter(
        axes=dict(cell="member", metacell="group", age="value"),
        data={"cell;metacell": "group", "cell;age": "value"},
        back_data={"group;value": "age.mean"},
    ) as work:
        aggregate_group_data1d(work, np.mean)
    assert list(data.get_vector("metacell;age.mean")) == [1.0, 3.5 / 3]

    with data.adapter(
        axes=dict(cell="member", metacell="group", age="value"),
        data={"cell;metacell": "group", "cell;age": "value"},
        back_data={"group;value": "age.frequent"},
    ) as work:
        aggregate_group_data1d(work, most_frequent)
    assert list(data.get_vector("metacell;age.frequent")) == [1.0, 1.0]


def test_aggregate_gap_group_data1d() -> None:
    data = sample_data(gap=True)

    with data.adapter(
        axes=dict(cell="member", metacell="group", age="value"),
        data={"cell;metacell": "group", "cell;age": "value"},
        back_data={"group;value": "age.mean"},
    ) as work:
        aggregate_group_data1d(work, np.mean)
    assert list(data.get_vector("metacell;age.mean")) == [1.0, None, 3.5 / 3]

    with data.adapter(
        axes=dict(cell="member", metacell="group", age="value"),
        data={"cell;metacell": "group", "cell;age": "value"},
        back_data={"group;value": "age.frequent"},
    ) as work:
        aggregate_group_data1d(work, most_frequent)
    assert list(data.get_vector("metacell;age.frequent")) == [1.0, None, 1.0]


def test_aggregate_group_data2d() -> None:
    data = sample_data()
    with data.adapter(
        axes=dict(cell="member", metacell="group", gene="data"),
        data={"cell;metacell": "group", "cell,gene;UMIs": "value"},
        back_data={"group,data;value": "UMIs"},
    ) as work:
        aggregate_group_data2d(work, np.sum)
    assert np.allclose(data.get_matrix("metacell,gene;UMIs"), np.array([[5, 6], [8, 20]]))


def test_aggregate_gap_group_data2d() -> None:
    data = sample_data(gap=True)
    with data.adapter(
        axes=dict(cell="member", metacell="group", gene="data"),
        data={"cell;metacell": "group", "cell,gene;UMIs": "value"},
        back_data={"group,data;value": "UMIs"},
    ) as work:
        aggregate_group_data2d(work, np.sum, default=0)
    assert np.allclose(data.get_matrix("metacell,gene;UMIs"), np.array([[5, 6], [0, 0], [8, 20]]))


def test_aggregate_group_sparse_data2d() -> None:
    data = sample_data(sparse=True)
    with data.adapter(
        axes=dict(cell="member", metacell="group", gene="data"),
        data={"cell;metacell": "group", "cell,gene;UMIs": "value"},
        back_data={"group,data;value": "UMIs"},
    ) as work:
        aggregate_group_data2d(work, np.sum)
    assert np.allclose(data.get_matrix("metacell,gene;UMIs"), np.array([[5, 6], [8, 20]]))


def test_aggregate_gap_group_sparse_data2d() -> None:
    data = sample_data(gap=True, sparse=True)
    with data.adapter(
        axes=dict(cell="member", metacell="group", gene="data"),
        data={"cell;metacell": "group", "cell,gene;UMIs": "value"},
        back_data={"group,data;value": "UMIs"},
    ) as work:
        aggregate_group_data2d(work, np.sum, default=0)
    assert np.allclose(data.get_matrix("metacell,gene;UMIs"), np.array([[5, 6], [0, 0], [8, 20]]))
