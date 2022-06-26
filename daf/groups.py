"""
Functions for projecting data between members and groups.

A common idiom is to have two axes such that one is a grouping of the other. For example, in scRNA-seq data, it is
common to group cells into clusters, so we have a ``cell`` axis and a ``cluster`` axis. Often this is a multi-level
grouping (``cell``, ``sub-cluster``, ``cluster``).

In this idiom, by convention there is a 1D data property for the "member" axis specifying the entry of the "group" axis
it belongs to. That is, we may see something like ``cell;cluster`` which gives for each cell the (integer, 0-based)
index of the cluster it belongs to. Since integers don't support ``NaN``, by convention any negative value (typically
``-1``) is used to say "this cell belongs to no cluster".

Computing such groups is the goal of complex analysis pipelines and is very much out of scope for a low-level package
such as ``daf``. However, once such group(s) are computed, there are universal operations, which it does make sense
to provide here:

**Aggregation**
    Compute 1D data for the group axis based on 1D data of the members axis. For example, suppose we have a
    ``cell;cluster`` as above, and also a ``cell;age`` 1D data. It is natural to want to aggregate it into a 1D
    ``cluster;age.mean`` which gives the mean cell age in each cluster.

**Counting**
    Compute 2D data for the group axis based on discrete 1D data of the members axis. For example, if ``age`` is
    actually discrete (which is common in scRNA-seq data), then for more detailed analysis, one may create an ``age``
    axis and then collect ``cluster,age;cells`` 2D data, which counts for each cluster and age the number of cells in
    the cluster that have that age.

**Assignment**
    Compute 1D data for the member axis based on 1D data of the group axis. For example, an analyst may provide
    ``cluster;type`` 1D data, which assigns a human-readable "cell type" to each cluster of cells. It would be natural
    to assign these types into a ``cell;type`` 1D data, which assigns each cell the type of the cluster it
    belongs to.
"""

# pylint: disable=duplicate-code

from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union
from typing import overload

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from .access import DafWriter
from .access import computation
from .typing import INT_DTYPES
from .typing import NUM_DTYPES
from .typing import DType
from .typing import Frame
from .typing import Series
from .typing import Vector
from .typing import as_vector
from .typing import assert_data
from .typing import be_frame
from .typing import be_vector
from .typing import dtype_of
from .typing import has_dtype
from .typing import is_dense
from .typing import is_dtype

# pylint: enable=duplicate-code

__all__ = [
    "aggregate_group_data1d",
    "aggregate_group_data2d",
    "most_frequent",
    "create_group_axis",
    "count_group_members",
    "count_group_values",
    "assign_group_values",
]


@computation(
    required_inputs={
        "member;group": """
            The index of the group each member belongs to.
            If negative, the member is not a part of any group.
            """,
        "member;value": "The value associated with each individual member.",
    },
    assured_outputs={"group;value": "The aggregated value associated with each group."},
)
def aggregate_group_data1d(
    data: DafWriter,
    aggregation: Callable[[Vector], Any],
    *,
    default: Any = None,
    dtype: Optional[DType] = None,
    overwrite: bool = False,
) -> None:
    """
    Compute a per-group value which is the result of applying the ``aggregation`` function to the vector of values of
    the members of the group.

    The ``aggregation`` function can be any function that converts a vector of all member values into a single group
    value. For example, for discrete data, `.most_frequent` will pick the value that appears in the highest number of
    members. An optimized version is used if the ``aggregation`` is one of ``numpy.sum``, ``numpy.mean``, ``numpy.var``,
    ``numpy.std``, ``numpy.median``, ``numpy.min`` or ``numpy.max``.

    The resulting per-group 1D data will have the specified ``dtype``. By default is the same as the data type of of the
    member values. This is acceptable for an aggregation like ``np.sum``, but would fail for an aggregation like
    ``np.mean`` for integer data.

    If no members are assigned to some existing group, then it is given the ``default`` value. By default this is
    ``None`` which is acceptable for floating point values (becomes a ``NaN``), but would fail for integer data.

    __DAF__
    """
    assert dtype is None or is_dtype(dtype), f"invalid dtype: {dtype}"
    member_groups = data.get_vector("member;group")
    assert_data(has_dtype(member_groups, INT_DTYPES), "group indices data", member_groups, dtype=INT_DTYPES)

    member_values_series = data.get_series("member;value")
    grouped = _aggregate_pandas(member_values_series, aggregation, member_groups, dtype)
    grouped = grouped[grouped.index >= 0]

    groups_count = data.axis_size("group")
    if len(grouped) == groups_count:
        group_values = grouped[np.arange(groups_count)]
    else:
        group_values = np.full(groups_count, default, dtype=dtype)
        group_values[grouped.index] = grouped.values

    data.set_data1d("group;value", group_values, overwrite=overwrite)


@computation(
    required_inputs={
        "member;group": "The index of the group each member belongs to. If negative, it is not a part of any group.",
        "member,data;value": "The value associated with each individual member and data axis entry.",
    },
    assured_outputs={"group,data;value": "The aggregated value associated with each group and data axis entry."},
)
def aggregate_group_data2d(  # pylint: disable=too-many-locals
    data: DafWriter,
    aggregation: Callable[[Vector], Any],
    *,
    default: Any = None,
    dtype: Optional[DType] = None,
    overwrite: bool = False,
) -> None:
    """
    Compute per-group-per-data values which are the result of applying the ``aggregation`` function to the vector of
    values of the members of the group.

    The ``aggregation`` function can be any function that converts a vector of all member values into a single group
    value. For example, for discrete data, `.most_frequent` will pick the value that appears in the highest number of
    members. An optimized version is used if the ``aggregation`` is one of ``numpy.sum``, ``numpy.mean``, ``numpy.var``,
    ``numpy.std``, ``numpy.median``, ``numpy.min`` or ``numpy.max``.

    The resulting per-group 1D data will have the specified ``dtype``. By default is the same as the data type of of the
    member values. This is acceptable for an aggregation like ``np.sum``, but would fail for an aggregation like
    ``np.mean`` for integer data.

    If no members are assigned to some existing group, then it is given the ``default`` value for all entries. By
    default this is ``None`` which is acceptable for floating point values (becomes a ``NaN``), but would fail for
    integer data.

    .. todo::

        Optimize `.aggregate_group_data2d` to avoid creating a temporary dense matrix per group for sparse data, and/or
        to parallelize the operation in general.

    __DAF__
    """
    assert dtype is None or is_dtype(dtype), f"invalid dtype: {dtype}"
    member_groups = data.get_vector("member;group")
    assert_data(has_dtype(member_groups, INT_DTYPES), "group indices data", member_groups, dtype=INT_DTYPES)

    member_data_values = data.get_matrix("member,data;value")
    dtype = dtype or dtype_of(member_data_values)
    assert dtype is not None
    groups_count = data.axis_size("group")
    data_count = data.axis_size("data")

    if is_dense(member_data_values):
        member_data_values_frame = pd.DataFrame(member_data_values)
        grouped = _aggregate_pandas(member_data_values_frame, aggregation, member_groups, dtype)
        grouped = grouped.iloc[grouped.index >= 0, :]

        if grouped.shape[0] == groups_count:
            group_values = grouped.iloc[np.arange(groups_count), :]
            data.set_data2d("group,data;value", group_values, overwrite=overwrite)

        else:
            with data.create_dense_in_rows("group,data;value", dtype=dtype, overwrite=overwrite) as group_values:
                group_values[:] = default
                group_values[grouped.index, :] = grouped.values[:]

    else:
        group_vectors: List[Vector] = []
        for group in range(groups_count):
            group_mask = member_groups == group
            if np.any(group_mask):
                group_member_data_values_frame = be_frame(pd.DataFrame(member_data_values[group_mask, :].toarray()))
                same_group = be_vector(np.full(group_member_data_values_frame.shape[0], group))
                grouped = _aggregate_pandas(group_member_data_values_frame, aggregation, same_group, dtype)
                group_vectors.append(as_vector(grouped))
            else:
                group_vectors.append(be_vector(np.full(data_count, default)))
        data.set_data2d("group,data;value", np.vstack(group_vectors), overwrite=overwrite)


@overload
def _aggregate_pandas(
    data: Frame, aggregation: Callable[[Vector], Any], groups: Vector, dtype: Optional[DType]
) -> Frame:
    ...


@overload
def _aggregate_pandas(
    data: Series, aggregation: Callable[[Vector], Any], groups: Vector, dtype: Optional[DType]
) -> Series:
    ...


def _aggregate_pandas(
    data: Union[Frame, Series], aggregation: Callable[[Vector], Any], groups: Vector, dtype: Optional[DType]
) -> Union[Frame, Series]:
    dtype = dtype or dtype_of(data)
    assert dtype is not None
    grouped = data.groupby(groups)
    if aggregation is np.sum:
        result = grouped.sum()
    if aggregation is np.mean:
        result = grouped.mean()
    if aggregation is np.var:
        result = grouped.var()
    if aggregation is np.std:
        result = grouped.std()
    if aggregation is np.median:
        result = grouped.median()
    if aggregation is np.min:
        result = grouped.min()
    if aggregation is np.max:
        result = grouped.max()
    else:
        result = grouped.apply(aggregation)
    return result.astype(dtype)


def most_frequent(vector: Vector) -> Any:
    """
    Return the most frequent value in a ``vector``.

    There is no guarantee that this value appears in the majority of the entries, or in general that it is "very
    common". The only guarantee is that there is no other value that is more common.
    """
    unique, positions = np.unique(vector, return_inverse=True)
    counts = np.bincount(positions)
    maxpos = np.argmax(counts)
    return unique[maxpos]


@computation(
    required_inputs={
        "member;group": "The index of the group each member belongs to. If negative, it is not a part of any group.",
    },
    assured_outputs={"group;members": "How many members exist in the group."},
)
def count_group_members(data: DafWriter, *, dtype: DType = "int32", overwrite: bool = False) -> None:
    """
    Count how many members are included in each group.

    The resulting per-group 1D data will have the specified ``dtype``. By default is ``int32`` which is a reasonable
    value for storing counts.

    __DAF__
    """
    assert is_dtype(dtype, INT_DTYPES), f"non-integer dtype: {dtype}"
    member_groups = data.get_vector("member;group")
    assert_data(has_dtype(member_groups, INT_DTYPES), "group indices data", member_groups, dtype=INT_DTYPES)
    member_groups = member_groups[member_groups >= 0]
    groups_count = data.axis_size("group")
    data.set_data1d(
        "group;members", np.bincount(member_groups, minlength=groups_count).astype(dtype), overwrite=overwrite
    )


@computation(
    required_inputs={
        "member;group": "The index of the group each member belongs to. If negative, it is not a part of any group.",
        "member;value": "The value associated with each individual member.",
    },
    assured_outputs={"group,value;members": "How many members have each value in each group."},
)
def count_group_values(
    data: DafWriter, *, dtype: DType = "int32", dense: bool = False, overwrite: bool = False
) -> None:
    """
    Count how many members of each group have each possible value.

    In ``daf``, axis entries always have string values. However, the per-member values 1D data need not contain strings,
    the only requirement is that converting them to strings will match the values axis entry names. This allows us to
    deal with data such as "age" which may take a few ``float`` values (e.g. would only be one of 6, 6.5, 7 days).

    The resulting per-group 2D data will have the specified ``dtype``. By default is ``int32`` which is a reasonable
    value for storing counts.

    By default, store the data in `.Sparse` format. If ``dense``, store it in `.Dense` format.

    __DAF__
    """
    assert is_dtype(dtype, NUM_DTYPES), f"non-numeric dtype: {dtype}"
    groups_count = data.axis_size("group")
    values_count = data.axis_size("value")

    member_groups = data.get_vector("member;group")
    assert_data(has_dtype(member_groups, INT_DTYPES), "group indices data", member_groups, dtype=INT_DTYPES)
    member_values = data.get_vector("member;value")
    grouped_mask = member_groups >= 0
    member_groups = member_groups[grouped_mask]
    member_values = member_values[grouped_mask].astype("str")

    value_entries = data.axis_entries("value")
    sorted_entry_indices = np.argsort(value_entries)
    sorted_value_entries = value_entries[sorted_entry_indices]
    sorted_member_value_indices = np.searchsorted(sorted_value_entries, member_values)
    member_value_indices = sorted_entry_indices[sorted_member_value_indices]

    data2d = sp.coo_matrix(
        (np.full(len(member_values), 1, dtype=dtype), (member_groups, member_value_indices)),
        shape=(groups_count, values_count),
    )
    if dense:
        data2d = data2d.toarray()
    assert has_dtype(data2d, dtype)

    data.set_data2d("group,value;members", data2d, overwrite=overwrite)


@computation(
    required_inputs={
        "member;group": "The index of the group each member belongs to. If negative, it is not a part of any group.",
        "group;value": "The value associated with each group.",
    },
    assured_outputs={"member;value": "The value associated with the group of each member."},
)
def assign_group_values(
    data: DafWriter, *, dtype: Optional[DType] = None, default: Any = None, overwrite: bool = False
) -> None:
    """
    Assign the per-group value to each members of the group.

    The resulting per-member 1D data will have the specified ``dtype``. By default is the same as the data type of of
    the group values.

    Members that are not a part of any group are given the ``default`` value. This is ``None`` by default, which is
    acceptable for floating point values (becomes a ``NaN``), but would fail for integer data.

    __DAF__
    """
    assert dtype is None or is_dtype(dtype), f"invalid dtype: {dtype}"
    member_groups = data.get_vector("member;group")
    assert_data(has_dtype(member_groups, INT_DTYPES), "group indices data", member_groups, dtype=INT_DTYPES)
    group_values = data.get_vector("group;value")
    dtype = dtype or dtype_of(group_values)
    assert dtype is not None

    if np.any(member_groups < 0):
        member_groups = be_vector(member_groups + 1)
        member_groups[member_groups < 0] = 0
        group_values = np.concatenate([[default], group_values], dtype=dtype)  # pylint: disable=unexpected-keyword-arg

    member_values = group_values.astype(dtype)[member_groups]
    data.set_data1d("member;value", member_values, overwrite=overwrite)


@computation(
    required_inputs={
        "member;group": "The index of the group each member belongs to. If negative, it is not a part of any group.",
    },
    assured_outputs={"group;": "A new axis with one entry per group."},
)
def create_group_axis(
    data: DafWriter,
    *,
    format: str,  # pylint: disable=redefined-builtin
    overwrite: bool = False,  # pylint: disable=unused-argument
) -> None:
    """
    Create a new ``group`` axis to hold per-group data.

    Since in ``daf`` axis entry names are always strings, we use the ``format`` to convert the group index to a string.
    This format should include ``%s`` somewhere in it.

    .. note::

        The created axis will be continuous, that is, group axis entries will still be created for all the group indices
        from zero to the maximal used group index.

    __DAF__
    """
    member_groups = data.get_vector("member;group")
    assert_data(has_dtype(member_groups, INT_DTYPES), "group indices data", member_groups, dtype=INT_DTYPES)
    groups_count = np.max(member_groups) + 1
    group_entries = [format % group for group in range(groups_count)]
    data.create_axis("group", group_entries)
