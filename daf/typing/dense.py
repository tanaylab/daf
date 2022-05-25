"""
The types here describe dense 2D data. Currently there are
only two:

* :py:const:`Array2D` is a two-dimensional ``numpy.ndarray``.
* :py:const:`Table` is a ``pandas.DataFrame`` which contains an ``Array2D`` of a single element type, with indices of
  names for the rows and columns.

The :py:const:`Dense` type annotations is their union, that is, allows for "any" dense 2D. It isn't very useful but is
provided for completeness.

.. note::

    The :py:const:`Dense` type should be used in computations with **great care**, as some operations are subtly
    different for ``numpy`` 2D arrays and ``pandas`` data frames. It is typically better to use one of the concrete
    types instead.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Optional
from typing import Union

try:
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

from . import array2d as _array2d
from . import descriptions as _descriptions
from . import layouts as _layouts  # pylint: disable=cyclic-import
from . import tables as _tables

# pylint: enable=duplicate-code

__all__ = [
    "Dense",
    "is_dense",
    "be_dense",
    "DenseInRows",
    "is_dense_in_rows",
    "be_dense_in_rows",
    "DenseInColumns",
    "is_dense_in_columns",
    "be_dense_in_columns",
]

#: Any dense 2D data in row-major layout.
DenseInRows = Union[_array2d.ArrayInRows, _tables.TableInRows]


def is_dense_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[DenseInRows]:
    """
    Assert that some ``data`` is an :py:const:`DenseInRows`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    return _array2d.is_array_in_rows(data, dtype=dtype) or _tables.is_table_in_rows(data, dtype=dtype)


def be_dense_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> DenseInRows:
    """
    Assert that some ``data`` is a :py:const:`DenseInRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_dense_in_rows(data, dtype=dtype), "row-major dense matrix", data, dtype)
    return data


#: Any dense 2D data in column-major layout.
DenseInColumns = Union[_array2d.ArrayInColumns, _tables.TableInColumns]


def is_dense_in_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[DenseInColumns]:
    """
    Assert that some ``data`` is an :py:const:`DenseInColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    return _array2d.is_array_in_columns(data, dtype=dtype) or _tables.is_table_in_columns(data, dtype=dtype)


def be_dense_in_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> DenseInColumns:
    """
    Assert that some ``data`` is a :py:const:`DenseInColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_dense_in_columns(data, dtype=dtype), "column-major dense matrix", data, dtype)
    return data


#: Any dense 2D data without names.
Dense = Union[_array2d.Array2D, _tables.Table]


def is_dense(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None, layout: Optional[_layouts.AnyMajor] = None
) -> TypeGuard[Dense]:
    """
    Assert that some ``data`` is an :py:const:`Dense`, optionally only of some ``dtype``, optionally only of some
    ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    return _array2d.is_array2d(data, dtype=dtype, layout=layout) or _tables.is_table(data, dtype=dtype, layout=layout)


def be_dense(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None, layout: Optional[_layouts.AnyMajor] = None
) -> Dense:
    """
    Assert that some ``data`` is a :py:const:`Dense`, optionally only of some ``dtype``, optionally only of some
    ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    _descriptions.assert_data(is_dense(data, dtype=dtype, layout=layout), f"{layout.name} dense matrix", data, dtype)
    return data
