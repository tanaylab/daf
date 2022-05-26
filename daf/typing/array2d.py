"""
The types here describe a 2D ``numpy.ndarray``, which is one way to store 2D data in ``daf``.

.. note::

    We actively prevent using the deprecated data type ``numpy.matrix`` which is occasionally returned from some
    operations, ``isinstance`` of ``numpy.ndarray``, and behaves just differently enough from a normal ``numpy.ndarray``
    to cause subtle and difficult problems.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import NewType
from typing import Optional
from typing import Sequence
from typing import Union
from typing import overload

try:
    from typing import Annotated  # pylint: disable=unused-import
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import descriptions as _descriptions
from . import dtypes as _dtypes
from . import layouts as _layouts  # pylint: disable=cyclic-import
from . import matrices as _matrices  # pylint: disable=cyclic-import

# pylint: enable=duplicate-code


__all__ = [
    "Array2D",
    "is_array2d",
    "be_array2d",
    "Data2D",
    "as_array2d",
    "ArrayInRows",
    "is_array_in_rows",
    "be_array_in_rows",
    "ArrayInColumns",
    "is_array_in_columns",
    "be_array_in_columns",
]

#: 2-dimensional ``numpy`` array in :py:obj:`~daf.typing.layouts.ROW_MAJOR` layout.
ArrayInRows = NewType("ArrayInRows", "Annotated[np.ndarray, 'row_major']")


def is_array_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[ArrayInRows]:
    """
    Check whether some ``data`` is an :py:obj:`~ArrayInRows`, optionally only of some ``dtype``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    return is_array2d(data, dtype=dtype, layout=_layouts.ROW_MAJOR)


def be_array_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> ArrayInRows:
    """
    Assert that some ``data`` is a :py:obj:`~ArrayInRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_array_in_rows(data, dtype=dtype), "row-major numpy.ndarray", data, dtype)
    return data


#: 2-dimensional ``numpy`` array in column-major layout.
ArrayInColumns = NewType("ArrayInColumns", "Annotated[np.ndarray, 'column_major']")


def is_array_in_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[ArrayInColumns]:
    """
    Check whether some ``data`` is an :py:obj:`~ArrayInColumns`, optionally only of some ``dtype``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    return is_array2d(data, dtype=dtype, layout=_layouts.COLUMN_MAJOR)


def be_array_in_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> ArrayInColumns:
    """
    Assert that some ``data`` is a :py:obj:`~ArrayInColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_array_in_columns(data, dtype=dtype), "column-major numpy.ndarray", data, dtype)
    return data


#: 2-dimensional ``numpy`` in any-major layout.
Array2D = Union[ArrayInRows, ArrayInColumns]


def is_array2d(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None, layout: Optional[_layouts.AnyMajor] = None
) -> TypeGuard[Array2D]:
    """
    Check whether some ``data`` is an :py:obj:`~Array2D`, optionally only of some ``dtype``, optionally only of some
    ``layout``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    return (
        isinstance(data, np.ndarray)
        and data.ndim == 2
        and not isinstance(data, np.matrix)
        and _dtypes.is_dtype(str(data.dtype), dtype)
        and layout.is_layout_of(data)  # type: ignore
    )


def be_array2d(
    data: Any,
    *,
    dtype: Optional[Union[str, Collection[str]]] = None,
    layout: Optional[_layouts.AnyMajor] = None,
) -> Array2D:
    """
    Assert that some ``data`` is an :py:obj:`~Array2D` optionally only of some ``dtype``, optionally of only of a
    specific ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    _descriptions.assert_data(is_array2d(data, dtype=dtype, layout=layout), f"{layout.name} numpy.ndarray", data, dtype)
    return data


#: Anything that can be used to construct an :py:obj:`~Array2D`.
Data2D = Union[Sequence[Sequence[Any]], sp.spmatrix, pd.DataFrame, np.ndarray]


@overload
def as_array2d(data: _matrices.MatrixInRows, *, force_copy: bool = False) -> ArrayInRows:
    ...


@overload
def as_array2d(data: _matrices.MatrixInColumns, *, force_copy: bool = False) -> ArrayInColumns:
    ...


@overload
def as_array2d(
    data: Union[Sequence[Sequence[Any]], sp.spmatrix, pd.DataFrame, np.ndarray], *, force_copy: bool = False
) -> Array2D:
    ...


def as_array2d(data: Data2D, *, force_copy: bool = False) -> Array2D:
    """
    Access the internal 2D ``numpy`` array, if possible; otherwise, or if ``force_copy``, return a copy of the 2D data
    as a ``numpy`` array.

    Accepts as input many data types that aren't even a :py:obj:`~daf.typing.vectors.Matrix`, such as nested lists
    (basically anything that ``numpy.array`` can use to construct a 2D array).

    If the input is a ``pandas.DataFrame``, this will only work if all the data in the frame has the same type.

    This ensures that ``pandas`` strings (even if categorical) will be converted to proper ``numpy`` strings.
    """
    if isinstance(data, sp.spmatrix):
        return data.toarray()

    if isinstance(data, pd.DataFrame):
        data = data.values

    if force_copy or not isinstance(data, np.ndarray) or data.dtype == "category" or isinstance(data, np.matrix):
        data = np.array(data)

    _descriptions.assert_data(data.ndim == 2, "2D data", data, None)
    return data  # type: ignore
