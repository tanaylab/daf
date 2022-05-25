"""
The types here describe 2D data where all the elements have the same data type, inside a ``pandas.DataFrame``.

Logically and operationally this is a distinct data type from a generic data frame where each column has a different
data type (that is, a :py:const:`daf.typing.frames.Frame`), hence we call this type :py:const:`Table` (which admittedly
isn't a great name). Of course, ``pandas`` does not make this distinction, so even if/when it provides ``mypy``
annotations, we'd still need to set up the types here (similar to the problem with ``numpy.ndarray``).

In theory it should have been possible to store sparse data inside a ``pandas.DataFrame``, but in practice this fails in
svarious ways, so ``daf`` only stores such frames if they contain dense (that is,
:py:const:`daf.typing.array2d.Array2D`) data.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import NewType
from typing import Optional
from typing import Union

try:
    from typing import Annotated  # pylint: disable=unused-import
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

import numpy as np
import pandas as pd  # type: ignore

from . import array2d as _array2d
from . import descriptions as _descriptions
from . import fake_pandas as _fake_pandas  # pylint: disable=unused-import
from . import layouts as _layouts  # pylint: disable=cyclic-import

# pylint: enable=duplicate-code

__all__ = [
    "Table",
    "is_table",
    "be_table",
    "TableInRows",
    "is_table_in_rows",
    "be_table_in_rows",
    "TableInColumns",
    "is_table_in_columns",
    "be_table_in_columns",
]

#: 2-dimensional ``pandas`` frame in row-major layout.
TableInRows = NewType("TableInRows", "Annotated[_fake_pandas.PandasFrame, 'row_major']")


def is_table_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[TableInRows]:
    """
    Check whether some ``data`` is a :py:const:`TableInRows`, optionally only of some ``dtype``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    return is_table(data, dtype=dtype, layout=_layouts.ROW_MAJOR)


def be_table_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TableInRows:
    """
    Assert that some ``data`` is a :py:const:`TableInRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_table_in_rows(data, dtype=dtype), "row-major pandas Table", data, dtype)
    return data


#: 2-dimensional ``pandas`` frame in column-major layout.
TableInColumns = NewType("TableInColumns", "Annotated[_fake_pandas.PandasFrame, 'column_major']")


def is_table_in_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[TableInColumns]:
    """
    Check whether some ``data`` is a :py:const:`TableInColumns`, optionally only of some ``dtype``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    return is_table(data, dtype=dtype, layout=_layouts.COLUMN_MAJOR)


def be_table_in_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TableInColumns:
    """
    Assert that some ``data`` is a :py:const:`TableInColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_table_in_columns(data, dtype=dtype), "column-major pandas Table", data, dtype)
    return data


#: 2-dimensional ``pandas`` frame in any-major layout.
Table = Union[TableInRows, TableInColumns]


def is_table(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None, layout: Optional[_layouts.AnyMajor] = None
) -> TypeGuard[Table]:
    """
    Check whether some ``data`` is a :py:const:`Table`, optionally only of some ``dtype``, optionally only of some
    ``layout``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    return (
        isinstance(data, pd.DataFrame)
        and len(np.unique(data.dtypes)) == 1
        and _array2d.is_array2d(data.values, dtype=dtype, layout=layout)
    )


def be_table(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None, layout: Optional[_layouts.AnyMajor] = None
) -> Table:
    """
    Assert that some ``data`` is a :py:const:`Table` optionally only of some ``dtype``, optionally only of some
    ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of :py:const:`daf.typing.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    _descriptions.assert_data(is_table(data, dtype=dtype, layout=layout), f"{layout.name} pandas Table", data, dtype)
    return data
