"""
The types here describe 2D data inside a ``pandas.DataFrame``, where all the elements have the same data type, which can
be fetched from ``daf`` storage.

Logically and operationally this is a distinct data type from a generic data frame where each column has a different
data type (that is, a "real" data frame). Since ``pandas`` does not make this distinction, even if/when it provides
``mypy`` annotations, we'd still need to set up the types here (similar to the problem with ``numpy.ndarray``).

In theory it should have been possible to store sparse data inside a ``pandas.DataFrame``, but in practice this fails in
various ways, so **don't**. When fetching data from ``daf``, frames will alway contain dense (``numpy.ndarray`` 2D)
data.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import NewType
from typing import Optional
from typing import Tuple
from typing import Union

try:
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

import pandas as pd  # type: ignore

from . import dense as _dense
from . import descriptions as _descriptions
from . import dtypes as _dtypes
from . import fake_pandas as _fake_pandas  # pylint: disable=unused-import
from . import layouts as _layouts

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "Frame",
    "is_frame",
    "be_frame",
    "FrameInRows",
    "is_frame_in_rows",
    "be_frame_in_rows",
    "FrameInColumns",
    "is_frame_in_columns",
    "be_frame_in_columns",
]

#: A 2D ``pandas.DataFrame`` with homogeneous data elements, in row-major layout.
FrameInRows = NewType("FrameInRows", _fake_pandas.DataFrame)


def is_frame_in_rows(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> TypeGuard[FrameInRows]:
    """
    Check whether some ``data`` is a `.FrameInRows`, optionally only of some ``dtype``, optionally only of some
    ``shape``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return is_frame(data, dtype=dtype, shape=shape, layout=_layouts.ROW_MAJOR)


def be_frame_in_rows(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> FrameInRows:
    """
    Assert that some ``data`` is a `.FrameInRows`, optionally only of some ``dtype``, optionally only of some ``shape``,
    and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(
        is_frame_in_rows(data, dtype=dtype, shape=shape), "row-major pandas.DataFrame", data, dtype=dtype, shape=shape
    )
    return data


#: A 2D ``pandas.DataFrame`` with homogeneous data elements, in column-major layout.
FrameInColumns = NewType("FrameInColumns", _fake_pandas.DataFrame)


def is_frame_in_columns(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> TypeGuard[FrameInColumns]:
    """
    Check whether some ``data`` is a `.FrameInColumns`, optionally only of some ``dtype``, optionally only of some
    ``shape``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return is_frame(data, dtype=dtype, shape=shape, layout=_layouts.COLUMN_MAJOR)


def be_frame_in_columns(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> FrameInColumns:
    """
    Assert that some ``data`` is a `.FrameInColumns`, optionally only of some ``dtype``, optionally only of some
    ``shape``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(
        is_frame_in_columns(data, dtype=dtype, shape=shape),
        "column-major pandas.DataFrame",
        data,
        dtype=dtype,
        shape=shape,
    )
    return data


#: A 2D ``pandas.DataFrame`` with homogeneous data elements, in any-major layout.
Frame = Union[FrameInRows, FrameInColumns]


def is_frame(
    data: Any,
    *,
    dtype: Optional[_dtypes.DTypes] = None,
    shape: Optional[Tuple[int, int]] = None,
    layout: Optional[_layouts.AnyMajor] = None,
) -> TypeGuard[Frame]:
    """
    Check whether some ``data`` is a `.Frame`, optionally only of some ``dtype``, optionally only of some ``shape``,
    optionally only of some ``layout``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return (
        isinstance(data, pd.DataFrame)
        and len(set(data.dtypes)) == 1
        and _dense.is_dense(data.values, dtype=dtype, shape=shape, layout=layout)
    )


def be_frame(
    data: Any,
    *,
    dtype: Optional[_dtypes.DTypes] = None,
    shape: Optional[Tuple[int, int]] = None,
    layout: Optional[_layouts.AnyMajor] = None,
) -> Frame:
    """
    Assert that some ``data`` is a `.Frame` optionally only of some ``dtype``, optionally only of some ``shape``,
    optionally only of some ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    _descriptions.assert_data(
        is_frame(data, dtype=dtype, shape=shape, layout=layout),
        f"{layout.name} pandas.DataFrame",
        data,
        dtype=dtype,
        shape=shape,
    )
    return data
