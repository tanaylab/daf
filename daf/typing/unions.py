"""
Union types used in the rest of the code.

These types capture important concepts that allow generic ``daf`` functions to clearly express their intent. Due to to
limitations of ``mypy``, the existing type annotations in ``numpy`` and the lack of type annotations in ``scipy.sparse``
and ``pandas``, the ``Known...`` types are too permissive, but the ``is_known...``/``be_known...`` functions do the
right thing.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

try:
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import dense as _dense
from . import descriptions as _descriptions
from . import dtypes as _dtypes
from . import fake_pandas as _fake_pandas
from . import fake_sparse as _fake_sparse  # pylint: disable=unused-import
from . import frames as _frames
from . import layouts as _layouts
from . import series as _series
from . import sparse as _sparse
from . import vectors as _vectors

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "AnyData",
    "Known",
    "is_known",
    "be_known",
    "Known1D",
    "is_known1d",
    "be_known1d",
    "Known2D",
    "is_known2d",
    "be_known2d",
    "Proper",
    "is_proper",
    "be_proper",
    "Proper1D",
    "is_proper1d",
    "be_proper1d",
    "Proper2D",
    "is_proper2d",
    "be_proper2d",
    "ProperInRows",
    "is_proper_in_rows",
    "be_proper_in_rows",
    "ProperInColumns",
    "is_proper_in_columns",
    "be_proper_in_columns",
]


#: Any "proper" 1D data type that ``daf`` fully supports.
Proper1D = Union["_vectors.Vector", _fake_pandas.Series]


def is_proper1d(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, size: Optional[int] = None
) -> TypeGuard[Proper1D]:
    """
    Assert that some ``data`` is `.Proper1D`,, optionally only of some ``dtype``, optionally only of some ``size``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return _vectors.is_vector(data, dtype=dtype, size=size) or _series.is_series(data, dtype=dtype, size=size)


def be_proper1d(data: Any, *, dtype: Optional[_dtypes.DTypes] = None, size: Optional[int] = None) -> Proper1D:
    """
    Assert that some ``data`` is `.Proper1D`, optionally only of some ``dtype``, optionally only of some ``size``, and
    return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_proper1d(data, dtype=dtype, size=size), "proper 1D data", data, dtype=dtype, size=size)
    return data


#: Any "proper" `.ROW_MAJOR` 2D data type that ``daf`` fully supports.
ProperInRows = Union["_dense.DenseInRows", _sparse.SparseInRows, _frames.FrameInRows]


def is_proper_in_rows(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> TypeGuard[ProperInRows]:
    """
    Assert that some ``data`` is `.ProperInRows`.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return (
        _dense.is_dense_in_rows(data, dtype=dtype, shape=shape)
        or _sparse.is_sparse_in_rows(data, dtype=dtype, shape=shape)
        or _frames.is_frame_in_rows(data, dtype=dtype, shape=shape)
    )


def be_proper_in_rows(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> ProperInRows:
    """
    Assert that some ``data`` is `.ProperInRows`, optionally only of some ``dtype``, optionally only of some ``shape``,
    and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(
        is_proper_in_rows(data, dtype=dtype, shape=shape),
        "proper data",
        data,
        dtype=dtype,
        shape=shape,
        layout=_layouts.ROW_MAJOR,
    )
    return data


#: Any "proper" `.COLUMN_MAJOR` 2D data type that ``daf`` fully supports.
ProperInColumns = Union["_dense.DenseInColumns", _sparse.SparseInColumns, _frames.FrameInColumns]


def is_proper_in_columns(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> TypeGuard[ProperInColumns]:
    """
    Assert that some ``data`` is `.ProperInColumns`, optionally only of some ``dtype``, optionally only of some
    ``shape``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return (
        _dense.is_dense_in_columns(data, dtype=dtype, shape=shape)
        or _sparse.is_sparse_in_columns(data, dtype=dtype, shape=shape)
        or _frames.is_frame_in_columns(data, dtype=dtype, shape=shape)
    )


def be_proper_in_columns(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, shape: Optional[Tuple[int, int]] = None
) -> ProperInColumns:
    """
    Assert that some ``data`` is `.ProperInColumns`, optionally only of some ``dtype``, optionally only of some
    ``shape``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(
        is_proper_in_columns(data, dtype=dtype, shape=shape),
        "proper data",
        data,
        dtype=dtype,
        shape=shape,
        layout=_layouts.COLUMN_MAJOR,
    )
    return data


#: Any "proper" 2D data type that ``daf`` fully supports.
Proper2D = Union[ProperInRows, ProperInColumns]


def is_proper2d(
    data: Any,
    *,
    dtype: Optional[_dtypes.DTypes] = None,
    shape: Optional[Tuple[int, int]] = None,
    layout: Optional[_layouts.AnyMajor] = None,
) -> TypeGuard[Proper2D]:
    """
    Assert that some ``data`` is `.Proper2D`, optionally only of some ``dtype``, optionally only of some ``shape``,
    optionally optionally only of a specific ``layout``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return (
        _dense.is_dense(data, dtype=dtype, shape=shape, layout=layout)
        or _sparse.is_sparse(data, dtype=dtype, shape=shape, layout=layout)
        or _frames.is_frame(data, dtype=dtype, shape=shape, layout=layout)
    )


def be_proper2d(
    data: Any,
    *,
    dtype: Optional[_dtypes.DTypes] = None,
    shape: Optional[Tuple[int, int]] = None,
    layout: Optional[_layouts.AnyMajor] = None,
) -> Proper2D:
    """
    Assert that some ``data`` is `.Proper2D`, optionally only of some ``dtype``, optionally only of some ``shape``,
    optionally optionally only of a specific ``layout``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(
        is_proper2d(data, dtype=dtype, shape=shape, layout=layout), "proper 2D data", data, dtype=dtype, shape=shape
    )
    return data


#: Any "proper" 1D/2D data type that ``daf`` fully supports.
#:
#: Such data need not be `.is_optimal`.
Proper = Union[Proper1D, Proper2D]


def is_proper(data: Any, *, dtype: Optional[_dtypes.DTypes] = None) -> TypeGuard[Proper]:
    """
    Assert that some ``data`` is `.Proper`, optionally only of some ``dtype``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return is_proper1d(data, dtype=dtype) or is_proper2d(data, dtype=dtype)


def be_proper(data: Any, *, dtype: Optional[_dtypes.DTypes] = None) -> Proper:
    """
    Assert that some ``data`` is `.Proper`, optionally only of some ``dtype``, and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_proper(data, dtype=dtype), "proper 1D/2D data", data, dtype=dtype)
    return data


#: Any 1D data type that ``daf`` knows about.
Known1D = Union[np.ndarray, _fake_pandas.Series]


def is_known1d(data: Any) -> TypeGuard[Known1D]:
    """
    Assert that some ``data`` is `.Known1D` (and actually 1D data).
    """
    return data.ndim == 1 if isinstance(data, np.ndarray) else isinstance(data, pd.Series)


def be_known1d(data: Any) -> Known1D:
    """
    Assert that some ``data`` is `.Known1D`, and return it as such for ``mypy``.
    """
    _descriptions.assert_data(is_known1d(data), "known 1D data", data)
    return data


#: Any 2D data type that ``daf`` knows about.
#:
#: Due to ``numpy.ndarray`` type annotation limitations, we can't use ``mypy`` to ensure this is actually 2D data.
Known2D = Union[np.ndarray, "_fake_sparse.spmatrix", _fake_pandas.DataFrame]


def is_known2d(data: Any) -> TypeGuard[Known2D]:
    """
    Assert that some ``data`` is `.Known2D` (and actually 2D data).
    """
    return data.ndim == 2 if isinstance(data, np.ndarray) else isinstance(data, (sp.spmatrix, pd.DataFrame))


def be_known2d(data: Any) -> Proper:
    """
    Assert that some ``data`` is `.Known2D` (and actually 2D data), and return it as such for ``mypy``.
    """
    _descriptions.assert_data(is_known2d(data), "known 2D data", data)
    return data


#: Any 1D/2D data type that ``daf`` knows about.
#:
#: Due to ``numpy.ndarray`` type annotation limitations, we can't use ``mypy`` to ensure this is actually 1D/2D data.
Known = Union[np.ndarray, "_fake_sparse.spmatrix", _fake_pandas.Series, _fake_pandas.DataFrame]


def is_known(data: Any) -> TypeGuard[Known]:
    """
    Assert that some ``data`` is `.Known` (and actually 1D/2D data), and return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return is_known1d(data) or is_known2d(data)


def be_known(data: Any) -> Known:
    """
    Assert that some ``data`` is `.Known` (and actually 1D/2D data), and return it as such for ``mypy``.
    """
    _descriptions.assert_data(is_known(data), "known 1D/2D data", data)
    return data


#: "Any" data that can be used to construct "proper" 1D/2D data.
#:
#: We don't distinguish between 1D and 2D data here, because one can use 2D-ish data with an axis of size 1 to construct
#: 1D data (that is, a matrix with a single row can be used to construct a vector).
#:
#: There are probably other alternatives that should be listed here, but this covers most usages.
AnyData = Union[Sequence[Any], np.ndarray, "_fake_sparse.spmatrix", _fake_pandas.Series, _fake_pandas.DataFrame]
