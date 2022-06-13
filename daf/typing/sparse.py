"""
The types here describe a sparse compressed ``scipy.sparse.csr_matrix`` and ``scipy.sparse.csc_matrix`` data, which
can be fetched from ``daf``.

In theory it should have been possible to store sparse data inside a ``pandas.DataFrame``, but in practice this fails in
various ways, so **don't**. When fetching data from ``daf``, frames will alway contain dense (``numpy.ndarray`` 2D)
data.

.. note::

    Other sparse formats (e.g. ``scipy.sparse.coo_matrix``) can't be fetched from ``daf``. This allows ``daf`` to
    require that all stored data is in either `.ROW_MAJOR` or `.COLUMN_MAJOR` layout, which greatly simplifies the code
    accessing the data, and also makes it easier to store the data in a consistent way.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import NewType
from typing import Optional
from typing import Union

try:
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

import scipy.sparse as sp  # type: ignore

from . import descriptions as _descriptions
from . import dtypes as _dtypes
from . import fake_sparse as _fake_sparse  # pylint: disable=unused-import
from . import layouts as _layouts

# pylint: enable=duplicate-code,cyclic-import


__all__ = [
    "Sparse",
    "is_sparse",
    "be_sparse",
    "SparseInRows",
    "is_sparse_in_rows",
    "be_sparse_in_rows",
    "SparseInColumns",
    "is_sparse_in_columns",
    "be_sparse_in_columns",
]

#: 2D ``scipy.sparse.spmatrix`` in CSR layout (that is, ``scipy.sparse.csr_matrix``).
SparseInRows = NewType("SparseInRows", "_fake_sparse.cs_matrix")


def is_sparse_in_rows(data: Any, *, dtype: Optional[_dtypes.DTypes] = None) -> TypeGuard[SparseInRows]:
    """
    Check whether some ``data`` is a `.SparseInRows`, optionally only of some ``dtype``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return isinstance(data, sp.csr_matrix) and _dtypes.has_dtype(data, dtype)


def be_sparse_in_rows(data: Any, *, dtype: Optional[_dtypes.DTypes] = None) -> SparseInRows:
    """
    Assert that some ``data`` is a `.SparseInRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_sparse_in_rows(data, dtype=dtype), "scipy.sparse.csr_matrix", data, dtype=dtype)
    return data


#: 2D ``scipy.sparse.spmatrix`` in CSC layout (that is, ``scipy.sparse.csc_matrix``).
SparseInColumns = NewType("SparseInColumns", "_fake_sparse.cs_matrix")


def is_sparse_in_columns(data: Any, *, dtype: Optional[_dtypes.DTypes] = None) -> TypeGuard[SparseInColumns]:
    """
    Check whether some ``data`` is a `.SparseInColumns`, optionally only of some ``dtype``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    return isinstance(data, sp.csc_matrix) and _dtypes.has_dtype(data, dtype)


def be_sparse_in_columns(data: Any, *, dtype: Optional[_dtypes.DTypes] = None) -> SparseInColumns:
    """
    Assert that some ``data`` is a `.SparseInColumns`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_sparse_in_columns(data, dtype=dtype), "scipy.sparse.csc_matrix", data, dtype=dtype)
    return data


#: 2D ``scipy.sparse.spmatrix`` in compressed layout.
Sparse = Union[SparseInRows, SparseInColumns]


def is_sparse(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, layout: Optional[_layouts.AnyMajor] = None
) -> TypeGuard[Sparse]:
    """
    Check whether some ``data`` is a `.Sparse`, optionally only of some ``dtype``, optionally only of some ``layout``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    return isinstance(data, layout.sparse_class) and _dtypes.has_dtype(data, dtype)


def be_sparse(
    data: Any, *, dtype: Optional[_dtypes.DTypes] = None, layout: Optional[_layouts.AnyMajor] = None
) -> Sparse:
    """
    Assert that some ``data`` is a `.Sparse` optionally only of some ``dtype``, optionally of some ``layout``, and
    return it as such for ``mypy``.

    By default, checks that the data type is one of `.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    _descriptions.assert_data(is_sparse(data, dtype=dtype, layout=layout), layout.sparse_class_name, data, dtype=dtype)
    return data
