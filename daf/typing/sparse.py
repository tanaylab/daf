"""
The types here describe a sparse compressed ``scipy.sparse.csr_matrix`` and ``scipy.sparse.csc_matrix`` data, which is
two ways to store 2D data in ``daf``.

In theory it should have been possible to store sparse data inside a ``pandas.DataFrame``, but in practice this fails in
various ways, so ``daf`` only stores such frames if they contain dense (that is, :py:obj:`~daf.typing.array2d.Array2D`)
data.

.. note::

    Other sparse formats (e.g. ``scipy.sparse.coo_matrix``) can't be stored in ``daf``. This allows ``daf`` to require
    that all stored data is in either :py:obj:`~daf.typing.layouts.ROW_MAJOR` or
    :py:obj:`~daf.typing.layouts.COLUMN_MAJOR` layout, which greatly simplifies the code accessing the data and also
    makes it easier to store the data in a consistent way.
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

import scipy.sparse as sp  # type: ignore

from . import descriptions as _descriptions
from . import dtypes as _dtypes
from . import fake_sparse as _fake_sparse  # pylint: disable=unused-import
from . import layouts as _layouts  # pylint: disable=cyclic-import

# pylint: enable=duplicate-code


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

#: 2-dimensional ``scipy.sparse`` matrix in CSR layout.
SparseInRows = NewType("SparseInRows", "Annotated[_fake_sparse.SparseMatrix, 'csr']")


def is_sparse_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[SparseInRows]:
    """
    Check whether some ``data`` is a :py:obj:`~SparseInRows`, optionally only of some ``dtype``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    return isinstance(data, sp.csr_matrix) and _dtypes.is_dtype(str(data.data.dtype), dtype)


def be_sparse_in_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> SparseInRows:
    """
    Assert that some ``data`` is a :py:obj:`~SparseInRows`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_sparse_in_rows(data, dtype=dtype), "scipy.sparse.csr_matrix", data, dtype)
    return data


#: 2-dimensional ``scipy.sparse`` matrix in CSC layout.
SparseInColumns = NewType("SparseInColumns", "Annotated[_fake_sparse.SparseMatrix, 'csc']")


def is_sparse_in_columns(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None
) -> TypeGuard[SparseInColumns]:
    """
    Check whether some ``data`` is a :py:obj:`~SparseInColumns`, optionally only of some ``dtype``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    return isinstance(data, sp.csc_matrix) and _dtypes.is_dtype(str(data.data.dtype), dtype)


def be_sparse_in_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> SparseInColumns:
    """
    Assert that some ``data`` is a :py:obj:`~SparseInColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    _descriptions.assert_data(is_sparse_in_columns(data, dtype=dtype), "scipy.sparse.csc_matrix", data, dtype)
    return data


#: 2-dimensional ``scipy.sparse`` matrix in compressed layout.
Sparse = Union[SparseInRows, SparseInColumns]


def is_sparse(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None, layout: Optional[_layouts.AnyMajor] = None
) -> TypeGuard[Sparse]:
    """
    Check whether some ``data`` is a :py:obj:`~Sparse`, optionally only of some ``dtype``, optionally only of some
    ``layout``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    return isinstance(data, layout.sparse_class) and _dtypes.is_dtype(str(data.data.dtype), dtype)


def be_sparse(
    data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None, layout: Optional[_layouts.AnyMajor] = None
) -> Sparse:
    """
    Assert that some ``data`` is a :py:obj:`~Sparse` optionally only of some ``dtype``, optionally of some ``layout``,
    and return it as such for ``mypy``.

    By default, checks that the data type is one of :py:obj:`~daf.typing.ALL_DTYPES`.
    """
    layout = layout or _layouts._ANY_MAJOR  # pylint: disable=protected-access
    _descriptions.assert_data(is_sparse(data, dtype=dtype, layout=layout), layout.sparse_class_name, data, dtype)
    return data
