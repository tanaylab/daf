"""
``scypy.sparse`` doesn't provide type annotations. We don't try to overcome this here, but we do want to allow for
saying "this is a sparse matrix in CSR format", at least as an option. To do this we need to use ``typing.Annotate``
which requires the base class to be known. As a workaround we define fake sparse matrix and use it instead. To shut
``mypy`` up we need to populate them with all the public interface of the real classes.

Hopefully `<https://github.com/python/mypy/issues/12757>`_ will be implemented and all of this mess would be able to be
deleted. Or, of course, maybe one day ``scipy.sparse`` will provide some form of type annotations, but that seems even
less likely.
"""

from __future__ import annotations

from abc import ABC
from typing import Any
from typing import Tuple

from . import array1d as _array1d

# pylint: disable=unused-argument,missing-function-docstring,no-self-use,invalid-name
# pylint: disable=too-many-public-methods,too-many-lines


__all__ = ["SparseMatrix"]


class SparseMatrix(ABC):
    """
    Fake class for ``mypy``.

    .. todo::

        If ``mypy`` implements `<https://github.com/python/mypy/issues/12757>`_ then we'd be able to get rid of
        ``SparseMatrix``.
    """

    data: _array1d.Array1D
    dtype: Any
    format: str
    has_canonical_format: bool
    has_sorted_indices: bool
    indices: _array1d.Array1D
    indptr: _array1d.Array1D
    maxprint: Any
    ndim: int
    nnz: int
    shape: Tuple[int, int]

    def arcsinh(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def arcsin(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def arctanh(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def arctan(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def argmax(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def argmin(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def asformat(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def asfptype(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def astype(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def ceil(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def check_format(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def conj(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def conjugate(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def copy(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def count_nonzero(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def deg2rad(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def diagonal(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def dot(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def eliminate_zeros(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def expm1(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def floor(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def getcol(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def getformat(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def getH(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def getmaxprint(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def getnnz(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def getrow(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def get_shape(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def log1p(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def maximum(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def max(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def minimum(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def min(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def multiply(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def nonzero(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def power(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def prune(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rad2deg(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def reshape(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def resize(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rint(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def setdiag(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def set_shape(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sign(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sinh(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sin(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sorted_indices(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sort_indices(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sqrt(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sum_duplicates(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tanh(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tan(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def toarray(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tobsr(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tocoo(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tocsc(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tocsr(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def todense(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def todia(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def todok(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tolil(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def trace(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def transpose(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def trunc(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __abs__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __add__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __bool__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __div__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __getattr__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __getitem__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __iadd__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __idiv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __imul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __isub__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __iter__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __itruediv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __len__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __matmul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __mul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __neg__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __nonzero__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __pow__(self, *args: Any, **kwargs: Any) -> Any:  # pylint: disable=unexpected-special-method-signature
        ...

    def __radd__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rdiv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rmatmul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rmul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __round__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rsub__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rtruediv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __setitem__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __str__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __sub__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __subclasshook__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __truediv__(self, *args: Any, **kwargs: Any) -> Any:
        ...
