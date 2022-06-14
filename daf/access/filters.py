"""
Filters whose results can be cached in ``daf`` data sets.

There are some operations that apply to 1D and 2D data, that are commonly used, so it makes sense to cache their results
instead of computing them every time from scratch. A trivial example is the sum of the values in each row of a matrix,
which is used for computing averages, normalizing the values into fractions, etc.

To allow for efficiently caching such computations, we place severe restrictions on them. Specifically we support two
kind of operations:

* `.ElementWise` operations transform 1D/2D data values but maintain its shape (e.g. absolute value).

* `.Reduction` operations remove one dimension of the data (e.g. sum), so that vectors become scalars
  and matrices become vectors. We always apply the reduction to each row of the input (`.ROW_MAJOR`) function.

In both cases, the operation must be "pure", that is, the result must depend only on the input data and the operation
parameters (if any).
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.sparse as sp  # type: ignore

from ..typing import INT_DTYPES
from ..typing import DenseInRows
from ..typing import SparseInRows
from ..typing import Vector
from ..typing import as_vector
from ..typing import be_dense_in_rows
from ..typing import be_vector
from ..typing import is_dtype

# pylint: enable=duplicate-code,cyclic-import


__all__ = [
    "element_wise",
    "Abs",
    "Floor",
    "Round",
    "Ceil",
    "Clip",
    "Log",
    "Densify",
    "Sparsify",
    "Significant",
    "reduction",
    "Sum",
    "Min",
    "Max",
    "Mean",
    "Var",
    "Std",
    "parse_float_parameter",
    "parse_int_parameter",
    "parse_bool_parameter",
    "float_dtype_for",
    "Filter",
    "ElementWise",
    "Reduction",
]


class Filter(ABC):  # pylint: disable=too-few-public-methods
    """
    Common functionality for all filters.

    When a user writes ``...|Operation,parameter=value,...``, this is converted to creating an instance of a sub-class
    of this base class, passing it all the parameter values as keyword arguments, with the addition of a
    ``_input_dtype`` keyword parameter.

    The sub-class then calls ``super().__init__(...)`` to initialize itself as `.Filter`. It can decide on the value of
    the following parameters based on the ``_input_dtype`` as well as user-provided parameters, if any:
    """

    # The registered filters.
    _registry: Dict[str, Tuple[str, type]] = {}

    def __init__(self, *, dtype: Optional[str] = None, canonical: Optional[str] = None, nop: bool = False) -> None:
        #: Normally the output ``dtype`` would be the same as the ``_input_dtype``, but sometimes it needs to be
        #: different (e.g. `.Mean` will generate floating point results for integer data). It is expected that all
        #: sub-classes will allow the user to specify an optional explicit ``dtype`` parameter to override this.
        self.dtype = dtype

        if canonical is None:
            canonical = self.__class__.__name__
        else:
            canonical = self.__class__.__name__ + "," + canonical

        if dtype is not None:
            canonical += ",dtype=" + dtype

        #: This is used for caching; that is, if the canonical form of the operation is the same for two instances, the
        #: operation is assumed to be identical. To maximize its effectiveness, the canonical form should make all the
        #: parameters explicit, in the same order, with standard formatting. As a convenience, the ``dtype`` is
        #: automatically added to the ``canonical`` parameters string.
        self.canonical = canonical

        #: If ``nop``, the sub-class indicates the whole operation is a no-op. In this case we just directly use the
        #: input data as the output data, unless this is an `.ElementWise` operation that ``densifies`` or
        #: ``sparsifies`` the data.
        self.nop = nop

    @staticmethod
    def register(kind: str, klass: type) -> None:
        """
        Register a ``klass`` as a ``kind`` or filter.

        The ``kind`` should be one of ``element_wise`` or ``reduction``.
        """
        assert kind in ("element_wise", "reduction")
        conflict = Filter._registry.get(klass.__name__)
        assert conflict is None, (
            f"ambiguous filter: {klass.__name__} "
            f"could be either the {conflict[0]} {conflict[1].__module__}.{conflict[1].__qualname__} "
            f"or the {kind} {klass.__module__}.{klass.__qualname__}"
        )
        Filter._registry[klass.__name__] = (kind, klass)


class ElementWise(Filter):
    """
    Describe an element-wise operation (e.g., absolute value).

    Technically, name "element-wise" is inaccurate, since in principle each output element may depend on all the input
    elements. The actual requirement is that the input and output of element-wise operations have the same shape. They
    need not produce the same data type, and the output may be dense even if the input is sparse.

    The concrete sub-class needs to specify whether this operation ``densifies`` sparse matrices or ``sparsifies`` dense
    matrices. This, plus the type of the input, will determine which of the member functions will be called to actually
    do the computation.
    """

    def __init__(
        self,
        *,
        densifies: bool,
        sparsifies: bool,
        dtype: Optional[str] = None,
        canonical: Optional[str] = None,
        nop: bool = False,
    ) -> None:
        super().__init__(dtype=dtype, canonical=canonical, nop=nop)

        #: Whether the result of the operation on a `.Sparse` input matrix will be `.Dense`. This determines whether
        #: `.ElementWise.sparse_to_sparse` or `.ElementWise.sparse_to_dense` will be called when the input is `.Sparse`.
        self.densifies = densifies

        #: Whether the result of the operation on a `.Dense` input matrix will be `.Sparse`. This determines whether
        #: `.ElementWise.dense_to_dense` or `.ElementWise.dense_to_sparse` will be called when the input is `.Dense`.
        self.sparsifies = sparsifies

    def vector_to_vector(self, input_vector: Vector) -> Vector:
        """
        Compute the operation on an ``input`` vector into a new output vector.
        """
        output_vector = np.empty(input_vector.size, dtype=self.dtype)
        self._ndarray_to_ndarray(input_vector, output_vector)
        return be_vector(output_vector)

    def dense_to_dense(self, input_dense: DenseInRows, output_dense: DenseInRows) -> None:
        """
        Compute the operation on a dense `.ROW_MAJOR` ``input`` matrix into a dense `.ROW_MAJOR` output matrix.

        This allows us to pre-allocate the output matrix using `.StorageWriter.create_dense_in_rows`, allowing us to
        efficiently process large data without consuming excessive amount of RAM.
        """
        self._ndarray_to_ndarray(input_dense, output_dense)

    def dense_to_sparse(self, input_dense: DenseInRows) -> SparseInRows:
        """
        Compute the operation on a dense `.ROW_MAJOR` ``input`` matrix into a new sparse `.ROW_MAJOR` output matrix.
        """
        output_dense = np.empty(input_dense.shape, dtype=self.dtype)
        self._ndarray_to_ndarray(input_dense, output_dense)
        return sp.csr_matrix(output_dense)

    def sparse_to_dense(self, input_sparse: SparseInRows, output_dense: DenseInRows) -> None:
        """
        Compute the operation on a sparse `.ROW_MAJOR` ``input`` matrix into a dense `.ROW_MAJOR` output matrix.

        This allows us to pre-allocate the output matrix using `.StorageWriter.create_dense_in_rows`, allowing us to
        efficiently process large data without consuming excessive amount of RAM.
        """
        return self._ndarray_to_ndarray(input_sparse.toarray(order="C"), output_dense)

    def sparse_to_sparse(self, input_sparse: SparseInRows) -> SparseInRows:
        """
        Compute the operation on a sparse `.ROW_MAJOR` ``input`` matrix into a new sparse `.ROW_MAJOR` output matrix.

        A common idiom is to reuse the ``indices`` and ``indptr`` arrays of the input and only create a new ``data``
        array.
        """
        output_data = np.empty(input_sparse.data.size, dtype=self.dtype)
        self._ndarray_to_ndarray(input_sparse.data, output_data)
        return sp.csr_matrix((output_data, input_sparse.indices, input_sparse.indptr))

    @abstractmethod
    def _ndarray_to_ndarray(self, input_ndarray: np.ndarray, output_ndarray: np.ndarray) -> None:
        """
        Do element-wise computation from the ``input_ndarray`` directly into the ``output_ndarray``.
        """


def element_wise(klass: type) -> type:
    """
    Mark a class as implementing an `.ElementWise` operation.

    The class should inherit from `.ElementWise` (though in theory it need not to, as long as it implements the same
    interface). The point of the annotation is to register the class in the list of known operations, so it would be
    available for use in ``...|OperationName...``.
    """
    Filter.register("element_wise", klass)
    return klass


@element_wise
class Abs(ElementWise):
    """
    A filter that converts each value to its absolute value.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use the same data type as the input.
    """

    def __init__(self, *, _input_dtype: str, dtype: Optional[str] = None) -> None:
        dtype = dtype or _input_dtype
        nop = dtype == _input_dtype and is_dtype(dtype, ("uint8", "uint16", "uint32", "uint64"))
        super().__init__(densifies=False, sparsifies=False, dtype=dtype, nop=nop)

    def _ndarray_to_ndarray(self, input_ndarray: np.ndarray, output_ndarray: np.ndarray) -> None:
        np.abs(input_ndarray, out=output_ndarray, casting="same_kind")


@element_wise
class Floor(ElementWise):
    """
    A filter that converts each value to the largest integer no bigger than the value.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use the same data type as the input.
    """

    def __init__(self, *, _input_dtype: str, dtype: Optional[str] = None) -> None:
        dtype = dtype or _input_dtype
        nop = dtype == _input_dtype and is_dtype(dtype, INT_DTYPES)
        super().__init__(densifies=False, sparsifies=False, dtype=dtype, nop=nop)

    def _ndarray_to_ndarray(self, input_ndarray: np.ndarray, output_ndarray: np.ndarray) -> None:
        np.floor(input_ndarray, out=output_ndarray, casting="unsafe")


@element_wise
class Round(ElementWise):
    """
    A filter that converts each value to the nearest integer.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use the same data type as the input.
    """

    def __init__(self, *, _input_dtype: str, dtype: Optional[str] = None) -> None:
        dtype = dtype or _input_dtype
        nop = dtype == _input_dtype and is_dtype(dtype, INT_DTYPES)
        super().__init__(densifies=False, sparsifies=False, dtype=dtype, nop=nop)

    def _ndarray_to_ndarray(self, input_ndarray: np.ndarray, output_ndarray: np.ndarray) -> None:
        np.rint(input_ndarray, out=output_ndarray, casting="unsafe")


@element_wise
class Ceil(ElementWise):
    """
    A filter that converts each value to the lowest integer no smaller than the value.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use the same data type as the input.
    """

    def __init__(self, *, _input_dtype: str, dtype: Optional[str] = None) -> None:
        dtype = dtype or _input_dtype
        nop = dtype == _input_dtype and is_dtype(dtype, INT_DTYPES)
        super().__init__(densifies=False, sparsifies=False, dtype=dtype, nop=nop)

    def _ndarray_to_ndarray(self, input_ndarray: np.ndarray, output_ndarray: np.ndarray) -> None:
        np.ceil(input_ndarray, out=output_ndarray, casting="unsafe")


@element_wise
class Clip(ElementWise):
    """
    A filter that converts each value to the lowest integer no smaller than the value.

    **Required Parameters:**

    ``min``
        The minimal allowed value in the result. Lower values will be raised to this value.

    ``max``
        The maximal allowed value. Higher values will be lowered to this value.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use the same data type as the input.
    """

    def __init__(
        self,
        *,
        _input_dtype: str,
        dtype: Optional[str] = None,
        min: str,  # pylint: disable=redefined-builtin
        max: str,  # pylint: disable=redefined-builtin
    ) -> None:
        self._min = parse_float_parameter(min, "parameter: min of the filter: Clip")
        self._max = parse_float_parameter(max, "parameter: max of the filter: Clip")
        assert self._min < self._max, f"empty allowed values Clip range [ {self._min} .. {self._max} ]"
        dtype = dtype or _input_dtype
        canonical = f"min={self._min},max={self._max}"
        super().__init__(densifies=False, sparsifies=False, dtype=dtype, canonical=canonical)

    def _ndarray_to_ndarray(self, input_ndarray: np.ndarray, output_ndarray: np.ndarray) -> None:
        np.clip(input_ndarray, self._min, self._max, out=output_ndarray, casting="same_kind")


@element_wise
class Log(ElementWise):
    """
    A filter that converts each value to its ``log``.

    **Required Parameters:**

    ``base``
        The base of the log. This can be a number or the special value ``e`` to designate the natural logarithm.

    ``factor``
        A normalization factor added to all values before computing the ``log``, to avoid running into ``log(0)``.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use ``float32`` if the input data type is up to 32 bits and
        ``float64`` otherwise.
    """

    def __init__(self, *, _input_dtype: str, dtype: Optional[str] = None, base: str, factor: str) -> None:
        self._base = None if base == "e" else parse_float_parameter(base, "parameter: base of the filter: Log")
        assert self._base is None or self._base > 0, f"negative base: {self._base} for Log filter"
        self._factor = parse_float_parameter(factor, "parameter: factor of the filter: Log")
        dtype = dtype or float_dtype_for(_input_dtype)
        canonical = f"base={self._base or 'e'},factor={self._factor}"
        super().__init__(densifies=True, sparsifies=False, dtype=dtype, canonical=canonical)

    def _ndarray_to_ndarray(self, input_ndarray: np.ndarray, output_ndarray: np.ndarray) -> None:
        if self._factor != 0:
            output_ndarray[:] = input_ndarray[:]
            input_ndarray = output_ndarray
            input_ndarray += self._factor

        if self._base is None:
            np.log(input_ndarray, out=output_ndarray, casting="same_kind")

        elif self._base == 2:
            np.log2(input_ndarray, out=output_ndarray, casting="same_kind")

        elif self._base == 10:
            np.log10(input_ndarray, out=output_ndarray, casting="same_kind")

        else:
            np.log(input_ndarray, out=output_ndarray, casting="same_kind")
            output_ndarray /= np.log(self._base)


class Reformat(ElementWise):
    """
    A filter that converts between `.Sparse` and `.Dense` matrices.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use the same data type as the input.
    """

    def __init__(self, *, _input_dtype: str, dtype: Optional[str] = None, densifies: bool, sparsifies: bool) -> None:
        dtype = dtype or _input_dtype
        nop = dtype == _input_dtype
        super().__init__(densifies=densifies, sparsifies=sparsifies, dtype=dtype, nop=nop)

    def vector_to_vector(self, input_vector: Vector) -> Vector:
        return be_vector(input_vector.astype(self.dtype))

    def dense_to_dense(self, input_dense: DenseInRows, output_dense: DenseInRows) -> None:
        output_dense[:] = input_dense[:]

    def dense_to_sparse(self, input_dense: DenseInRows) -> SparseInRows:
        return sp.csr_matrix(input_dense, dtype=self.dtype)

    def sparse_to_dense(self, input_sparse: SparseInRows, output_dense: DenseInRows) -> None:
        input_sparse.toarray(out=output_dense)

    def sparse_to_sparse(self, input_sparse: SparseInRows) -> SparseInRows:
        return input_sparse

    def _ndarray_to_ndarray(self, input_ndarray: np.ndarray, output_ndarray: np.ndarray) -> None:
        assert False, "never happens"


@element_wise
class Densify(Reformat):
    """
    A filter that converts `.Sparse` matrices to `.Dense`.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use the same data type as the input.
    """

    def __init__(self, *, _input_dtype: str, dtype: Optional[str] = None) -> None:
        super().__init__(_input_dtype=_input_dtype, dtype=dtype, densifies=True, sparsifies=False)


@element_wise
class Sparsify(Reformat):
    """
    A filter that converts `.Dense` matrices to `.Sparse`.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use the same data type as the input.
    """

    def __init__(self, *, _input_dtype: str, dtype: Optional[str] = None) -> None:
        super().__init__(_input_dtype=_input_dtype, dtype=dtype, densifies=False, sparsifies=True)


@element_wise
class Significant(ElementWise):
    """
    A filter that converts any data to sparse format, preserving only the significant values.

    **Required Parameters:**

    ``high``
        A value of at least this is always preserved.

    ``low``
        A value of at least this is preserved if, in the same row, there is at least one value which is at least
        ``high``.

    **Optional Parameters:**

    ``abs``
        Whether to consider the absolute values when doing the filtering (by default, ``True``). E.g., for fold factors,
        this will preserve both the strong positive and strong negative fold factors, which is exactly what you'd want
        for visualization in a heatmap.

    ``dtype``
        The data type of the output. By default, we use the same data type as the input.

    .. todo::

        Is it possible to implement `.Significant` more efficiently for sparse matrices (in pure Python)?
    """

    def __init__(
        self,
        *,
        _input_dtype: str,
        dtype: Optional[str] = None,
        low: str,
        high: str,
        abs: str = "True",  # pylint: disable=redefined-builtin
    ) -> None:
        self._abs = parse_bool_parameter(abs, "parameter: abs of the filter: Significant")
        self._low = parse_float_parameter(low, "parameter: low of the filter: Significant")
        self._high = parse_float_parameter(high, "parameter: high of the filter: Significant")
        assert self._low <= self._high, f"Significant low: {self._low} is above high: {self._high}"
        dtype = dtype or _input_dtype
        canonical = f"low={self._low},high={self._high},abs={self._abs}"
        super().__init__(densifies=False, sparsifies=True, dtype=dtype, canonical=canonical)

    def vector_to_vector(self, input_vector: Vector) -> Vector:
        """
        :meta private:
        """
        if self._abs:
            output_vector = np.abs(input_vector, dtype=self.dtype, casting="same_kind")
        else:
            output_vector = input_vector.astype(self.dtype, casting="same_kind")

        max_value = output_vector.max()
        if max_value < self._high:
            output_vector[:] = 0

        else:
            if self._abs:
                mask = output_vector >= self._low
                output_vector[:] = 0
                output_vector[mask] = input_vector[mask]
            else:
                output_vector[output_vector < self._low] = 0

        return be_vector(output_vector)

    def dense_to_dense(self, input_dense: DenseInRows, output_dense: DenseInRows) -> None:
        """
        :meta private:
        """
        if self._abs:
            np.abs(input_dense, out=output_dense, casting="same_kind")
        else:
            output_dense[:] = input_dense[:]

        max_value_of_rows = output_dense.max(axis=1)
        insignificant_rows_mask = max_value_of_rows < self._high
        insignificant_entries_mask = output_dense < self._low

        if self._abs:
            output_dense[:] = input_dense[:]

        output_dense[insignificant_rows_mask, :] = 0
        output_dense[insignificant_entries_mask] = 0

    def dense_to_sparse(self, input_dense: DenseInRows) -> SparseInRows:
        """
        :meta private:
        """
        tmp_dense = np.empty(input_dense.shape, dtype=self.dtype)
        self.dense_to_dense(input_dense, be_dense_in_rows(tmp_dense))
        return sp.csr_matrix(tmp_dense)

    def sparse_to_dense(self, input_sparse: SparseInRows, output_dense: DenseInRows) -> None:
        """
        :meta private:
        """
        return self.dense_to_dense(input_sparse.toarray(order="C"), output_dense)

    def sparse_to_sparse(self, input_sparse: SparseInRows) -> SparseInRows:
        """
        :meta private:
        """
        tmp_dense = np.empty(input_sparse.shape, dtype=self.dtype)
        self.sparse_to_dense(input_sparse, be_dense_in_rows(tmp_dense))
        return sp.csr_matrix(tmp_dense)

    def _ndarray_to_ndarray(self, input_ndarray: np.ndarray, output_ndarray: np.ndarray) -> None:
        assert False, "never happens"


class Reduction(Filter):
    """
    Describe a reduction operation (e.g., sum).

    If the input is a vector, the output is a scalar.

    If the input is a matrix, the output is a vector with a value for each row. Functionally this should be identical to
    applying the vector reduction to each matrix row.
    """

    @abstractmethod
    def vector_to_scalar(self, input_vector: Vector) -> Any:
        """
        Reduce an input ``vector`` to a single scalar value.
        """

    @abstractmethod
    def dense_to_vector(self, input_dense: DenseInRows) -> Vector:
        """
        Reduce a dense `.ROW_MAJOR` ``input`` matrix into a new per-row output vector.
        """

    @abstractmethod
    def sparse_to_vector(self, input_sparse: SparseInRows) -> Vector:
        """
        Reduce a sparse `.ROW_MAJOR` ``input`` matrix into a new per-row output vector.
        """


def reduction(klass: type) -> type:
    """
    Mark a class as implementing a `.Reduction` operation.

    The class should inherit from `.Reduction` (though in theory it need not to, as long as it implements the same
    interface). The point of the annotation is to register the class in the list of known operations, so it would be
    available for use in ``...|OperationName...``.

    For simplicity we register the operation under the (unqualified) class name and assert there are no ambiguities.
    """
    Filter.register("reduction", klass)
    return klass


@reduction
class Sum(Reduction):
    """
    A filter that sums all the values, assuming there are no ``None`` (that is, ``NaN``) values in the data.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use the same data type as the input. This may be insufficient if the
        input is a small integer type.
    """

    def __init__(self, *, _input_dtype: str, dtype: Optional[str] = None) -> None:
        super().__init__(dtype=dtype or _input_dtype)

    def vector_to_scalar(self, input_vector: Vector) -> Any:
        """
        :meta private:
        """
        return input_vector.sum()

    def dense_to_vector(self, input_dense: DenseInRows) -> Vector:
        """
        :meta private:
        """
        return input_dense.sum(axis=1, dtype=self.dtype)

    def sparse_to_vector(self, input_sparse: SparseInRows) -> Vector:
        """
        :meta private:
        """
        return as_vector(input_sparse.sum(axis=1, dtype=self.dtype))


@reduction
class Min(Reduction):
    """
    A filter that returns the minimal value, assuming there are no ``None`` (that is, ``NaN``) values in the data.
    """

    def __init__(self, *, _input_dtype: str) -> None:
        super().__init__(dtype=_input_dtype)

    def vector_to_scalar(self, input_vector: Vector) -> Any:
        """
        :meta private:
        """
        return input_vector.min()

    def dense_to_vector(self, input_dense: DenseInRows) -> Vector:
        """
        :meta private:
        """
        return input_dense.min(axis=1)

    def sparse_to_vector(self, input_sparse: SparseInRows) -> Vector:
        """
        :meta private:
        """
        return as_vector(input_sparse.min(axis=1))


@reduction
class Max(Reduction):
    """
    A filter that returns the maximal value, assuming there are no ``None`` (that is, ``NaN``) values in the data.
    """

    def __init__(self, *, _input_dtype: str) -> None:
        super().__init__(dtype=_input_dtype)

    def vector_to_scalar(self, input_vector: Vector) -> Any:
        """
        :meta private:
        """
        return input_vector.max()

    def dense_to_vector(self, input_dense: DenseInRows) -> Vector:
        """
        :meta private:
        """
        return input_dense.max(axis=1)

    def sparse_to_vector(self, input_sparse: SparseInRows) -> Vector:
        """
        :meta private:
        """
        return as_vector(input_sparse.max(axis=1))


@reduction
class Mean(Reduction):
    """
    A filter that returns the mean value, assuming there are no ``None`` (that is, ``NaN``) values in the data.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use ``float32`` if the input data type is up to 32 bits and
        ``float64`` otherwise.
    """

    def __init__(self, *, _input_dtype: str, dtype: Optional[str] = None) -> None:
        super().__init__(dtype=dtype or float_dtype_for(_input_dtype))

    def vector_to_scalar(self, input_vector: Vector) -> Any:
        """
        :meta private:
        """
        return input_vector.mean()

    def dense_to_vector(self, input_dense: DenseInRows) -> Vector:
        """
        :meta private:
        """
        return input_dense.mean(axis=1, dtype=self.dtype)

    def sparse_to_vector(self, input_sparse: SparseInRows) -> Vector:
        """
        :meta private:
        """
        return as_vector(input_sparse.mean(axis=1, dtype=self.dtype))


@reduction
class Var(Reduction):
    """
    A filter that returns the variance of the values, assuming there are no ``None`` (that is, ``NaN``) values in the
    data.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use ``float32`` if the input data type is up to 32 bits and
        ``float64`` otherwise.
    """

    def __init__(self, *, _input_dtype: str, dtype: Optional[str] = None) -> None:
        super().__init__(dtype=dtype or float_dtype_for(_input_dtype))

    def vector_to_scalar(self, input_vector: Vector) -> Any:
        """
        :meta private:
        """
        return input_vector.var()

    def dense_to_vector(self, input_dense: DenseInRows) -> Vector:
        """
        :meta private:
        """
        return input_dense.var(axis=1, dtype=self.dtype)

    def sparse_to_vector(self, input_sparse: SparseInRows) -> Vector:
        """
        :meta private:
        """
        sum_of_values_in_rows = as_vector(input_sparse.sum(axis=1, dtype=self.dtype))

        squared_sum_of_values_in_rows = sum_of_values_in_rows
        squared_sum_of_values_in_rows *= squared_sum_of_values_in_rows  # type: ignore

        squared_nnz_data = input_sparse.data.copy()
        squared_nnz_data *= squared_nnz_data  # type: ignore

        squared_sparse = sp.csr_matrix((squared_nnz_data, input_sparse.indices, input_sparse.indptr))
        sum_of_squared_values_in_rows = as_vector(squared_sparse.sum(axis=1, dtype=self.dtype))

        variance = squared_sum_of_values_in_rows
        variance /= -input_sparse.shape[1]  # type: ignore
        variance += sum_of_squared_values_in_rows  # type: ignore
        variance /= input_sparse.shape[1]  # type: ignore

        return variance


@reduction
class Std(Var):
    """
    A filter that returns the standard deviation of the values, assuming there are no ``None`` (that is, ``NaN``) values
    in the data.

    **Optional Parameters:**

    ``dtype``
        The data type of the output. By default, we use ``float32`` if the input data type is up to 32 bits and
        ``float64`` otherwise.
    """

    def vector_to_scalar(self, input_vector: Vector) -> Any:
        """
        :meta private:
        """
        return input_vector.std()

    def dense_to_vector(self, input_dense: DenseInRows) -> Vector:
        """
        :meta private:
        """
        return input_dense.std(axis=1, dtype=self.dtype)

    def sparse_to_vector(self, input_sparse: SparseInRows) -> Vector:
        """
        :meta private:
        """
        variance = super().sparse_to_vector(input_sparse)

        std = variance
        np.sqrt(std, out=std)

        return std


def parse_float_parameter(text: str, description: str) -> float:
    """
    Given the ``text`` value of a parameter, convert it to a floating point value,
    and if it is invalid, assert using the ``description``.
    """
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(str(error) + " for the " + description)  # pylint: disable=raise-missing-from


def parse_int_parameter(text: str, description: str) -> int:
    """
    Given the ``text`` value of a parameter, convert it to an integer value, and if it is invalid, assert using the
    ``description``.
    """
    try:
        return int(text)
    except ValueError as error:
        raise ValueError(str(error) + " for the " + description)  # pylint: disable=raise-missing-from


def parse_bool_parameter(text: str, description: str) -> bool:
    """
    Given the ``text`` value of a parameter, convert it to a boolean value,
    and if it is invalid, assert using the ``description``.
    """
    if text in ("TRUE", "True", "true", "T", "t", "YES", "Yes", "yes", "Y", "y"):
        return True
    if text in ("FALSE", "False", "false", "F", "f", "NO", "No", "no", "N", "n"):
        return False
    raise ValueError(f"invalid boolean value: {text} for the {description}")


def float_dtype_for(dtype: str) -> Optional[str]:
    """
    Given an input ``dtype``, return a reasonable output dtype for operations with floating point output.
    """
    if dtype in ("float16", "float32", "float64"):
        return dtype
    if dtype in ("int64", "uint64"):
        return "float64"
    return "float32"
