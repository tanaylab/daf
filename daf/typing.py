"""
Typing
------

The code has to deal with many different alternative data types for what is essentially two basic data types: 2D
matrices and 1D vectors. Even though python supports "duck typing", these alternative data types expose different APIs
and require different code paths for realistic algorithms. Even a simple thing like checking whether the data contains
strings is different between the different APIs.

.. todo::

    The check for string values in ``pandas`` data will accept any object type.

Matrices in particular can be represented in a wide range of formats and variants within each format. Efficient code for
each representation requires different code paths which places a high burden on the consumers of matrix data from daf
containers.

To minimize this burden, daf restricts the matrices it stores to a few variants, specifically either using row-major or
column-major layout for dense matrices and either CSR or CSC format for sparse matrices, as these are the most commonly
used layouts, and these require only a small number of code paths to ensure efficient computation.

Finally, it would be nice to track the data type of the elements, but this would result in a combinatorical explosion of
types (as ``mypy`` generic types are not up to the task).

We therefore provide here only the minimal ``mypy`` annotations allowing expressing the code's intent when it comes to
code paths, and provide some utilities to at least assert the element data type is as expected. In particular, these
type annotations only support the restricted subset we allow to store out of the full set of data types available in
``numpy``, ``pandas`` and ``scipy.sparse``.
"""


# pylint: disable=too-many-lines

from abc import ABC
from typing import Annotated
from typing import Any
from typing import Collection
from typing import NewType
from typing import Optional
from typing import TypeGuard
from typing import TypeVar
from typing import Union
from typing import overload

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from .fake_pandas import PandasFrame
from .fake_pandas import PandasSeries
from .fake_sparse import SparseMatrix

__all__ = [
    "is_optimized",
    "be_optimized",
    "optimize",
    # 1D vectors:
    "Vector",
    "is_vector",
    "be_vector",
    # 1D numpy arrays:
    "Array1D",
    "is_array1d",
    "be_array1d",
    # 1D pandas series:
    "Series",
    "is_series",
    "be_series",
    "as_array1d",
    # 2D Layout
    "AnyMajor",
    "ROW_MAJOR",
    "COLUMN_MAJOR",
    "matrix_layout",
    # 2D data:
    "Matrix",
    "is_matrix",
    "be_matrix",
    "MatrixRows",
    "is_matrix_rows",
    "be_matrix_rows",
    "MatrixColumns",
    "is_matrix_columns",
    "be_matrix_columns",
    # 2D numpy arrays:
    "Array2D",
    "is_array2d",
    "be_array2d",
    "ArrayRows",
    "is_array_rows",
    "be_array_rows",
    "ArrayColumns",
    "is_array_columns",
    "be_array_columns",
    # 2D pandas frames:
    "Frame",
    "is_frame",
    "be_frame",
    "FrameRows",
    "is_frame_rows",
    "be_frame_rows",
    "FrameColumns",
    "is_frame_columns",
    "be_frame_columns",
    "as_array2d",
    # 2D sparse matrices:
    "Sparse",
    "is_sparse",
    "be_sparse",
    "SparseRows",
    "is_sparse_rows",
    "be_sparse_rows",
    "SparseColumns",
    "is_sparse_columns",
    "be_sparse_columns",
]

#: 1-dimensional ``numpy`` array of bool values.
Array1D = NewType("Array1D", Annotated[np.ndarray, "1D"])


def is_array1d(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[Array1D]:
    """
    Check whether some ``data`` is an :py:const:`Array1D`, optionally only of some ``dtype``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    an array of strings.
    """
    return isinstance(data, np.ndarray) and data.ndim == 1 and _is_numpy_dtypes(str(data.dtype), dtype)


def be_array1d(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> Array1D:
    """
    Assert that some ``data`` is an :py:const:`Array1D`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    an array of strings.
    """
    if dtype is None:
        assert is_array1d(data), f"expected a 1D numpy.ndarray of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_array1d(
            data, dtype=dtype
        ), f"expected a 1D numpy.ndarray of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: 1-dimensional ``pandas`` series of bool values.
Series = NewType("Series", PandasSeries)


def is_series(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[Series]:
    """
    Check whether some ``data`` is a :py:const:`Series`, optionally only of some ``dtype``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a series of strings.
    """
    return (
        isinstance(data, pd.Series)
        and (
            (isinstance(data.values, np.ndarray) and data.values.ndim == 1) or str(data.dtype) in ("string", "category")
        )
        and _is_pandas_dtypes(str(data.dtype), dtype)
    )


def be_series(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> Series:
    """
    Assert that some ``data`` is a :py:const:`Series`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a series of strings.
    """
    if dtype is None:
        assert is_series(data), f"expected a pandas.Series of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_series(data), f"expected a pandas.Series of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


@overload
def as_array1d(data: Series, *, force_copy: bool = False) -> Array1D:
    ...


@overload
def as_array1d(data: "Matrix", *, force_copy: bool = False) -> Array1D:
    ...


def as_array1d(data: Any, *, force_copy: bool = False) -> Array1D:
    """
    Access the internal 1D ``numpy`` array, if possible; otherwise, or if ``force_copy``, return a copy of the 1D data
    as a ``numpy`` array.

    This ensures that ``pandas`` strings (even if categorical) will be converted to proper ``numpy`` strings.

    This will reshape any matrix with a single row or a single column into a vector.
    """
    if is_matrix(data):
        if is_sparse(data):
            array2d = data.toarray()
        elif is_frame(data):
            array2d = as_array2d(data)
        else:
            array2d = be_array2d(data)
        assert min(array2d.shape) < 2, f"can't convert a matrix of shape {array2d.shape} to a vector"
        return be_array1d(np.reshape(array2d, -1))

    series_dtype = str(data.dtype)
    if series_dtype in ("string", "category"):
        array1d_dtype = "U"
    else:
        array1d_dtype = series_dtype

    array1d = data.values

    if force_copy or not isinstance(array1d, np.ndarray) or array1d_dtype != series_dtype:
        array1d = np.array(array1d, array1d_dtype)

    return array1d


#: Any 1D data.
Vector = Union[Array1D, Series]


def is_vector(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[Vector]:
    """
    Assert that some ``data`` is an :py:const:`Vector`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a vector of strings.
    """
    return is_array1d(data, dtype=dtype) or is_series(data, dtype=dtype)


def be_vector(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> Vector:
    """
    Assert that some ``data`` is a :py:const:`Vector`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a vector of strings.
    """
    if dtype is None:
        assert is_vector(data), f"expected a vector of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_vector(data), f"expected a vector of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: 2-dimensional ``numpy`` array in row-major layout.
ArrayRows = NewType("ArrayRows", Annotated[np.ndarray, "row_major"])


def is_array_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[ArrayRows]:
    """
    Check whether some ``data`` is an :py:const:`ArrayRows`, optionally only of some ``dtype``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    return is_array2d(data, dtype=dtype) and _is_array_row_major(data)


def be_array_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> ArrayRows:
    """
    Assert that some ``data`` is a :py:const:`ArrayRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    if dtype is None:
        assert is_array_rows(
            data
        ), f"expected a row-major numpy.ndarray of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_array_rows(
            data, dtype=dtype
        ), f"expected a row-major numpy.ndarray of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: 2-dimensional ``numpy`` array in column-major layout.
ArrayColumns = NewType("ArrayColumns", Annotated[np.ndarray, "column_major"])


def is_array_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[ArrayColumns]:
    """
    Check whether some ``data`` is an :py:const:`ArrayColumns`, optionally only of some ``dtype``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    return is_array2d(data, dtype=dtype) and _is_array_column_major(data)


def be_array_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> ArrayColumns:
    """
    Assert that some ``data`` is a :py:const:`ArrayColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    if dtype is None:
        assert is_array_columns(
            data
        ), f"expected a column-major numpy.ndarray of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_array_columns(
            data, dtype=dtype
        ), f"expected a column-major numpy.ndarray of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: 2-dimensional ``numpy`` in any-major layout.
Array2D = Union[ArrayRows, ArrayColumns]


def is_array2d(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[Array2D]:
    """
    Check whether some ``data`` is an :py:const:`Array2D`, optionally only of some ``dtype``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.

    .. note::

        This explicitly forbids the deprecated data type ``numpy.matrix`` which like a zombie keeps combing back from
        the grave and causes much havoc when it does.
    """
    return isinstance(data, np.ndarray) and data.ndim == 2 and _is_numpy_dtypes(str(data.dtype), dtype)


def be_array2d(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> Array2D:
    """
    Assert that some ``data`` is an :py:const:`Array2D` optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.

    .. note::

        This explicitly forbids the deprecated data type ``numpy.matrix`` which like a zombie keeps combing back from
        the grave and causes much havoc when it does.
    """
    if dtype is None:
        assert is_array2d(
            data
        ), f"expected a any-major numpy.ndarray of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_array2d(
            data, dtype=dtype
        ), f"expected a any-major numpy.ndarray of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame in row-major layout.
FrameRows = NewType("FrameRows", Annotated[PandasFrame, "row_major"])


def is_frame_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[FrameRows]:
    """
    Check whether some ``data`` is an :py:const:`FrameRows`, optionally only of some ``dtype``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    return is_frame(data, dtype=dtype) and _is_array_row_major(data.values)


def be_frame_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> FrameRows:
    """
    Assert that some ``data`` is a :py:const:`FrameRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    if dtype is None:
        assert is_frame_rows(
            data
        ), f"expected a row-major pandas.DataFrame of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_frame_rows(
            data
        ), f"expected a row-major pandas.DataFrame of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame in column-major layout.
FrameColumns = NewType("FrameColumns", Annotated[PandasFrame, "column_major"])


def is_frame_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[FrameColumns]:
    """
    Check whether some ``data`` is an :py:const:`FrameColumns`, optionally only of some ``dtype``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    return is_frame(data, dtype=dtype) and _is_array_column_major(data.values)


def be_frame_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> FrameColumns:
    """
    Assert that some ``data`` is a :py:const:`FrameColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    if dtype is None:
        assert is_frame_columns(
            data
        ), f"expected a column-major pandas.DataFrame of any reasonable type, got {_data_description(data)}"
    else:
        assert is_frame_columns(
            data
        ), f"expected a column-major pandas.DataFrame of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame in any-major layout.
Frame = Union[FrameRows, FrameColumns]


def is_frame(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[Frame]:
    """
    Check whether some ``data`` is an :py:const:`Frame`, optionally only of some ``dtype``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    return (
        isinstance(data, pd.DataFrame)
        and isinstance(data.values, np.ndarray)
        and data.ndim == 2
        and bool(np.all([_is_pandas_dtypes(str(column_dtype), dtype) for column_dtype in data.dtypes]))
    )


def be_frame(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> Frame:
    """
    Assert that some ``data`` is an :py:const:`Frame` optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    if dtype is None:
        assert is_frame(
            data
        ), f"expected a any-major pandas.DataFrame of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_frame(
            data
        ), f"expected a any-major pandas.DataFrame of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


@overload
def as_array2d(frame: FrameRows, *, force_copy: bool = False) -> ArrayRows:
    ...


@overload
def as_array2d(frame: FrameColumns, *, force_copy: bool = False) -> ArrayColumns:
    ...


def as_array2d(frame: Frame, *, force_copy: bool = False) -> Array2D:
    """
    Access the internal 2D ``numpy`` array, if possible; otherwise, or if ``force_copy``, return a copy of the 2D data
    as a ``numpy`` array.

    This only works if all the data in the frame has the same type.

    This ensures that ``pandas`` strings (even if categorical) will be converted to proper ``numpy`` strings.
    """
    array2d_dtype: Optional[str] = None
    for column_dtype in frame.dtypes:
        dtype = str(column_dtype)
        if dtype in ("string", "category", "object"):
            dtype = "str"
        if array2d_dtype is None:
            array2d_dtype = dtype
        assert (
            array2d_dtype == dtype
        ), f"can't convert to numpy array a frame that contains both {array2d_dtype} and {dtype} data"

    assert array2d_dtype is not None, "can't convert to numpy array an empty frame"

    if array2d_dtype == "str":
        array2d_dtype = "U"

    array2d = frame.values

    if (
        force_copy
        or not isinstance(array2d, np.ndarray)
        or isinstance(array2d, np.matrix)
        or array2d_dtype != array2d.dtype
    ):
        array2d = np.array(array2d, array2d_dtype)

    return array2d


#: 2-dimensional ``scipy.sparse`` matrix in CSR layout.
SparseRows = NewType("SparseRows", Annotated[SparseMatrix, "csr"])


def is_sparse_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[SparseRows]:
    """
    Check whether some ``data`` is an :py:const:`SparseRows`, optionally only of some ``dtype``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    return isinstance(data, sp.csr_matrix) and _is_numpy_dtypes(str(data.data.dtype), dtype)


def be_sparse_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> SparseRows:
    """
    Assert that some ``data`` is a :py:const:`SparseRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    if dtype is None:
        assert is_sparse_rows(
            data
        ), f"expected a scipy.sparse.csr_matrix of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_sparse_rows(
            data
        ), f"expected a scipy.sparse.csr_matrix of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: 2-dimensional ``scipy.sparse`` matrix in CSC layout.
SparseColumns = NewType("SparseColumns", Annotated[SparseMatrix, "csc"])


def is_sparse_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[SparseColumns]:
    """
    Check whether some ``data`` is an :py:const:`SparseColumns`, optionally only of some ``dtype``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    return isinstance(data, sp.csc_matrix) and _is_numpy_dtypes(str(data.data.dtype), dtype)


def be_sparse_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> SparseColumns:
    """
    Assert that some ``data`` is a :py:const:`SparseColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    if dtype is None:
        assert is_sparse_columns(
            data
        ), f"expected a scipy.sparse.csc_matrix of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_sparse_columns(
            data
        ), f"expected a scipy.sparse.csc_matrix of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: 2-dimensional ``scipy.sparse`` matrix in compressed layout.
Sparse = Union[SparseRows, SparseColumns]


def is_sparse(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[Sparse]:
    """
    Check whether some ``data`` is an :py:const:`Sparse`, optionally only of some ``dtype``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    return isinstance(data, (sp.csr_matrix, sp.csc_matrix)) and _is_numpy_dtypes(str(data.data.dtype), dtype)


def be_sparse(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> Sparse:
    """
    Assert that some ``data`` is an :py:const:`Sparse` optionally only of some ``dtype``, and return it as
    such for ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    if dtype is None:
        assert is_sparse(
            data
        ), f"expected a scipy.sparse.csr/csc_matrix of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_sparse(
            data
        ), f"expected a scipy.sparse.csr/csc_matrix of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: Any 2D data.
Matrix = Union[Array2D, Frame, Sparse]


def is_matrix(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[Matrix]:
    """
    Assert that some ``data`` is an :py:const:`Matrix`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    return is_array2d(data, dtype=dtype) or is_frame(data, dtype=dtype) or is_sparse(data, dtype=dtype)


def be_matrix(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> Matrix:
    """
    Assert that some ``data`` is a :py:const:`Matrix`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    if dtype is None:
        assert is_matrix(data), f"expected a any-major matrix of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_matrix(data), f"expected a any-major matrix of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: Any 2D data in row-major layout.
MatrixRows = Union[ArrayRows, FrameRows, SparseRows]


def is_matrix_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[MatrixRows]:
    """
    Assert that some ``data`` is an :py:const:`MatrixRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    return is_array_rows(data, dtype=dtype) or is_frame_rows(data, dtype=dtype) or is_sparse_rows(data, dtype=dtype)


def be_matrix_rows(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> MatrixRows:
    """
    Assert that some ``data`` is a :py:const:`MatrixRows`, optionally only of some ``dtype``, and return it as such for
    ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    if dtype is None:
        assert is_matrix_rows(
            data
        ), f"expected a row-major matrix of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_matrix_rows(
            data
        ), f"expected a row-major matrix of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


#: Any 2D data in column-major layout.
MatrixColumns = Union[ArrayColumns, FrameColumns, SparseColumns]


def is_matrix_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> TypeGuard[MatrixColumns]:
    """
    Assert that some ``data`` is an :py:const:`MatrixColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    return (
        is_array_columns(data, dtype=dtype)
        or is_frame_columns(data, dtype=dtype)
        or is_sparse_columns(data, dtype=dtype)
    )


def be_matrix_columns(data: Any, *, dtype: Optional[Union[str, Collection[str]]] = None) -> MatrixColumns:
    """
    Assert that some ``data`` is a :py:const:`MatrixColumns`, optionally only of some ``dtype``, and return it as such
    for ``mypy``.

    By default, checks that the data type is "reasonable" (bool, int, float, or a string).

    Since ``numpy`` and ``pandas`` can't decide on what the ``dtype`` of a string is, use the value ``str`` to check for
    a matrix of strings.
    """
    if dtype is None:
        assert is_matrix_columns(
            data
        ), f"expected a column-major matrix of any reasonable type, got {_data_description(data)}"
    else:
        if isinstance(dtype, str):
            dtype = (dtype,)
        assert is_matrix_columns(
            data
        ), f"expected a row-major matrix of {' or '.join(dtype)}, got {_data_description(data)}"
    return data


class AnyMajor(ABC):
    """
    Allow for either row-major or column-major matrix layout.
    """

    def __init__(self) -> None:
        #: Name for messages.
        self.name = "any-major"

    def __eq__(self, other: Any) -> bool:
        return id(self.__class__) == id(other.__class__)

    def __ne__(self, other: Any) -> bool:
        return id(self.__class__) != id(other.__class__)


class RowMajor(AnyMajor):  # pylint: disable=too-few-public-methods
    """
    Require row-major layout.
    """

    def __init__(self) -> None:
        #: Name for messages.
        super().__init__()
        self.name = "row-major"


#: Require row-major layout.
ROW_MAJOR = RowMajor()


class ColumnMajor(AnyMajor):  # pylint: disable=too-few-public-methods
    """
    Require column-major layout.
    """

    def __init__(self) -> None:
        #: Name for messages.
        super().__init__()
        self.name = "column-major"


#: Require column-major layout.
COLUMN_MAJOR = ColumnMajor()


@overload
def matrix_layout(matrix: MatrixRows) -> RowMajor:
    ...


@overload
def matrix_layout(matrix: MatrixColumns) -> ColumnMajor:
    ...


def matrix_layout(matrix: Matrix) -> AnyMajor:
    """
    Return the layout of a matrix.
    """
    if is_matrix_rows(matrix):
        return ROW_MAJOR
    if is_matrix_columns(matrix):
        return COLUMN_MAJOR
    assert False, f"expected an any-major matrix of any reasonable type, got {_data_description(matrix)}"


def is_optimized(data: Any) -> bool:
    """
    Whether the ``data`` is in an "optimized" format.

    Even if keeping within the subset of data types supported by daf, there are still cases where the format is
    sub-optimal, resulting in inefficient processing code. For example, a CSR or CSC matrix may contain duplicate or
    unsorted indices, or a matrix or a vector may have strides between its elements.

    It is possible to end up with sub-optimal data formats by performing all sort of operations on "optimized" format
    inputs; this is especially common for sparse matrices. This isn't necessarily an issue for intermediate results.
    However, we restrict data stored in daf containers to only be in "optimized" formats.

    This function tests whether "any" matrix representation is, in fact, in one of the supported, "optimized" formats.
    """
    if isinstance(data, sp.spmatrix):
        return (
            isinstance(data, (sp.csr_matrix, sp.csc_matrix)) and data.has_canonical_format and data.has_sorted_indices
        )
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
        if not isinstance(data, np.ndarray):
            return True
    if isinstance(data, np.ndarray) and 1 <= data.ndim <= 2:
        return data.flags["F_CONTIGUOUS"] or data.flags["C_CONTIGUOUS"]
    assert False, f"expected a matrix or a vector, got {_data_description(data)}"


T = TypeVar("T")


def be_optimized(data: T) -> T:
    """
    Assert that some data is in "optimized" format and return it as-is.
    """
    assert is_optimized(data), f"expected an optimized matrix or a vector, got {_data_description(data)}"
    return data


@overload
def optimize(data: Array1D, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> Array1D:
    ...


@overload
def optimize(data: Series, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> Series:
    ...


@overload
def optimize(data: ArrayRows, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> ArrayRows:
    ...


@overload
def optimize(data: ArrayColumns, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> ArrayColumns:
    ...


@overload
def optimize(data: np.ndarray, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> np.ndarray:
    ...


@overload
def optimize(data: FrameRows, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> FrameRows:
    ...


@overload
def optimize(data: FrameColumns, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> FrameColumns:
    ...


@overload
def optimize(data: PandasFrame, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> PandasFrame:
    ...


@overload
def optimize(data: SparseRows, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> SparseRows:
    ...


@overload
def optimize(data: SparseColumns, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> SparseColumns:
    ...


@overload
def optimize(data: SparseMatrix, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> SparseMatrix:
    ...


@overload
def optimize(data: sp.spmatrix, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> sp.spmatrix:
    ...


def optimize(data: Any, *, force_copy: bool = False, preferred_layout: AnyMajor = ROW_MAJOR) -> Any:
    """
    Given some ``data`` in any (reasonable) format, return it in a supported, "optimized" format.

    If possible, and ``force_copy`` is not specified, this optimizes the data in-place. Otherwise, a copy is created.
    E.g. this will sort the indices of a CSR or CSC matrix in-place.

    If the data is a matrix, and it has no clear layout, a copy will be created using the ``preferred_layout``. E.g.
    this will determine whether a COO matrix will be converted to a CSR or CSC matrix. For vector data, this argument is
    ignored.
    """
    if isinstance(data, np.ndarray) and 1 <= data.ndim <= 2:
        if force_copy or (not data.flags["F_CONTIGUOUS"] and not data.flags["C_CONTIGUOUS"]):
            return np.array(data)
        return data

    if isinstance(data, pd.Series):
        if force_copy or not is_optimized(data):
            data = pd.Series(np.array(data.values), index=data.index)
        return data

    if isinstance(data, pd.DataFrame):
        if force_copy or not is_optimized(data):
            data = pd.DataFrame(np.array(data.values), index=data.index, columns=data.columns)
        return data

    if isinstance(data, sp.spmatrix):
        if isinstance(data, (sp.csr_matrix, sp.csc_matrix)):
            klass = data.__class__
        elif preferred_layout == ROW_MAJOR:
            force_copy = True
            klass = sp.csr_matrix
        else:
            assert preferred_layout == COLUMN_MAJOR
            force_copy = True
            klass = sp.csc_matrix
        if force_copy:
            data = klass(data)
        data.sum_duplicates()
        data.sort_indices()
        return data

    assert False, f"expected a matrix or a vector, got {_data_description(data)}"


def _is_numpy_dtypes(dtype: str, dtypes: Optional[Union[str, Collection[str]]]) -> bool:
    if isinstance(dtypes, str):
        dtypes = (dtypes,)
    if dtypes is not None:
        return dtype in dtypes or ("str" in dtypes and "U" in dtype)
    return "U" in dtype or dtype in (
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
    )


def _is_array_column_major(array: np.ndarray) -> bool:
    return array.flags["F_CONTIGUOUS"]


def _is_array_row_major(array: np.ndarray) -> bool:
    return array.flags["C_CONTIGUOUS"]


def _is_pandas_dtypes(dtype: str, dtypes: Optional[Union[str, Collection[str]]]) -> bool:
    if isinstance(dtypes, str):
        dtypes = (dtypes,)
    if dtypes is None:
        dtypes = (
            "bool",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float16",
            "float32",
            "float64",
            "object",
            "str",
        )
    return dtype in dtypes or ("str" in dtypes and dtype in ("string", "category", "object"))


def _data_description(data: Any) -> str:  # pylint: disable=too-many-return-statements,too-many-branches
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return f"a none-major 1D numpy.ndarray of {data.dtype}"

        if data.ndim == 2:
            is_column_major = _is_array_column_major(data)
            is_row_major = _is_array_row_major(data)
            if is_column_major and is_row_major:
                return f"a both-major 2D numpy.ndarray of {data.dtype}"
            if is_column_major:
                return f"a column-major 2D numpy.ndarray of {data.dtype}"
            if is_row_major:
                return f"a row-major 2D numpy.ndarray of {data.dtype}"
            return f"a none-major 2D numpy.ndarray of {data.dtype}"

        return f"a {data.ndim}D numpy.ndarray of {data.dtype}"

    if isinstance(data, pd.Series):
        if not isinstance(data.values, np.ndarray):
            return (
                "a pandas.Series containing an instance of "
                f"{data.values.__class__.__module__}.{data.values.__class__.__qualname__}"
            )
        assert data.values.ndim == 1
        return f"a pandas.Series of {data.dtype}"

    if isinstance(data, pd.DataFrame):
        if not isinstance(data.values, np.ndarray):
            return (
                "a pandas.DataFrame containing an instance of "
                f"{data.values.__class__.__module__}.{data.values.__class__.__qualname__}"
            )
        assert data.values.ndim == 2
        is_column_major = _is_array_column_major(data.values)
        is_row_major = _is_array_row_major(data.values)
        dtypes = np.unique(data.dtypes)
        if len(dtypes) == 1:
            dtype = str(dtypes[0])
        else:
            dtype = "mixed types"
        if is_column_major and is_row_major:
            return f"a both-major pandas.DataFrame of {dtype}"
        if is_column_major:
            return f"a column-major pandas.DataFrame of {dtype}"
        if is_row_major:
            return f"a row-major pandas.DataFrame of {dtype}"
        return f"a none-major pandas.DataFrame of {dtype}"

    if isinstance(data, sp.csr_matrix):
        return f"a scipy.sparse.csr_matrix of {data.dtype}"

    if isinstance(data, sp.csc_matrix):
        return f"a scipy.sparse.csc_matrix of {data.dtype}"

    if isinstance(data, sp.base.spmatrix):
        return f"a {data.__class__.__module__}.{data.__class__.__qualname__} of {data.dtype}"

    return f"an instance of {data.__class__.__module__}.{data.__class__.__qualname__}"
