"""
Typing
------

The code has to deal with many different alternative data types for what is essentially two basic data types: 2D
matrices and 1D vectors.

Alas, the combination of ``mypy`` limitations, the fact that ``pandas`` and ``scipy`` don't even use it in the 1st
place, and that ``numpy`` does but specifies a single ``ndarray`` type which mixes both vectors and matrices (and
n-dimensional tensors) all conspire against having actually useful type annotations.

Here we provide some type aliases and supporting utilities to allow code to express its intent and help ``mypy`` check
at least basic correctness. Ideally most of this would be added into ``numpy``, ``pandas`` and/or ``scipy``.
Realistically it would be simpler to switch to Julia instead ;-)
"""

from typing import Annotated
from typing import Any
from typing import Collection
from typing import NewType
from typing import TypeGuard
from typing import Union

import numpy as np

__all__ = [
    "is_of_dtypes",
    "is_of_str",
    "Vector",
    "is_vector",
    "be_vector",
    "VectorOfBool",
    "is_vector_of_bool",
    "be_vector_of_bool",
    "VectorOfInt",
    "is_vector_of_int",
    "be_vector_of_int",
    "VectorOfInt8",
    "is_vector_of_int8",
    "be_vector_of_int8",
    "VectorOfUInt8",
    "is_vector_of_uint8",
    "be_vector_of_uint8",
    "VectorOfInt16",
    "is_vector_of_int16",
    "be_vector_of_int16",
    "VectorOfUInt16",
    "is_vector_of_uint16",
    "be_vector_of_uint16",
    "VectorOfInt32",
    "is_vector_of_int32",
    "be_vector_of_int32",
    "VectorOfUInt32",
    "is_vector_of_uint32",
    "be_vector_of_uint32",
    "VectorOfInt64",
    "is_vector_of_int64",
    "be_vector_of_int64",
    "VectorOfUInt64",
    "is_vector_of_uint64",
    "be_vector_of_uint64",
    "VectorOfFloat",
    "is_vector_of_float",
    "be_vector_of_float",
    "VectorOfFloat16",
    "is_vector_of_float16",
    "be_vector_of_float16",
    "VectorOfFloat32",
    "is_vector_of_float32",
    "be_vector_of_float32",
    "VectorOfFloat64",
    "is_vector_of_float64",
    "be_vector_of_float64",
    "VectorOfNum",
    "is_vector_of_num",
    "be_vector_of_num",
    "VectorOfStr",
    "is_vector_of_str",
    "be_vector_of_str",
    "Matrix",
    "is_matrix",
    "be_matrix",
    "MatrixOfBool",
    "is_matrix_of_bool",
    "be_matrix_of_bool",
    "MatrixOfInt",
    "is_matrix_of_int",
    "be_matrix_of_int",
    "MatrixOfInt8",
    "is_matrix_of_int8",
    "be_matrix_of_int8",
    "MatrixOfUInt8",
    "is_matrix_of_uint8",
    "be_matrix_of_uint8",
    "MatrixOfInt16",
    "is_matrix_of_int16",
    "be_matrix_of_int16",
    "MatrixOfUInt16",
    "is_matrix_of_uint16",
    "be_matrix_of_uint16",
    "MatrixOfInt32",
    "is_matrix_of_int32",
    "be_matrix_of_int32",
    "MatrixOfUInt32",
    "is_matrix_of_uint32",
    "be_matrix_of_uint32",
    "MatrixOfInt64",
    "is_matrix_of_int64",
    "be_matrix_of_int64",
    "MatrixOfUInt64",
    "is_matrix_of_uint64",
    "be_matrix_of_uint64",
    "MatrixOfFloat",
    "is_matrix_of_float",
    "be_matrix_of_float",
    "MatrixOfFloat16",
    "is_matrix_of_float16",
    "be_matrix_of_float16",
    "MatrixOfFloat32",
    "is_matrix_of_float32",
    "be_matrix_of_float32",
    "MatrixOfFloat64",
    "is_matrix_of_float64",
    "be_matrix_of_float64",
    "MatrixOfNum",
    "is_matrix_of_num",
    "be_matrix_of_num",
    "MatrixOfStr",
    "is_matrix_of_str",
    "be_matrix_of_str",
]

#: Numpy 1-dimensional data of bool values.
VectorOfBool = NewType("VectorOfBool", Annotated[np.ndarray, "1D", "bool"])


def is_vector_of_bool(array: Any) -> TypeGuard[VectorOfBool]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfBool`.
    """
    return is_vector(array) and is_of_dtypes(array, ("bool"))


def be_vector_of_bool(array: Any) -> VectorOfBool:
    """
    Assert that an ``array`` is a :py:const:`VectorOfBool` and return it as such for ``mypy``.
    """
    assert is_vector_of_bool(array), f"expected a 1D numpy.ndarray of bool, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of int8 values.
VectorOfInt8 = NewType("VectorOfInt8", Annotated[np.ndarray, "1D", "int8"])


def is_vector_of_int8(array: Any) -> TypeGuard[VectorOfInt8]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfInt8`.
    """
    return is_vector(array) and is_of_dtypes(array, ("int8"))


def be_vector_of_int8(array: Any) -> VectorOfInt8:
    """
    Assert that an ``array`` is a :py:const:`VectorOfInt8` and return it as such for ``mypy``.
    """
    assert is_vector_of_int8(array), f"expected a 1D numpy.ndarray of int8, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of uint8 values.
VectorOfUInt8 = NewType("VectorOfUInt8", Annotated[np.ndarray, "1D", "uint8"])


def is_vector_of_uint8(array: Any) -> TypeGuard[VectorOfUInt8]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfInt8`.
    """
    return is_vector(array) and is_of_dtypes(array, ("uint8"))


def be_vector_of_uint8(array: Any) -> VectorOfUInt8:
    """
    Assert that an ``array`` is a :py:const:`VectorOfUInt8` and return it as such for ``mypy``.
    """
    assert is_vector_of_uint8(array), f"expected a 1D numpy.ndarray of uint8, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of int16 values.
VectorOfInt16 = NewType("VectorOfInt16", Annotated[np.ndarray, "1D", "int16"])


def is_vector_of_int16(array: Any) -> TypeGuard[VectorOfInt16]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfInt16`.
    """
    return is_vector(array) and is_of_dtypes(array, ("int16"))


def be_vector_of_int16(array: Any) -> VectorOfInt16:
    """
    Assert that an ``array`` is a :py:const:`VectorOfInt16` and return it as such for ``mypy``.
    """
    assert is_vector_of_int16(array), f"expected a 1D numpy.ndarray of int16, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of uint16 values.
VectorOfUInt16 = NewType("VectorOfUInt16", Annotated[np.ndarray, "1D", "uint16"])


def is_vector_of_uint16(array: Any) -> TypeGuard[VectorOfUInt16]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfInt16`.
    """
    return is_vector(array) and is_of_dtypes(array, ("uint16"))


def be_vector_of_uint16(array: Any) -> VectorOfUInt16:
    """
    Assert that an ``array`` is a :py:const:`VectorOfUInt16` and return it as such for ``mypy``.
    """
    assert is_vector_of_uint16(array), f"expected a 1D numpy.ndarray of uint16, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of int32 values.
VectorOfInt32 = NewType("VectorOfInt32", Annotated[np.ndarray, "1D", "int32"])


def is_vector_of_int32(array: Any) -> TypeGuard[VectorOfInt32]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfInt32`.
    """
    return is_vector(array) and is_of_dtypes(array, ("int32"))


def be_vector_of_int32(array: Any) -> VectorOfInt32:
    """
    Assert that an ``array`` is a :py:const:`VectorOfInt32` and return it as such for ``mypy``.
    """
    assert is_vector_of_int32(array), f"expected a 1D numpy.ndarray of int32, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of uint32 values.
VectorOfUInt32 = NewType("VectorOfUInt32", Annotated[np.ndarray, "1D", "uint32"])


def is_vector_of_uint32(array: Any) -> TypeGuard[VectorOfUInt32]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfInt32`.
    """
    return is_vector(array) and is_of_dtypes(array, ("uint32"))


def be_vector_of_uint32(array: Any) -> VectorOfUInt32:
    """
    Assert that an ``array`` is a :py:const:`VectorOfUInt32` and return it as such for ``mypy``.
    """
    assert is_vector_of_uint32(array), f"expected a 1D numpy.ndarray of uint32, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of int64 values.
VectorOfInt64 = NewType("VectorOfInt64", Annotated[np.ndarray, "1D", "int64"])


def is_vector_of_int64(array: Any) -> TypeGuard[VectorOfInt64]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfInt64`.
    """
    return is_vector(array) and is_of_dtypes(array, ("int64"))


def be_vector_of_int64(array: Any) -> VectorOfInt64:
    """
    Assert that an ``array`` is a :py:const:`VectorOfInt64` and return it as such for ``mypy``.
    """
    assert is_vector_of_int64(array), f"expected a 1D numpy.ndarray of int64, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of uint64 values.
VectorOfUInt64 = NewType("VectorOfUInt64", Annotated[np.ndarray, "1D", "uint64"])


def is_vector_of_uint64(array: Any) -> TypeGuard[VectorOfUInt64]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfInt64`.
    """
    return is_vector(array) and is_of_dtypes(array, ("uint64"))


def be_vector_of_uint64(array: Any) -> VectorOfUInt64:
    """
    Assert that an ``array`` is a :py:const:`VectorOfUInt64` and return it as such for ``mypy``.
    """
    assert is_vector_of_uint64(array), f"expected a 1D numpy.ndarray of uint64, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of any integer values.
VectorOfInt = Union[
    VectorOfInt8,
    VectorOfUInt8,
    VectorOfInt16,
    VectorOfUInt16,
    VectorOfInt32,
    VectorOfUInt32,
    VectorOfInt64,
    VectorOfUInt64,
]


def is_vector_of_int(array: Any) -> TypeGuard[VectorOfInt]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfInt`.
    """
    return is_vector(array) and is_of_dtypes(
        array, ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
    )


def be_vector_of_int(array: Any) -> VectorOfInt:
    """
    Assert that an ``array`` is a :py:const:`VectorOfInt` and return it as such for ``mypy``.
    """
    assert is_vector_of_int(array), f"expected a 1D numpy.ndarray of [u]int*, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of float16 values.
VectorOfFloat16 = NewType("VectorOfFloat16", Annotated[np.ndarray, "1D", "float16"])


def is_vector_of_float16(array: Any) -> TypeGuard[VectorOfFloat16]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfFloat16`.
    """
    return is_vector(array) and is_of_dtypes(array, ("float16"))


def be_vector_of_float16(array: Any) -> VectorOfFloat16:
    """
    Assert that an ``array`` is a :py:const:`VectorOfFloat16` and return it as such for ``mypy``.
    """
    assert is_vector_of_float16(array), f"expected a 1D numpy.ndarray of float16, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of float32 values.
VectorOfFloat32 = NewType("VectorOfFloat32", Annotated[np.ndarray, "1D", "float32"])


def is_vector_of_float32(array: Any) -> TypeGuard[VectorOfFloat32]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfFloat32`.
    """
    return is_vector(array) and is_of_dtypes(array, ("float32"))


def be_vector_of_float32(array: Any) -> VectorOfFloat32:
    """
    Assert that an ``array`` is a :py:const:`VectorOfFloat32` and return it as such for ``mypy``.
    """
    assert is_vector_of_float32(array), f"expected a 1D numpy.ndarray of float32, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of float64 values.
VectorOfFloat64 = NewType("VectorOfFloat64", Annotated[np.ndarray, "1D", "float64"])


def is_vector_of_float64(array: Any) -> TypeGuard[VectorOfFloat64]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfFloat64`.
    """
    return is_vector(array) and is_of_dtypes(array, ("float64"))


def be_vector_of_float64(array: Any) -> VectorOfFloat64:
    """
    Assert that an ``array`` is a :py:const:`VectorOfFloat64` and return it as such for ``mypy``.
    """
    assert is_vector_of_float64(array), f"expected a 1D numpy.ndarray of float64, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of any float values.
#:
#: .. todo::
#:
#:    If/when ``numpy`` supports things like ``bfloat16``, and/or ``float8``, add them as well.
VectorOfFloat = Union[VectorOfFloat16, VectorOfFloat32, VectorOfFloat64]


def is_vector_of_float(array: Any) -> TypeGuard[VectorOfFloat]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfFloat`.
    """
    return is_vector(array) and is_of_dtypes(array, ("float16", "float32", "float64"))


def be_vector_of_float(array: Any) -> VectorOfFloat:
    """
    Assert that an ``array`` is a :py:const:`VectorOfFloat` and return it as such for ``mypy``.
    """
    assert is_vector_of_float(array), f"expected a 1D numpy.ndarray of float*, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of any numeric values.
VectorOfNum = Union[VectorOfInt, VectorOfFloat]


def is_vector_of_num(array: Any) -> TypeGuard[VectorOfNum]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfNum`.
    """
    return is_vector(array) and is_of_dtypes(
        array,
        ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64"),
    )


def be_vector_of_num(array: Any) -> VectorOfNum:
    """
    Assert that an ``array`` is a :py:const:`VectorOfNum` and return it as such for ``mypy``.
    """
    assert is_vector_of_num(array), f"expected a 1D numpy.ndarray of [u]int* or float*, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of any string values.
VectorOfStr = Union[VectorOfInt, VectorOfFloat]


def is_vector_of_str(array: Any) -> TypeGuard[VectorOfStr]:
    """
    Check whether some ``array`` is a :py:const:`VectorOfStr`.
    """
    return is_vector(array) and is_of_str(array)


def be_vector_of_str(array: Any) -> VectorOfStr:
    """
    Assert that an ``array`` is a :py:const:`VectorOfStr` and return it as such for ``mypy``.
    """
    assert is_vector_of_str(array), f"expected a 1D numpy.ndarray of str, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional data of any (reasonable) data type.
Vector = Union[VectorOfBool, VectorOfNum, VectorOfStr]


def is_vector(array: Any) -> TypeGuard[Vector]:
    """
    Check whether some ``array`` is a :py:const:`Vector`.
    """
    return isinstance(array, np.ndarray) and array.ndim == 1


def be_vector(array: Any) -> Vector:
    """
    Assert that an ``array`` is a :py:const:`Vector` and return it as such for ``mypy``.
    """
    assert is_vector(array), f"expected a 1D numpy.ndarray of any, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of bool values.
MatrixOfBool = NewType("MatrixOfBool", Annotated[np.ndarray, "2D", "bool"])


def is_matrix_of_bool(array: Any) -> TypeGuard[MatrixOfBool]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfBool`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("bool"))


def be_matrix_of_bool(array: Any) -> MatrixOfBool:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfBool` and return it as such for ``mypy``.
    """
    assert is_matrix_of_bool(array), f"expected a 2D numpy.ndarray of bool, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of int8 values.
MatrixOfInt8 = NewType("MatrixOfInt8", Annotated[np.ndarray, "2D", "int8"])


def is_matrix_of_int8(array: Any) -> TypeGuard[MatrixOfInt8]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfInt8`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("int8"))


def be_matrix_of_int8(array: Any) -> MatrixOfInt8:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfInt8` and return it as such for ``mypy``.
    """
    assert is_matrix_of_int8(array), f"expected a 2D numpy.ndarray of int8, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of uint8 values.
MatrixOfUInt8 = NewType("MatrixOfUInt8", Annotated[np.ndarray, "2D", "uint8"])


def is_matrix_of_uint8(array: Any) -> TypeGuard[MatrixOfUInt8]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfInt8`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("uint8"))


def be_matrix_of_uint8(array: Any) -> MatrixOfUInt8:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfUInt8` and return it as such for ``mypy``.
    """
    assert is_matrix_of_uint8(array), f"expected a 2D numpy.ndarray of uint8, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of int16 values.
MatrixOfInt16 = NewType("MatrixOfInt16", Annotated[np.ndarray, "2D", "int16"])


def is_matrix_of_int16(array: Any) -> TypeGuard[MatrixOfInt16]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfInt16`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("int16"))


def be_matrix_of_int16(array: Any) -> MatrixOfInt16:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfInt16` and return it as such for ``mypy``.
    """
    assert is_matrix_of_int16(array), f"expected a 2D numpy.ndarray of int16, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of uint16 values.
MatrixOfUInt16 = NewType("MatrixOfUInt16", Annotated[np.ndarray, "2D", "uint16"])


def is_matrix_of_uint16(array: Any) -> TypeGuard[MatrixOfUInt16]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfInt16`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("uint16"))


def be_matrix_of_uint16(array: Any) -> MatrixOfUInt16:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfUInt16` and return it as such for ``mypy``.
    """
    assert is_matrix_of_uint16(array), f"expected a 2D numpy.ndarray of uint16, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of int32 values.
MatrixOfInt32 = NewType("MatrixOfInt32", Annotated[np.ndarray, "2D", "int32"])


def is_matrix_of_int32(array: Any) -> TypeGuard[MatrixOfInt32]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfInt32`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("int32"))


def be_matrix_of_int32(array: Any) -> MatrixOfInt32:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfInt32` and return it as such for ``mypy``.
    """
    assert is_matrix_of_int32(array), f"expected a 2D numpy.ndarray of int32, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of uint32 values.
MatrixOfUInt32 = NewType("MatrixOfUInt32", Annotated[np.ndarray, "2D", "uint32"])


def is_matrix_of_uint32(array: Any) -> TypeGuard[MatrixOfUInt32]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfInt32`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("uint32"))


def be_matrix_of_uint32(array: Any) -> MatrixOfUInt32:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfUInt32` and return it as such for ``mypy``.
    """
    assert is_matrix_of_uint32(array), f"expected a 2D numpy.ndarray of uint32, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of int64 values.
MatrixOfInt64 = NewType("MatrixOfInt64", Annotated[np.ndarray, "2D", "int64"])


def is_matrix_of_int64(array: Any) -> TypeGuard[MatrixOfInt64]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfInt64`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("int64"))


def be_matrix_of_int64(array: Any) -> MatrixOfInt64:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfInt64` and return it as such for ``mypy``.
    """
    assert is_matrix_of_int64(array), f"expected a 2D numpy.ndarray of int64, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of uint64 values.
MatrixOfUInt64 = NewType("MatrixOfUInt64", Annotated[np.ndarray, "2D", "uint64"])


def is_matrix_of_uint64(array: Any) -> TypeGuard[MatrixOfUInt64]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfInt64`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("uint64"))


def be_matrix_of_uint64(array: Any) -> MatrixOfUInt64:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfUInt64` and return it as such for ``mypy``.
    """
    assert is_matrix_of_uint64(array), f"expected a 2D numpy.ndarray of uint64, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of any integer values.
MatrixOfInt = Union[
    MatrixOfInt8,
    MatrixOfUInt8,
    MatrixOfInt16,
    MatrixOfUInt16,
    MatrixOfInt32,
    MatrixOfUInt32,
    MatrixOfInt64,
    MatrixOfUInt64,
]


def is_matrix_of_int(array: Any) -> TypeGuard[MatrixOfInt]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfInt`.
    """
    return is_matrix(array) and is_of_dtypes(
        array, ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
    )


def be_matrix_of_int(array: Any) -> MatrixOfInt:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfInt` and return it as such for ``mypy``.
    """
    assert is_matrix_of_int(array), f"expected a 2D numpy.ndarray of [u]int*, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of float16 values.
MatrixOfFloat16 = NewType("MatrixOfFloat16", Annotated[np.ndarray, "2D", "float16"])


def is_matrix_of_float16(array: Any) -> TypeGuard[MatrixOfFloat16]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfFloat16`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("float16"))


def be_matrix_of_float16(array: Any) -> MatrixOfFloat16:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfFloat16` and return it as such for ``mypy``.
    """
    assert is_matrix_of_float16(array), f"expected a 2D numpy.ndarray of float16, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of float32 values.
MatrixOfFloat32 = NewType("MatrixOfFloat32", Annotated[np.ndarray, "2D", "float32"])


def is_matrix_of_float32(array: Any) -> TypeGuard[MatrixOfFloat32]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfFloat32`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("float32"))


def be_matrix_of_float32(array: Any) -> MatrixOfFloat32:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfFloat32` and return it as such for ``mypy``.
    """
    assert is_matrix_of_float32(array), f"expected a 2D numpy.ndarray of float32, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of float64 values.
MatrixOfFloat64 = NewType("MatrixOfFloat64", Annotated[np.ndarray, "2D", "float64"])


def is_matrix_of_float64(array: Any) -> TypeGuard[MatrixOfFloat64]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfFloat64`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("float64"))


def be_matrix_of_float64(array: Any) -> MatrixOfFloat64:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfFloat64` and return it as such for ``mypy``.
    """
    assert is_matrix_of_float64(array), f"expected a 2D numpy.ndarray of float64, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of any float values.
#:
#: .. todo::
#:
#:    If/when ``numpy`` supports things like ``bfloat16``, and/or ``float8``, add them as well.
MatrixOfFloat = Union[MatrixOfFloat16, MatrixOfFloat32, MatrixOfFloat64]


def is_matrix_of_float(array: Any) -> TypeGuard[MatrixOfFloat]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfFloat`.
    """
    return is_matrix(array) and is_of_dtypes(array, ("float16", "float32", "float64"))


def be_matrix_of_float(array: Any) -> MatrixOfFloat:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfFloat` and return it as such for ``mypy``.
    """
    assert is_matrix_of_float(array), f"expected a 2D numpy.ndarray of float*, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of any numeric values.
MatrixOfNum = Union[MatrixOfInt, MatrixOfFloat]


def is_matrix_of_num(array: Any) -> TypeGuard[MatrixOfNum]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfNum`.
    """
    return is_matrix(array) and is_of_dtypes(
        array,
        ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64"),
    )


def be_matrix_of_num(array: Any) -> MatrixOfNum:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfNum` and return it as such for ``mypy``.
    """
    assert is_matrix_of_num(array), f"expected a 2D numpy.ndarray of [u]int* or float*, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of any string values.
MatrixOfStr = Union[MatrixOfInt, MatrixOfFloat]


def is_matrix_of_str(array: Any) -> TypeGuard[MatrixOfStr]:
    """
    Check whether some ``array`` is a :py:const:`MatrixOfStr`.
    """
    return is_matrix(array) and is_of_str(array)


def be_matrix_of_str(array: Any) -> MatrixOfStr:
    """
    Assert that an ``array`` is a :py:const:`MatrixOfStr` and return it as such for ``mypy``.
    """
    assert is_matrix_of_str(array), f"expected a 2D numpy.ndarray of str, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional data of any (reasonable) data type.
Matrix = Union[MatrixOfBool, MatrixOfNum, MatrixOfStr]


def is_matrix(array: Any) -> TypeGuard[Matrix]:
    """
    Check whether some ``array`` is a :py:const:`Matrix`.
    """
    return isinstance(array, np.ndarray) and array.ndim == 2


def be_matrix(array: Any) -> Matrix:
    """
    Assert that an ``array`` is a :py:const:`Matrix` and return it as such for ``mypy``.
    """
    assert is_matrix(array), f"expected a 2D numpy.ndarray of any, got {_array_description(array)}"
    return array


def is_of_dtypes(array: Any, dtypes: Collection[str]) -> bool:
    """
    Check whether some ``array`` is a numpy array of one of the ``dtypes``.
    """
    return isinstance(array, np.ndarray) and str(array.dtype) in dtypes


def is_of_str(array: Any) -> bool:
    """
    Check whether some ``array`` is a numpy array of string values.
    """
    return isinstance(array, np.ndarray) and "U" in str(array.dtype)


def _array_description(array: Any) -> str:
    if isinstance(array, np.ndarray):
        return f"a {array.ndim}D numpy.ndarray of {array.dtype}"
    return f"an instance of {array.__class__.__module__}.{array.__class__.__qualname__}"
