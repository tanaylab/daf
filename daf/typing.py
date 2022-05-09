"""
Typing
------

The code has to deal with many different alternative data types for what is essentially two basic data types: 2D
matrices and 1D array1ds.

Alas, the combination of ``mypy`` limitations, the fact that ``pandas`` and ``scipy`` don't even use it in the 1st
place, and that ``numpy`` does but specifies a single ``ndarray`` type which mixes both 1D vectors and 2D matrices (and
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
    "Array1D",
    "is_array1d",
    "be_array1d",
    "Array1DofBool",
    "is_array1d_of_bool",
    "be_array1d_of_bool",
    "Array1DofInt",
    "is_array1d_of_int",
    "be_array1d_of_int",
    "Array1DofInt8",
    "is_array1d_of_int8",
    "be_array1d_of_int8",
    "Array1DofUInt8",
    "is_array1d_of_uint8",
    "be_array1d_of_uint8",
    "Array1DofInt16",
    "is_array1d_of_int16",
    "be_array1d_of_int16",
    "Array1DofUInt16",
    "is_array1d_of_uint16",
    "be_array1d_of_uint16",
    "Array1DofInt32",
    "is_array1d_of_int32",
    "be_array1d_of_int32",
    "Array1DofUInt32",
    "is_array1d_of_uint32",
    "be_array1d_of_uint32",
    "Array1DofInt64",
    "is_array1d_of_int64",
    "be_array1d_of_int64",
    "Array1DofUInt64",
    "is_array1d_of_uint64",
    "be_array1d_of_uint64",
    "Array1DofFloat",
    "is_array1d_of_float",
    "be_array1d_of_float",
    "Array1DofFloat16",
    "is_array1d_of_float16",
    "be_array1d_of_float16",
    "Array1DofFloat32",
    "is_array1d_of_float32",
    "be_array1d_of_float32",
    "Array1DofFloat64",
    "is_array1d_of_float64",
    "be_array1d_of_float64",
    "Array1DofNum",
    "is_array1d_of_num",
    "be_array1d_of_num",
    "Array1DofStr",
    "is_array1d_of_str",
    "be_array1d_of_str",
    "Array2D",
    "is_array2d",
    "be_array2d",
    "Array2DofBool",
    "is_array2d_of_bool",
    "be_array2d_of_bool",
    "Array2DofInt",
    "is_array2d_of_int",
    "be_array2d_of_int",
    "Array2DofInt8",
    "is_array2d_of_int8",
    "be_array2d_of_int8",
    "Array2DofUInt8",
    "is_array2d_of_uint8",
    "be_array2d_of_uint8",
    "Array2DofInt16",
    "is_array2d_of_int16",
    "be_array2d_of_int16",
    "Array2DofUInt16",
    "is_array2d_of_uint16",
    "be_array2d_of_uint16",
    "Array2DofInt32",
    "is_array2d_of_int32",
    "be_array2d_of_int32",
    "Array2DofUInt32",
    "is_array2d_of_uint32",
    "be_array2d_of_uint32",
    "Array2DofInt64",
    "is_array2d_of_int64",
    "be_array2d_of_int64",
    "Array2DofUInt64",
    "is_array2d_of_uint64",
    "be_array2d_of_uint64",
    "Array2DofFloat",
    "is_array2d_of_float",
    "be_array2d_of_float",
    "Array2DofFloat16",
    "is_array2d_of_float16",
    "be_array2d_of_float16",
    "Array2DofFloat32",
    "is_array2d_of_float32",
    "be_array2d_of_float32",
    "Array2DofFloat64",
    "is_array2d_of_float64",
    "be_array2d_of_float64",
    "Array2DofNum",
    "is_array2d_of_num",
    "be_array2d_of_num",
    "Array2DofStr",
    "is_array2d_of_str",
    "be_array2d_of_str",
]

#: Numpy 1-dimensional array of bool values.
Array1DofBool = NewType("Array1DofBool", Annotated[np.ndarray, "1D", "bool"])


def is_array1d_of_bool(array: Any) -> TypeGuard[Array1DofBool]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofBool`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("bool"))


def be_array1d_of_bool(array: Any) -> Array1DofBool:
    """
    Assert that an ``array`` is a :py:const:`Array1DofBool` and return it as such for ``mypy``.
    """
    assert is_array1d_of_bool(array), f"expected a 1D numpy.ndarray of bool, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of int8 values.
Array1DofInt8 = NewType("Array1DofInt8", Annotated[np.ndarray, "1D", "int8"])


def is_array1d_of_int8(array: Any) -> TypeGuard[Array1DofInt8]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofInt8`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("int8"))


def be_array1d_of_int8(array: Any) -> Array1DofInt8:
    """
    Assert that an ``array`` is a :py:const:`Array1DofInt8` and return it as such for ``mypy``.
    """
    assert is_array1d_of_int8(array), f"expected a 1D numpy.ndarray of int8, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of uint8 values.
Array1DofUInt8 = NewType("Array1DofUInt8", Annotated[np.ndarray, "1D", "uint8"])


def is_array1d_of_uint8(array: Any) -> TypeGuard[Array1DofUInt8]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofInt8`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("uint8"))


def be_array1d_of_uint8(array: Any) -> Array1DofUInt8:
    """
    Assert that an ``array`` is a :py:const:`Array1DofUInt8` and return it as such for ``mypy``.
    """
    assert is_array1d_of_uint8(array), f"expected a 1D numpy.ndarray of uint8, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of int16 values.
Array1DofInt16 = NewType("Array1DofInt16", Annotated[np.ndarray, "1D", "int16"])


def is_array1d_of_int16(array: Any) -> TypeGuard[Array1DofInt16]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofInt16`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("int16"))


def be_array1d_of_int16(array: Any) -> Array1DofInt16:
    """
    Assert that an ``array`` is a :py:const:`Array1DofInt16` and return it as such for ``mypy``.
    """
    assert is_array1d_of_int16(array), f"expected a 1D numpy.ndarray of int16, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of uint16 values.
Array1DofUInt16 = NewType("Array1DofUInt16", Annotated[np.ndarray, "1D", "uint16"])


def is_array1d_of_uint16(array: Any) -> TypeGuard[Array1DofUInt16]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofInt16`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("uint16"))


def be_array1d_of_uint16(array: Any) -> Array1DofUInt16:
    """
    Assert that an ``array`` is a :py:const:`Array1DofUInt16` and return it as such for ``mypy``.
    """
    assert is_array1d_of_uint16(array), f"expected a 1D numpy.ndarray of uint16, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of int32 values.
Array1DofInt32 = NewType("Array1DofInt32", Annotated[np.ndarray, "1D", "int32"])


def is_array1d_of_int32(array: Any) -> TypeGuard[Array1DofInt32]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofInt32`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("int32"))


def be_array1d_of_int32(array: Any) -> Array1DofInt32:
    """
    Assert that an ``array`` is a :py:const:`Array1DofInt32` and return it as such for ``mypy``.
    """
    assert is_array1d_of_int32(array), f"expected a 1D numpy.ndarray of int32, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of uint32 values.
Array1DofUInt32 = NewType("Array1DofUInt32", Annotated[np.ndarray, "1D", "uint32"])


def is_array1d_of_uint32(array: Any) -> TypeGuard[Array1DofUInt32]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofInt32`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("uint32"))


def be_array1d_of_uint32(array: Any) -> Array1DofUInt32:
    """
    Assert that an ``array`` is a :py:const:`Array1DofUInt32` and return it as such for ``mypy``.
    """
    assert is_array1d_of_uint32(array), f"expected a 1D numpy.ndarray of uint32, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of int64 values.
Array1DofInt64 = NewType("Array1DofInt64", Annotated[np.ndarray, "1D", "int64"])


def is_array1d_of_int64(array: Any) -> TypeGuard[Array1DofInt64]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofInt64`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("int64"))


def be_array1d_of_int64(array: Any) -> Array1DofInt64:
    """
    Assert that an ``array`` is a :py:const:`Array1DofInt64` and return it as such for ``mypy``.
    """
    assert is_array1d_of_int64(array), f"expected a 1D numpy.ndarray of int64, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of uint64 values.
Array1DofUInt64 = NewType("Array1DofUInt64", Annotated[np.ndarray, "1D", "uint64"])


def is_array1d_of_uint64(array: Any) -> TypeGuard[Array1DofUInt64]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofInt64`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("uint64"))


def be_array1d_of_uint64(array: Any) -> Array1DofUInt64:
    """
    Assert that an ``array`` is a :py:const:`Array1DofUInt64` and return it as such for ``mypy``.
    """
    assert is_array1d_of_uint64(array), f"expected a 1D numpy.ndarray of uint64, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of any integer values.
Array1DofInt = Union[
    Array1DofInt8,
    Array1DofUInt8,
    Array1DofInt16,
    Array1DofUInt16,
    Array1DofInt32,
    Array1DofUInt32,
    Array1DofInt64,
    Array1DofUInt64,
]


def is_array1d_of_int(array: Any) -> TypeGuard[Array1DofInt]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofInt`.
    """
    return is_array1d(array) and is_of_dtypes(
        array, ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
    )


def be_array1d_of_int(array: Any) -> Array1DofInt:
    """
    Assert that an ``array`` is a :py:const:`Array1DofInt` and return it as such for ``mypy``.
    """
    assert is_array1d_of_int(array), f"expected a 1D numpy.ndarray of [u]int*, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of float16 values.
Array1DofFloat16 = NewType("Array1DofFloat16", Annotated[np.ndarray, "1D", "float16"])


def is_array1d_of_float16(array: Any) -> TypeGuard[Array1DofFloat16]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofFloat16`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("float16"))


def be_array1d_of_float16(array: Any) -> Array1DofFloat16:
    """
    Assert that an ``array`` is a :py:const:`Array1DofFloat16` and return it as such for ``mypy``.
    """
    assert is_array1d_of_float16(array), f"expected a 1D numpy.ndarray of float16, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of float32 values.
Array1DofFloat32 = NewType("Array1DofFloat32", Annotated[np.ndarray, "1D", "float32"])


def is_array1d_of_float32(array: Any) -> TypeGuard[Array1DofFloat32]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofFloat32`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("float32"))


def be_array1d_of_float32(array: Any) -> Array1DofFloat32:
    """
    Assert that an ``array`` is a :py:const:`Array1DofFloat32` and return it as such for ``mypy``.
    """
    assert is_array1d_of_float32(array), f"expected a 1D numpy.ndarray of float32, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of float64 values.
Array1DofFloat64 = NewType("Array1DofFloat64", Annotated[np.ndarray, "1D", "float64"])


def is_array1d_of_float64(array: Any) -> TypeGuard[Array1DofFloat64]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofFloat64`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("float64"))


def be_array1d_of_float64(array: Any) -> Array1DofFloat64:
    """
    Assert that an ``array`` is a :py:const:`Array1DofFloat64` and return it as such for ``mypy``.
    """
    assert is_array1d_of_float64(array), f"expected a 1D numpy.ndarray of float64, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of any float values.
#:
#: .. todo::
#:
#:    If/when ``numpy`` supports things like ``bfloat16``, and/or ``float8``, add them as well.
Array1DofFloat = Union[Array1DofFloat16, Array1DofFloat32, Array1DofFloat64]


def is_array1d_of_float(array: Any) -> TypeGuard[Array1DofFloat]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofFloat`.
    """
    return is_array1d(array) and is_of_dtypes(array, ("float16", "float32", "float64"))


def be_array1d_of_float(array: Any) -> Array1DofFloat:
    """
    Assert that an ``array`` is a :py:const:`Array1DofFloat` and return it as such for ``mypy``.
    """
    assert is_array1d_of_float(array), f"expected a 1D numpy.ndarray of float*, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of any numeric values.
Array1DofNum = Union[Array1DofInt, Array1DofFloat]


def is_array1d_of_num(array: Any) -> TypeGuard[Array1DofNum]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofNum`.
    """
    return is_array1d(array) and is_of_dtypes(
        array,
        ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64"),
    )


def be_array1d_of_num(array: Any) -> Array1DofNum:
    """
    Assert that an ``array`` is a :py:const:`Array1DofNum` and return it as such for ``mypy``.
    """
    assert is_array1d_of_num(
        array
    ), f"expected a 1D numpy.ndarray of [u]int* or float*, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of any string values.
Array1DofStr = Union[Array1DofInt, Array1DofFloat]


def is_array1d_of_str(array: Any) -> TypeGuard[Array1DofStr]:
    """
    Check whether some ``array`` is a :py:const:`Array1DofStr`.
    """
    return is_array1d(array) and is_of_str(array)


def be_array1d_of_str(array: Any) -> Array1DofStr:
    """
    Assert that an ``array`` is a :py:const:`Array1DofStr` and return it as such for ``mypy``.
    """
    assert is_array1d_of_str(array), f"expected a 1D numpy.ndarray of str, got {_array_description(array)}"
    return array


#: Numpy 1-dimensional array of any (reasonable) data type.
Array1D = Union[Array1DofBool, Array1DofNum, Array1DofStr]


def is_array1d(data: Any) -> TypeGuard[Array1D]:
    """
    Check whether some ``data`` is a :py:const:`Array1D`.
    """
    return isinstance(data, np.ndarray) and data.ndim == 1


def be_array1d(array: Any) -> Array1D:
    """
    Assert that an ``array`` is a :py:const:`Array1D` and return it as such for ``mypy``.
    """
    assert is_array1d(array), f"expected a 1D numpy.ndarray of any, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of bool values.
Array2DofBool = NewType("Array2DofBool", Annotated[np.ndarray, "2D", "bool"])


def is_array2d_of_bool(array: Any) -> TypeGuard[Array2DofBool]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofBool`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("bool"))


def be_array2d_of_bool(array: Any) -> Array2DofBool:
    """
    Assert that an ``array`` is a :py:const:`Array2DofBool` and return it as such for ``mypy``.
    """
    assert is_array2d_of_bool(array), f"expected a 2D numpy.ndarray of bool, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of int8 values.
Array2DofInt8 = NewType("Array2DofInt8", Annotated[np.ndarray, "2D", "int8"])


def is_array2d_of_int8(array: Any) -> TypeGuard[Array2DofInt8]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofInt8`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("int8"))


def be_array2d_of_int8(array: Any) -> Array2DofInt8:
    """
    Assert that an ``array`` is a :py:const:`Array2DofInt8` and return it as such for ``mypy``.
    """
    assert is_array2d_of_int8(array), f"expected a 2D numpy.ndarray of int8, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of uint8 values.
Array2DofUInt8 = NewType("Array2DofUInt8", Annotated[np.ndarray, "2D", "uint8"])


def is_array2d_of_uint8(array: Any) -> TypeGuard[Array2DofUInt8]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofInt8`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("uint8"))


def be_array2d_of_uint8(array: Any) -> Array2DofUInt8:
    """
    Assert that an ``array`` is a :py:const:`Array2DofUInt8` and return it as such for ``mypy``.
    """
    assert is_array2d_of_uint8(array), f"expected a 2D numpy.ndarray of uint8, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of int16 values.
Array2DofInt16 = NewType("Array2DofInt16", Annotated[np.ndarray, "2D", "int16"])


def is_array2d_of_int16(array: Any) -> TypeGuard[Array2DofInt16]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofInt16`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("int16"))


def be_array2d_of_int16(array: Any) -> Array2DofInt16:
    """
    Assert that an ``array`` is a :py:const:`Array2DofInt16` and return it as such for ``mypy``.
    """
    assert is_array2d_of_int16(array), f"expected a 2D numpy.ndarray of int16, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of uint16 values.
Array2DofUInt16 = NewType("Array2DofUInt16", Annotated[np.ndarray, "2D", "uint16"])


def is_array2d_of_uint16(array: Any) -> TypeGuard[Array2DofUInt16]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofInt16`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("uint16"))


def be_array2d_of_uint16(array: Any) -> Array2DofUInt16:
    """
    Assert that an ``array`` is a :py:const:`Array2DofUInt16` and return it as such for ``mypy``.
    """
    assert is_array2d_of_uint16(array), f"expected a 2D numpy.ndarray of uint16, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of int32 values.
Array2DofInt32 = NewType("Array2DofInt32", Annotated[np.ndarray, "2D", "int32"])


def is_array2d_of_int32(array: Any) -> TypeGuard[Array2DofInt32]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofInt32`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("int32"))


def be_array2d_of_int32(array: Any) -> Array2DofInt32:
    """
    Assert that an ``array`` is a :py:const:`Array2DofInt32` and return it as such for ``mypy``.
    """
    assert is_array2d_of_int32(array), f"expected a 2D numpy.ndarray of int32, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of uint32 values.
Array2DofUInt32 = NewType("Array2DofUInt32", Annotated[np.ndarray, "2D", "uint32"])


def is_array2d_of_uint32(array: Any) -> TypeGuard[Array2DofUInt32]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofInt32`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("uint32"))


def be_array2d_of_uint32(array: Any) -> Array2DofUInt32:
    """
    Assert that an ``array`` is a :py:const:`Array2DofUInt32` and return it as such for ``mypy``.
    """
    assert is_array2d_of_uint32(array), f"expected a 2D numpy.ndarray of uint32, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of int64 values.
Array2DofInt64 = NewType("Array2DofInt64", Annotated[np.ndarray, "2D", "int64"])


def is_array2d_of_int64(array: Any) -> TypeGuard[Array2DofInt64]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofInt64`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("int64"))


def be_array2d_of_int64(array: Any) -> Array2DofInt64:
    """
    Assert that an ``array`` is a :py:const:`Array2DofInt64` and return it as such for ``mypy``.
    """
    assert is_array2d_of_int64(array), f"expected a 2D numpy.ndarray of int64, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of uint64 values.
Array2DofUInt64 = NewType("Array2DofUInt64", Annotated[np.ndarray, "2D", "uint64"])


def is_array2d_of_uint64(array: Any) -> TypeGuard[Array2DofUInt64]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofInt64`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("uint64"))


def be_array2d_of_uint64(array: Any) -> Array2DofUInt64:
    """
    Assert that an ``array`` is a :py:const:`Array2DofUInt64` and return it as such for ``mypy``.
    """
    assert is_array2d_of_uint64(array), f"expected a 2D numpy.ndarray of uint64, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of any integer values.
Array2DofInt = Union[
    Array2DofInt8,
    Array2DofUInt8,
    Array2DofInt16,
    Array2DofUInt16,
    Array2DofInt32,
    Array2DofUInt32,
    Array2DofInt64,
    Array2DofUInt64,
]


def is_array2d_of_int(array: Any) -> TypeGuard[Array2DofInt]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofInt`.
    """
    return is_array2d(array) and is_of_dtypes(
        array, ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
    )


def be_array2d_of_int(array: Any) -> Array2DofInt:
    """
    Assert that an ``array`` is a :py:const:`Array2DofInt` and return it as such for ``mypy``.
    """
    assert is_array2d_of_int(array), f"expected a 2D numpy.ndarray of [u]int*, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of float16 values.
Array2DofFloat16 = NewType("Array2DofFloat16", Annotated[np.ndarray, "2D", "float16"])


def is_array2d_of_float16(array: Any) -> TypeGuard[Array2DofFloat16]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofFloat16`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("float16"))


def be_array2d_of_float16(array: Any) -> Array2DofFloat16:
    """
    Assert that an ``array`` is a :py:const:`Array2DofFloat16` and return it as such for ``mypy``.
    """
    assert is_array2d_of_float16(array), f"expected a 2D numpy.ndarray of float16, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of float32 values.
Array2DofFloat32 = NewType("Array2DofFloat32", Annotated[np.ndarray, "2D", "float32"])


def is_array2d_of_float32(array: Any) -> TypeGuard[Array2DofFloat32]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofFloat32`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("float32"))


def be_array2d_of_float32(array: Any) -> Array2DofFloat32:
    """
    Assert that an ``array`` is a :py:const:`Array2DofFloat32` and return it as such for ``mypy``.
    """
    assert is_array2d_of_float32(array), f"expected a 2D numpy.ndarray of float32, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of float64 values.
Array2DofFloat64 = NewType("Array2DofFloat64", Annotated[np.ndarray, "2D", "float64"])


def is_array2d_of_float64(array: Any) -> TypeGuard[Array2DofFloat64]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofFloat64`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("float64"))


def be_array2d_of_float64(array: Any) -> Array2DofFloat64:
    """
    Assert that an ``array`` is a :py:const:`Array2DofFloat64` and return it as such for ``mypy``.
    """
    assert is_array2d_of_float64(array), f"expected a 2D numpy.ndarray of float64, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of any float values.
#:
#: .. todo::
#:
#:    If/when ``numpy`` supports things like ``bfloat16``, and/or ``float8``, add them as well.
Array2DofFloat = Union[Array2DofFloat16, Array2DofFloat32, Array2DofFloat64]


def is_array2d_of_float(array: Any) -> TypeGuard[Array2DofFloat]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofFloat`.
    """
    return is_array2d(array) and is_of_dtypes(array, ("float16", "float32", "float64"))


def be_array2d_of_float(array: Any) -> Array2DofFloat:
    """
    Assert that an ``array`` is a :py:const:`Array2DofFloat` and return it as such for ``mypy``.
    """
    assert is_array2d_of_float(array), f"expected a 2D numpy.ndarray of float*, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of any numeric values.
Array2DofNum = Union[Array2DofInt, Array2DofFloat]


def is_array2d_of_num(array: Any) -> TypeGuard[Array2DofNum]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofNum`.
    """
    return is_array2d(array) and is_of_dtypes(
        array,
        ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64"),
    )


def be_array2d_of_num(array: Any) -> Array2DofNum:
    """
    Assert that an ``array`` is a :py:const:`Array2DofNum` and return it as such for ``mypy``.
    """
    assert is_array2d_of_num(
        array
    ), f"expected a 2D numpy.ndarray of [u]int* or float*, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of any string values.
Array2DofStr = Union[Array2DofInt, Array2DofFloat]


def is_array2d_of_str(array: Any) -> TypeGuard[Array2DofStr]:
    """
    Check whether some ``array`` is a :py:const:`Array2DofStr`.
    """
    return is_array2d(array) and is_of_str(array)


def be_array2d_of_str(array: Any) -> Array2DofStr:
    """
    Assert that an ``array`` is a :py:const:`Array2DofStr` and return it as such for ``mypy``.
    """
    assert is_array2d_of_str(array), f"expected a 2D numpy.ndarray of str, got {_array_description(array)}"
    return array


#: Numpy 2-dimensional array of any (reasonable) data type.
Array2D = Union[Array2DofBool, Array2DofNum, Array2DofStr]


def is_array2d(data: Any) -> TypeGuard[Array2D]:
    """
    Check whether some ``data`` is a :py:const:`Array2D`.

    .. note::

        This explicitly forbids the deprecated data type ``numpy.matrix`` which like a zombie keeps combing back from
        the grave and causes much havoc when it does.
    """
    return isinstance(data, np.ndarray) and data.ndim == 2 and not isinstance(data, np.matrix)


def be_array2d(array: Any) -> Array2D:
    """
    Assert that an ``array`` is a :py:const:`Array2D` and return it as such for ``mypy``.
    """
    assert is_array2d(array), f"expected a 2D numpy.ndarray of any, got {_array_description(array)}"
    return array


def is_of_dtypes(data: Any, dtypes: Collection[str]) -> bool:
    """
    Check whether some ``data`` is a numpy array of one of the ``dtypes``.
    """
    return hasattr(data, "dtype") and str(data.dtype) in dtypes


def is_of_str(data: Any) -> bool:
    """
    Check whether some ``array`` is a numpy array of string values.
    """
    return hasattr(data, "dtype") and "U" in str(data.dtype)


def _array_description(array: Any) -> str:
    if isinstance(array, np.ndarray):
        return f"a {array.ndim}D numpy.ndarray of {array.dtype}"
    return f"an instance of {array.__class__.__module__}.{array.__class__.__qualname__}"
