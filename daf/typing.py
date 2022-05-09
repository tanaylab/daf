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


# pylint: disable=too-many-lines

from typing import Annotated
from typing import Any
from typing import Collection
from typing import NewType
from typing import Optional
from typing import TypeGuard
from typing import Union
from typing import overload

import numpy as np
import pandas as pd  # type: ignore

from .fake_pandas import PandasFrame
from .fake_pandas import PandasSeries

__all__ = [
    # 1D numpy arrays:
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
    # 1D pandas series:
    "Series",
    "is_series",
    "be_series",
    "SeriesOfBool",
    "is_series_of_bool",
    "be_series_of_bool",
    "SeriesOfInt",
    "is_series_of_int",
    "be_series_of_int",
    "SeriesOfInt8",
    "is_series_of_int8",
    "be_series_of_int8",
    "SeriesOfUInt8",
    "is_series_of_uint8",
    "be_series_of_uint8",
    "SeriesOfInt16",
    "is_series_of_int16",
    "be_series_of_int16",
    "SeriesOfUInt16",
    "is_series_of_uint16",
    "be_series_of_uint16",
    "SeriesOfInt32",
    "is_series_of_int32",
    "be_series_of_int32",
    "SeriesOfUInt32",
    "is_series_of_uint32",
    "be_series_of_uint32",
    "SeriesOfInt64",
    "is_series_of_int64",
    "be_series_of_int64",
    "SeriesOfUInt64",
    "is_series_of_uint64",
    "be_series_of_uint64",
    "SeriesOfFloat",
    "is_series_of_float",
    "be_series_of_float",
    "SeriesOfFloat16",
    "is_series_of_float16",
    "be_series_of_float16",
    "SeriesOfFloat32",
    "is_series_of_float32",
    "be_series_of_float32",
    "SeriesOfFloat64",
    "is_series_of_float64",
    "be_series_of_float64",
    "SeriesOfNum",
    "is_series_of_num",
    "be_series_of_num",
    "SeriesOfStr",
    "is_series_of_str",
    "be_series_of_str",
    "to_array1d",
    # 2D numpy arrays:
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
    # 2D pandas frames:
    "Frame",
    "is_frame",
    "be_frame",
    "FrameOfBool",
    "is_frame_of_bool",
    "be_frame_of_bool",
    "FrameOfInt",
    "is_frame_of_int",
    "be_frame_of_int",
    "FrameOfInt8",
    "is_frame_of_int8",
    "be_frame_of_int8",
    "FrameOfUInt8",
    "is_frame_of_uint8",
    "be_frame_of_uint8",
    "FrameOfInt16",
    "is_frame_of_int16",
    "be_frame_of_int16",
    "FrameOfUInt16",
    "is_frame_of_uint16",
    "be_frame_of_uint16",
    "FrameOfInt32",
    "is_frame_of_int32",
    "be_frame_of_int32",
    "FrameOfUInt32",
    "is_frame_of_uint32",
    "be_frame_of_uint32",
    "FrameOfInt64",
    "is_frame_of_int64",
    "be_frame_of_int64",
    "FrameOfUInt64",
    "is_frame_of_uint64",
    "be_frame_of_uint64",
    "FrameOfFloat",
    "is_frame_of_float",
    "be_frame_of_float",
    "FrameOfFloat16",
    "is_frame_of_float16",
    "be_frame_of_float16",
    "FrameOfFloat32",
    "is_frame_of_float32",
    "be_frame_of_float32",
    "FrameOfFloat64",
    "is_frame_of_float64",
    "be_frame_of_float64",
    "FrameOfNum",
    "is_frame_of_num",
    "be_frame_of_num",
    "FrameOfStr",
    "is_frame_of_str",
    "be_frame_of_str",
    "FrameOfAny",
    "is_frame_of_any",
    "be_frame_of_any",
    "FrameOfMany",
    "is_frame_of_many",
    "be_frame_of_many",
    "to_array2d",
]

#: 1-dimensional ``numpy`` array of bool values.
Array1DofBool = NewType("Array1DofBool", Annotated[np.ndarray, "1D", "bool"])


def is_array1d_of_bool(data: Any) -> TypeGuard[Array1DofBool]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofBool`.
    """
    return is_array1d(data, ("bool"))


def be_array1d_of_bool(data: Any) -> Array1DofBool:
    """
    Assert that some ``data`` is an :py:const:`Array1DofBool` and return it as such for ``mypy``.
    """
    assert is_array1d_of_bool(data), f"expected a 1D numpy.ndarray of bool, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of int8 values.
Array1DofInt8 = NewType("Array1DofInt8", Annotated[np.ndarray, "1D", "int8"])


def is_array1d_of_int8(data: Any) -> TypeGuard[Array1DofInt8]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofInt8`.
    """
    return is_array1d(data, ("int8"))


def be_array1d_of_int8(data: Any) -> Array1DofInt8:
    """
    Assert that some ``data`` is an :py:const:`Array1DofInt8` and return it as such for ``mypy``.
    """
    assert is_array1d_of_int8(data), f"expected a 1D numpy.ndarray of int8, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of uint8 values.
Array1DofUInt8 = NewType("Array1DofUInt8", Annotated[np.ndarray, "1D", "uint8"])


def is_array1d_of_uint8(data: Any) -> TypeGuard[Array1DofUInt8]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofInt8`.
    """
    return is_array1d(data, ("uint8"))


def be_array1d_of_uint8(data: Any) -> Array1DofUInt8:
    """
    Assert that some ``data`` is an :py:const:`Array1DofUInt8` and return it as such for ``mypy``.
    """
    assert is_array1d_of_uint8(data), f"expected a 1D numpy.ndarray of uint8, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of int16 values.
Array1DofInt16 = NewType("Array1DofInt16", Annotated[np.ndarray, "1D", "int16"])


def is_array1d_of_int16(data: Any) -> TypeGuard[Array1DofInt16]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofInt16`.
    """
    return is_array1d(data, ("int16"))


def be_array1d_of_int16(data: Any) -> Array1DofInt16:
    """
    Assert that some ``data`` is an :py:const:`Array1DofInt16` and return it as such for ``mypy``.
    """
    assert is_array1d_of_int16(data), f"expected a 1D numpy.ndarray of int16, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of uint16 values.
Array1DofUInt16 = NewType("Array1DofUInt16", Annotated[np.ndarray, "1D", "uint16"])


def is_array1d_of_uint16(data: Any) -> TypeGuard[Array1DofUInt16]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofInt16`.
    """
    return is_array1d(data, ("uint16"))


def be_array1d_of_uint16(data: Any) -> Array1DofUInt16:
    """
    Assert that some ``data`` is an :py:const:`Array1DofUInt16` and return it as such for ``mypy``.
    """
    assert is_array1d_of_uint16(data), f"expected a 1D numpy.ndarray of uint16, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of int32 values.
Array1DofInt32 = NewType("Array1DofInt32", Annotated[np.ndarray, "1D", "int32"])


def is_array1d_of_int32(data: Any) -> TypeGuard[Array1DofInt32]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofInt32`.
    """
    return is_array1d(data, ("int32"))


def be_array1d_of_int32(data: Any) -> Array1DofInt32:
    """
    Assert that some ``data`` is an :py:const:`Array1DofInt32` and return it as such for ``mypy``.
    """
    assert is_array1d_of_int32(data), f"expected a 1D numpy.ndarray of int32, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of uint32 values.
Array1DofUInt32 = NewType("Array1DofUInt32", Annotated[np.ndarray, "1D", "uint32"])


def is_array1d_of_uint32(data: Any) -> TypeGuard[Array1DofUInt32]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofInt32`.
    """
    return is_array1d(data, ("uint32"))


def be_array1d_of_uint32(data: Any) -> Array1DofUInt32:
    """
    Assert that some ``data`` is an :py:const:`Array1DofUInt32` and return it as such for ``mypy``.
    """
    assert is_array1d_of_uint32(data), f"expected a 1D numpy.ndarray of uint32, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of int64 values.
Array1DofInt64 = NewType("Array1DofInt64", Annotated[np.ndarray, "1D", "int64"])


def is_array1d_of_int64(data: Any) -> TypeGuard[Array1DofInt64]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofInt64`.
    """
    return is_array1d(data, ("int64"))


def be_array1d_of_int64(data: Any) -> Array1DofInt64:
    """
    Assert that some ``data`` is an :py:const:`Array1DofInt64` and return it as such for ``mypy``.
    """
    assert is_array1d_of_int64(data), f"expected a 1D numpy.ndarray of int64, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of uint64 values.
Array1DofUInt64 = NewType("Array1DofUInt64", Annotated[np.ndarray, "1D", "uint64"])


def is_array1d_of_uint64(data: Any) -> TypeGuard[Array1DofUInt64]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofInt64`.
    """
    return is_array1d(data, ("uint64"))


def be_array1d_of_uint64(data: Any) -> Array1DofUInt64:
    """
    Assert that some ``data`` is an :py:const:`Array1DofUInt64` and return it as such for ``mypy``.
    """
    assert is_array1d_of_uint64(data), f"expected a 1D numpy.ndarray of uint64, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of any integer values.
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


def is_array1d_of_int(data: Any) -> TypeGuard[Array1DofInt]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofInt`.
    """
    return is_array1d(data, ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"))


def be_array1d_of_int(data: Any) -> Array1DofInt:
    """
    Assert that some ``data`` is an :py:const:`Array1DofInt` and return it as such for ``mypy``.
    """
    assert is_array1d_of_int(data), f"expected a 1D numpy.ndarray of [u]int*, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of float16 values.
Array1DofFloat16 = NewType("Array1DofFloat16", Annotated[np.ndarray, "1D", "float16"])


def is_array1d_of_float16(data: Any) -> TypeGuard[Array1DofFloat16]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofFloat16`.
    """
    return is_array1d(data, ("float16"))


def be_array1d_of_float16(data: Any) -> Array1DofFloat16:
    """
    Assert that some ``data`` is an :py:const:`Array1DofFloat16` and return it as such for ``mypy``.
    """
    assert is_array1d_of_float16(data), f"expected a 1D numpy.ndarray of float16, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of float32 values.
Array1DofFloat32 = NewType("Array1DofFloat32", Annotated[np.ndarray, "1D", "float32"])


def is_array1d_of_float32(data: Any) -> TypeGuard[Array1DofFloat32]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofFloat32`.
    """
    return is_array1d(data, ("float32"))


def be_array1d_of_float32(data: Any) -> Array1DofFloat32:
    """
    Assert that some ``data`` is an :py:const:`Array1DofFloat32` and return it as such for ``mypy``.
    """
    assert is_array1d_of_float32(data), f"expected a 1D numpy.ndarray of float32, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of float64 values.
Array1DofFloat64 = NewType("Array1DofFloat64", Annotated[np.ndarray, "1D", "float64"])


def is_array1d_of_float64(data: Any) -> TypeGuard[Array1DofFloat64]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofFloat64`.
    """
    return is_array1d(data, ("float64"))


def be_array1d_of_float64(data: Any) -> Array1DofFloat64:
    """
    Assert that some ``data`` is an :py:const:`Array1DofFloat64` and return it as such for ``mypy``.
    """
    assert is_array1d_of_float64(data), f"expected a 1D numpy.ndarray of float64, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of any float values.
#:
#: .. todo::
#:
#:    If/when ``numpy`` supports things like ``bfloat16``, and/or ``float8``, add them as well.
Array1DofFloat = Union[Array1DofFloat16, Array1DofFloat32, Array1DofFloat64]


def is_array1d_of_float(data: Any) -> TypeGuard[Array1DofFloat]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofFloat`.
    """
    return is_array1d(data, ("float16", "float32", "float64"))


def be_array1d_of_float(data: Any) -> Array1DofFloat:
    """
    Assert that some ``data`` is an :py:const:`Array1DofFloat` and return it as such for ``mypy``.
    """
    assert is_array1d_of_float(data), f"expected a 1D numpy.ndarray of float*, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of any numeric values.
Array1DofNum = Union[Array1DofInt, Array1DofFloat]


def is_array1d_of_num(data: Any) -> TypeGuard[Array1DofNum]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofNum`.
    """
    return is_array1d(
        data,
        ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64"),
    )


def be_array1d_of_num(data: Any) -> Array1DofNum:
    """
    Assert that some ``data`` is an :py:const:`Array1DofNum` and return it as such for ``mypy``.
    """
    assert is_array1d_of_num(data), f"expected a 1D numpy.ndarray of [u]int* or float*, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of any string values.
Array1DofStr = NewType("Array1DofStr", Annotated[np.ndarray, "1D", "str"])


def is_array1d_of_str(data: Any) -> TypeGuard[Array1DofStr]:
    """
    Check whether some ``data`` is an :py:const:`Array1DofStr`.
    """
    return is_array1d(data) and "U" in str(data.dtype)


def be_array1d_of_str(data: Any) -> Array1DofStr:
    """
    Assert that some ``data`` is an :py:const:`Array1DofStr` and return it as such for ``mypy``.
    """
    assert is_array1d_of_str(data), f"expected a 1D numpy.ndarray of str, got {_array_description(data)}"
    return data


#: 1-dimensional ``numpy`` array of any (reasonable) data type.
Array1D = Union[Array1DofBool, Array1DofNum, Array1DofStr]


def is_array1d(data: Any, dtypes: Optional[Collection[str]] = None) -> TypeGuard[Array1D]:
    """
    Check whether some ``data`` is an :py:const:`Array1D`, optionally only of some ``dtypes``.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        return False
    if dtypes is None:
        return "U" in str(data.dtype) or str(data.dtype) in (
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
    return str(data.dtype) in dtypes


def be_array1d(data: Any, dtypes: Optional[Collection[str]] = None) -> Array1D:
    """
    Assert that some ``data`` is an :py:const:`Array1D`, optionally only of some ``dtypes``, and return it as such for
    ``mypy``.
    """
    if dtypes is None:
        assert is_array1d(data), f"expected a 1D numpy.ndarray of any type, got {_array_description(data)}"
    else:
        assert is_array1d(
            data, dtypes
        ), f"expected a 1D numpy.ndarray of {' or '.join(dtypes)}, got {_array_description(data)}"
    return data


#: 1-dimensional ``pandas`` series of bool values.
SeriesOfBool = NewType("SeriesOfBool", Annotated[PandasSeries, "bool"])


def is_series_of_bool(series: Any) -> TypeGuard[SeriesOfBool]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfBool`.
    """
    return is_series(series, ("bool"))


def be_series_of_bool(series: Any) -> SeriesOfBool:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfBool` and return it as such for ``mypy``.
    """
    assert is_series_of_bool(series), f"expected a pandas.Series of bool, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of int8 values.
SeriesOfInt8 = NewType("SeriesOfInt8", Annotated[PandasSeries, "int8"])


def is_series_of_int8(series: Any) -> TypeGuard[SeriesOfInt8]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfInt8`.
    """
    return is_series(series, ("int8"))


def be_series_of_int8(series: Any) -> SeriesOfInt8:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfInt8` and return it as such for ``mypy``.
    """
    assert is_series_of_int8(series), f"expected a pandas.Series of int8, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of uint8 values.
SeriesOfUInt8 = NewType("SeriesOfUInt8", Annotated[PandasSeries, "uint8"])


def is_series_of_uint8(series: Any) -> TypeGuard[SeriesOfUInt8]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfInt8`.
    """
    return is_series(series, ("uint8"))


def be_series_of_uint8(series: Any) -> SeriesOfUInt8:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfUInt8` and return it as such for ``mypy``.
    """
    assert is_series_of_uint8(series), f"expected a pandas.Series of uint8, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of int16 values.
SeriesOfInt16 = NewType("SeriesOfInt16", Annotated[PandasSeries, "int16"])


def is_series_of_int16(series: Any) -> TypeGuard[SeriesOfInt16]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfInt16`.
    """
    return is_series(series, ("int16"))


def be_series_of_int16(series: Any) -> SeriesOfInt16:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfInt16` and return it as such for ``mypy``.
    """
    assert is_series_of_int16(series), f"expected a pandas.Series of int16, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of uint16 values.
SeriesOfUInt16 = NewType("SeriesOfUInt16", Annotated[PandasSeries, "uint16"])


def is_series_of_uint16(series: Any) -> TypeGuard[SeriesOfUInt16]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfInt16`.
    """
    return is_series(series, ("uint16"))


def be_series_of_uint16(series: Any) -> SeriesOfUInt16:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfUInt16` and return it as such for ``mypy``.
    """
    assert is_series_of_uint16(series), f"expected a pandas.Series of uint16, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of int32 values.
SeriesOfInt32 = NewType("SeriesOfInt32", Annotated[PandasSeries, "int32"])


def is_series_of_int32(series: Any) -> TypeGuard[SeriesOfInt32]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfInt32`.
    """
    return is_series(series, ("int32"))


def be_series_of_int32(series: Any) -> SeriesOfInt32:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfInt32` and return it as such for ``mypy``.
    """
    assert is_series_of_int32(series), f"expected a pandas.Series of int32, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of uint32 values.
SeriesOfUInt32 = NewType("SeriesOfUInt32", Annotated[PandasSeries, "uint32"])


def is_series_of_uint32(series: Any) -> TypeGuard[SeriesOfUInt32]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfInt32`.
    """
    return is_series(series, ("uint32"))


def be_series_of_uint32(series: Any) -> SeriesOfUInt32:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfUInt32` and return it as such for ``mypy``.
    """
    assert is_series_of_uint32(series), f"expected a pandas.Series of uint32, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of int64 values.
SeriesOfInt64 = NewType("SeriesOfInt64", Annotated[PandasSeries, "int64"])


def is_series_of_int64(series: Any) -> TypeGuard[SeriesOfInt64]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfInt64`.
    """
    return is_series(series, ("int64"))


def be_series_of_int64(series: Any) -> SeriesOfInt64:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfInt64` and return it as such for ``mypy``.
    """
    assert is_series_of_int64(series), f"expected a pandas.Series of int64, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of uint64 values.
SeriesOfUInt64 = NewType("SeriesOfUInt64", Annotated[PandasSeries, "uint64"])


def is_series_of_uint64(series: Any) -> TypeGuard[SeriesOfUInt64]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfInt64`.
    """
    return is_series(series, ("uint64"))


def be_series_of_uint64(series: Any) -> SeriesOfUInt64:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfUInt64` and return it as such for ``mypy``.
    """
    assert is_series_of_uint64(series), f"expected a pandas.Series of uint64, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of any integer values.
SeriesOfInt = Union[
    SeriesOfInt8,
    SeriesOfUInt8,
    SeriesOfInt16,
    SeriesOfUInt16,
    SeriesOfInt32,
    SeriesOfUInt32,
    SeriesOfInt64,
    SeriesOfUInt64,
]


def is_series_of_int(series: Any) -> TypeGuard[SeriesOfInt]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfInt`.
    """
    return is_series(series, ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"))


def be_series_of_int(series: Any) -> SeriesOfInt:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfInt` and return it as such for ``mypy``.
    """
    assert is_series_of_int(series), f"expected a pandas.Series of [u]int*, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of float16 values.
SeriesOfFloat16 = NewType("SeriesOfFloat16", Annotated[PandasSeries, "float16"])


def is_series_of_float16(series: Any) -> TypeGuard[SeriesOfFloat16]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfFloat16`.
    """
    return is_series(series, ("float16"))


def be_series_of_float16(series: Any) -> SeriesOfFloat16:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfFloat16` and return it as such for ``mypy``.
    """
    assert is_series_of_float16(series), f"expected a pandas.Series of float16, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of float32 values.
SeriesOfFloat32 = NewType("SeriesOfFloat32", Annotated[PandasSeries, "float32"])


def is_series_of_float32(series: Any) -> TypeGuard[SeriesOfFloat32]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfFloat32`.
    """
    return is_series(series, ("float32"))


def be_series_of_float32(series: Any) -> SeriesOfFloat32:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfFloat32` and return it as such for ``mypy``.
    """
    assert is_series_of_float32(series), f"expected a pandas.Series of float32, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of float64 values.
SeriesOfFloat64 = NewType("SeriesOfFloat64", Annotated[PandasSeries, "float64"])


def is_series_of_float64(series: Any) -> TypeGuard[SeriesOfFloat64]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfFloat64`.
    """
    return is_series(series, ("float64"))


def be_series_of_float64(series: Any) -> SeriesOfFloat64:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfFloat64` and return it as such for ``mypy``.
    """
    assert is_series_of_float64(series), f"expected a pandas.Series of float64, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of any float values.
#:
#: .. todo::
#:
#:    If/when ``numpy`` supports things like ``bfloat16``, and/or ``float8``, add them as well.
SeriesOfFloat = Union[SeriesOfFloat16, SeriesOfFloat32, SeriesOfFloat64]


def is_series_of_float(series: Any) -> TypeGuard[SeriesOfFloat]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfFloat`.
    """
    return is_series(series, ("float16", "float32", "float64"))


def be_series_of_float(series: Any) -> SeriesOfFloat:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfFloat` and return it as such for ``mypy``.
    """
    assert is_series_of_float(series), f"expected a pandas.Series of float*, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of any numeric values.
SeriesOfNum = Union[SeriesOfInt, SeriesOfFloat]


def is_series_of_num(series: Any) -> TypeGuard[SeriesOfNum]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfNum`.
    """
    return is_series(
        series,
        ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64"),
    )


def be_series_of_num(series: Any) -> SeriesOfNum:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfNum` and return it as such for ``mypy``.
    """
    assert is_series_of_num(series), f"expected a pandas.Series of [u]int* or float*, got {_series_description(series)}"
    return series


#: 1-dimensional ``pandas`` series of any string values.
SeriesOfStr = NewType("SeriesOfStr", Annotated[PandasSeries, "str"])


def is_series_of_str(series: Any) -> TypeGuard[SeriesOfStr]:
    """
    Check whether some ``series`` is a :py:const:`SeriesOfStr`.
    """
    return is_series(series) and str(series.dtype) in ("string", "category")


def be_series_of_str(series: Any) -> SeriesOfStr:
    """
    Assert that some ``series`` is a :py:const:`SeriesOfStr` and return it as such for ``mypy``.
    """
    assert is_series_of_str(series), f"expected a pandas.Series of str, got {_series_description(series)}"
    return series


@overload
def to_array1d(series: SeriesOfBool) -> Array1DofBool:
    ...


@overload
def to_array1d(series: SeriesOfInt8) -> Array1DofInt8:
    ...


@overload
def to_array1d(series: SeriesOfUInt8) -> Array1DofUInt8:
    ...


@overload
def to_array1d(series: SeriesOfInt16) -> Array1DofInt16:
    ...


@overload
def to_array1d(series: SeriesOfUInt16) -> Array1DofUInt16:
    ...


@overload
def to_array1d(series: SeriesOfInt32) -> Array1DofInt32:
    ...


@overload
def to_array1d(series: SeriesOfUInt32) -> Array1DofUInt32:
    ...


@overload
def to_array1d(series: SeriesOfInt64) -> Array1DofInt64:
    ...


@overload
def to_array1d(series: SeriesOfUInt64) -> Array1DofUInt64:
    ...


@overload
def to_array1d(series: SeriesOfFloat16) -> Array1DofFloat16:
    ...


@overload
def to_array1d(series: SeriesOfFloat32) -> Array1DofFloat32:
    ...


@overload
def to_array1d(series: SeriesOfFloat64) -> Array1DofFloat64:
    ...


@overload
def to_array1d(series: SeriesOfStr) -> Array1DofStr:
    ...


def to_array1d(series: Any) -> Any:
    """
    Access the internal 1D ``numpy`` array of a ``pandas`` series.

    You would think it is sufficient to just access the ``.values`` member. You'd be wrong in case the pandas series
    contains categorical data.
    """
    array1d = series.values

    series_dtype = str(series.dtype)
    if series_dtype in ("string", "category"):
        array1d_dtype = "U"
    else:
        array1d_dtype = series_dtype

    if array1d_dtype != series_dtype or not isinstance(array1d, np.ndarray):
        array1d = np.array(array1d, array1d_dtype)

    return array1d


#: 1-dimensional ``pandas`` series of any (reasonable) data type.
Series = Union[SeriesOfBool, SeriesOfNum, SeriesOfStr]


def is_series(data: Any, dtypes: Optional[Collection[str]] = None) -> TypeGuard[Series]:
    """
    Check whether some ``data`` is a :py:const:`Series`, optionally only of some ``dtypes``.
    """
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
            "string",
            "category",
        )
    return isinstance(data, pd.Series) and str(data.dtype) in dtypes


def be_series(series: Any, dtypes: Optional[Collection[str]] = None) -> Series:
    """
    Assert that some ``series`` is a :py:const:`Series`, optionally only of some ``dtypes``, and return it as such for
    ``mypy``.
    """
    if dtypes is None:
        assert is_series(series), f"expected a pandas.Series of any reasonable type, got {_series_description(series)}"
    else:
        assert is_series(
            series
        ), f"expected a pandas.Series of {' or '.join(dtypes)}, got {_series_description(series)}"
    return series


#: 2-dimensional ``pandas`` series of bool values.
Array2DofBool = NewType("Array2DofBool", Annotated[np.ndarray, "2D", "bool"])


def is_array2d_of_bool(data: Any) -> TypeGuard[Array2DofBool]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofBool`.
    """
    return is_array2d(data, ("bool"))


def be_array2d_of_bool(data: Any) -> Array2DofBool:
    """
    Assert that some ``data`` is an :py:const:`Array2DofBool` and return it as such for ``mypy``.
    """
    assert is_array2d_of_bool(data), f"expected a 2D numpy.ndarray of bool, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of int8 values.
Array2DofInt8 = NewType("Array2DofInt8", Annotated[np.ndarray, "2D", "int8"])


def is_array2d_of_int8(data: Any) -> TypeGuard[Array2DofInt8]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofInt8`.
    """
    return is_array2d(data, ("int8"))


def be_array2d_of_int8(data: Any) -> Array2DofInt8:
    """
    Assert that some ``data`` is an :py:const:`Array2DofInt8` and return it as such for ``mypy``.
    """
    assert is_array2d_of_int8(data), f"expected a 2D numpy.ndarray of int8, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of uint8 values.
Array2DofUInt8 = NewType("Array2DofUInt8", Annotated[np.ndarray, "2D", "uint8"])


def is_array2d_of_uint8(data: Any) -> TypeGuard[Array2DofUInt8]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofInt8`.
    """
    return is_array2d(data, ("uint8"))


def be_array2d_of_uint8(data: Any) -> Array2DofUInt8:
    """
    Assert that some ``data`` is an :py:const:`Array2DofUInt8` and return it as such for ``mypy``.
    """
    assert is_array2d_of_uint8(data), f"expected a 2D numpy.ndarray of uint8, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of int16 values.
Array2DofInt16 = NewType("Array2DofInt16", Annotated[np.ndarray, "2D", "int16"])


def is_array2d_of_int16(data: Any) -> TypeGuard[Array2DofInt16]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofInt16`.
    """
    return is_array2d(data, ("int16"))


def be_array2d_of_int16(data: Any) -> Array2DofInt16:
    """
    Assert that some ``data`` is an :py:const:`Array2DofInt16` and return it as such for ``mypy``.
    """
    assert is_array2d_of_int16(data), f"expected a 2D numpy.ndarray of int16, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of uint16 values.
Array2DofUInt16 = NewType("Array2DofUInt16", Annotated[np.ndarray, "2D", "uint16"])


def is_array2d_of_uint16(data: Any) -> TypeGuard[Array2DofUInt16]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofInt16`.
    """
    return is_array2d(data, ("uint16"))


def be_array2d_of_uint16(data: Any) -> Array2DofUInt16:
    """
    Assert that some ``data`` is an :py:const:`Array2DofUInt16` and return it as such for ``mypy``.
    """
    assert is_array2d_of_uint16(data), f"expected a 2D numpy.ndarray of uint16, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of int32 values.
Array2DofInt32 = NewType("Array2DofInt32", Annotated[np.ndarray, "2D", "int32"])


def is_array2d_of_int32(data: Any) -> TypeGuard[Array2DofInt32]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofInt32`.
    """
    return is_array2d(data, ("int32"))


def be_array2d_of_int32(data: Any) -> Array2DofInt32:
    """
    Assert that some ``data`` is an :py:const:`Array2DofInt32` and return it as such for ``mypy``.
    """
    assert is_array2d_of_int32(data), f"expected a 2D numpy.ndarray of int32, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of uint32 values.
Array2DofUInt32 = NewType("Array2DofUInt32", Annotated[np.ndarray, "2D", "uint32"])


def is_array2d_of_uint32(data: Any) -> TypeGuard[Array2DofUInt32]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofInt32`.
    """
    return is_array2d(data, ("uint32"))


def be_array2d_of_uint32(data: Any) -> Array2DofUInt32:
    """
    Assert that some ``data`` is an :py:const:`Array2DofUInt32` and return it as such for ``mypy``.
    """
    assert is_array2d_of_uint32(data), f"expected a 2D numpy.ndarray of uint32, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of int64 values.
Array2DofInt64 = NewType("Array2DofInt64", Annotated[np.ndarray, "2D", "int64"])


def is_array2d_of_int64(data: Any) -> TypeGuard[Array2DofInt64]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofInt64`.
    """
    return is_array2d(data, ("int64"))


def be_array2d_of_int64(data: Any) -> Array2DofInt64:
    """
    Assert that some ``data`` is an :py:const:`Array2DofInt64` and return it as such for ``mypy``.
    """
    assert is_array2d_of_int64(data), f"expected a 2D numpy.ndarray of int64, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of uint64 values.
Array2DofUInt64 = NewType("Array2DofUInt64", Annotated[np.ndarray, "2D", "uint64"])


def is_array2d_of_uint64(data: Any) -> TypeGuard[Array2DofUInt64]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofInt64`.
    """
    return is_array2d(data, ("uint64"))


def be_array2d_of_uint64(data: Any) -> Array2DofUInt64:
    """
    Assert that some ``data`` is an :py:const:`Array2DofUInt64` and return it as such for ``mypy``.
    """
    assert is_array2d_of_uint64(data), f"expected a 2D numpy.ndarray of uint64, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of any integer values.
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


def is_array2d_of_int(data: Any) -> TypeGuard[Array2DofInt]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofInt`.
    """
    return is_array2d(data, ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"))


def be_array2d_of_int(data: Any) -> Array2DofInt:
    """
    Assert that some ``data`` is an :py:const:`Array2DofInt` and return it as such for ``mypy``.
    """
    assert is_array2d_of_int(data), f"expected a 2D numpy.ndarray of [u]int*, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of float16 values.
Array2DofFloat16 = NewType("Array2DofFloat16", Annotated[np.ndarray, "2D", "float16"])


def is_array2d_of_float16(data: Any) -> TypeGuard[Array2DofFloat16]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofFloat16`.
    """
    return is_array2d(data, ("float16"))


def be_array2d_of_float16(data: Any) -> Array2DofFloat16:
    """
    Assert that some ``data`` is an :py:const:`Array2DofFloat16` and return it as such for ``mypy``.
    """
    assert is_array2d_of_float16(data), f"expected a 2D numpy.ndarray of float16, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of float32 values.
Array2DofFloat32 = NewType("Array2DofFloat32", Annotated[np.ndarray, "2D", "float32"])


def is_array2d_of_float32(data: Any) -> TypeGuard[Array2DofFloat32]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofFloat32`.
    """
    return is_array2d(data, ("float32"))


def be_array2d_of_float32(data: Any) -> Array2DofFloat32:
    """
    Assert that some ``data`` is an :py:const:`Array2DofFloat32` and return it as such for ``mypy``.
    """
    assert is_array2d_of_float32(data), f"expected a 2D numpy.ndarray of float32, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of float64 values.
Array2DofFloat64 = NewType("Array2DofFloat64", Annotated[np.ndarray, "2D", "float64"])


def is_array2d_of_float64(data: Any) -> TypeGuard[Array2DofFloat64]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofFloat64`.
    """
    return is_array2d(data, ("float64"))


def be_array2d_of_float64(data: Any) -> Array2DofFloat64:
    """
    Assert that some ``data`` is an :py:const:`Array2DofFloat64` and return it as such for ``mypy``.
    """
    assert is_array2d_of_float64(data), f"expected a 2D numpy.ndarray of float64, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of any float values.
#:
#: .. todo::
#:
#:    If/when ``numpy`` supports things like ``bfloat16``, and/or ``float8``, add them as well.
Array2DofFloat = Union[Array2DofFloat16, Array2DofFloat32, Array2DofFloat64]


def is_array2d_of_float(data: Any) -> TypeGuard[Array2DofFloat]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofFloat`.
    """
    return is_array2d(data, ("float16", "float32", "float64"))


def be_array2d_of_float(data: Any) -> Array2DofFloat:
    """
    Assert that some ``data`` is an :py:const:`Array2DofFloat` and return it as such for ``mypy``.
    """
    assert is_array2d_of_float(data), f"expected a 2D numpy.ndarray of float*, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of any numeric values.
Array2DofNum = Union[Array2DofInt, Array2DofFloat]


def is_array2d_of_num(data: Any) -> TypeGuard[Array2DofNum]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofNum`.
    """
    return is_array2d(
        data,
        ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64"),
    )


def be_array2d_of_num(data: Any) -> Array2DofNum:
    """
    Assert that some ``data`` is an :py:const:`Array2DofNum` and return it as such for ``mypy``.
    """
    assert is_array2d_of_num(data), f"expected a 2D numpy.ndarray of [u]int* or float*, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of any string values.
Array2DofStr = NewType("Array2DofStr", Annotated[np.ndarray, "2D", "str"])


def is_array2d_of_str(data: Any) -> TypeGuard[Array2DofStr]:
    """
    Check whether some ``data`` is an :py:const:`Array2DofStr`.
    """
    return is_array2d(data) and "U" in str(data.dtype)


def be_array2d_of_str(data: Any) -> Array2DofStr:
    """
    Assert that some ``data`` is an :py:const:`Array2DofStr` and return it as such for ``mypy``.
    """
    assert is_array2d_of_str(data), f"expected a 2D numpy.ndarray of str, got {_array_description(data)}"
    return data


#: 2-dimensional ``numpy`` array of any (reasonable) data type.
Array2D = Union[Array2DofBool, Array2DofNum, Array2DofStr]


def is_array2d(data: Any, dtypes: Optional[Collection[str]] = None) -> TypeGuard[Array2D]:
    """
    Check whether some ``data`` is an :py:const:`Array2D`, optionally only of some ``dtypes``.

    .. note::

        This explicitly forbids the deprecated data type ``numpy.matrix`` which like a zombie keeps combing back from
        the grave and causes much havoc when it does.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2 or isinstance(data, np.matrix):
        return False
    if dtypes is None:
        return "U" in str(data.dtype) or str(data.dtype) in (
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
    return str(data.dtype) in dtypes


def be_array2d(data: Any, dtypes: Optional[Collection[str]] = None) -> Array2D:
    """
    Assert that some ``data`` is an :py:const:`Array2D` optionally only of some ``dtypes``, and return it as such for
    ``mypy``.
    """
    if dtypes is None:
        assert is_array2d(data), f"expected a 2D numpy.ndarray of any reasonable type, got {_array_description(data)}"
    else:
        assert is_array2d(data), f"expected a 2D numpy.ndarray of {' or '.join(dtypes)}, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of bool values.
FrameOfBool = NewType("FrameOfBool", Annotated[PandasFrame, "bool"])


def is_frame_of_bool(data: Any) -> TypeGuard[FrameOfBool]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfBool`.
    """
    return is_frame(data, ("bool"))


def be_frame_of_bool(data: Any) -> FrameOfBool:
    """
    Assert that some ``data`` is a :py:const:`FrameOfBool` and return it as such for ``mypy``.
    """
    assert is_frame_of_bool(data), f"expected a panda.DataFrame of bool, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of int8 values.
FrameOfInt8 = NewType("FrameOfInt8", Annotated[PandasFrame, "int8"])


def is_frame_of_int8(data: Any) -> TypeGuard[FrameOfInt8]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfInt8`.
    """
    return is_frame(data, ("int8"))


def be_frame_of_int8(data: Any) -> FrameOfInt8:
    """
    Assert that some ``data`` is a :py:const:`FrameOfInt8` and return it as such for ``mypy``.
    """
    assert is_frame_of_int8(data), f"expected a panda.DataFrame of int8, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of uint8 values.
FrameOfUInt8 = NewType("FrameOfUInt8", Annotated[PandasFrame, "uint8"])


def is_frame_of_uint8(data: Any) -> TypeGuard[FrameOfUInt8]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfInt8`.
    """
    return is_frame(data, ("uint8"))


def be_frame_of_uint8(data: Any) -> FrameOfUInt8:
    """
    Assert that some ``data`` is a :py:const:`FrameOfUInt8` and return it as such for ``mypy``.
    """
    assert is_frame_of_uint8(data), f"expected a panda.DataFrame of uint8, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of int16 values.
FrameOfInt16 = NewType("FrameOfInt16", Annotated[PandasFrame, "int16"])


def is_frame_of_int16(data: Any) -> TypeGuard[FrameOfInt16]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfInt16`.
    """
    return is_frame(data, ("int16"))


def be_frame_of_int16(data: Any) -> FrameOfInt16:
    """
    Assert that some ``data`` is a :py:const:`FrameOfInt16` and return it as such for ``mypy``.
    """
    assert is_frame_of_int16(data), f"expected a panda.DataFrame of int16, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of uint16 values.
FrameOfUInt16 = NewType("FrameOfUInt16", Annotated[PandasFrame, "uint16"])


def is_frame_of_uint16(data: Any) -> TypeGuard[FrameOfUInt16]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfInt16`.
    """
    return is_frame(data, ("uint16"))


def be_frame_of_uint16(data: Any) -> FrameOfUInt16:
    """
    Assert that some ``data`` is a :py:const:`FrameOfUInt16` and return it as such for ``mypy``.
    """
    assert is_frame_of_uint16(data), f"expected a panda.DataFrame of uint16, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of int32 values.
FrameOfInt32 = NewType("FrameOfInt32", Annotated[PandasFrame, "int32"])


def is_frame_of_int32(data: Any) -> TypeGuard[FrameOfInt32]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfInt32`.
    """
    return is_frame(data, ("int32"))


def be_frame_of_int32(data: Any) -> FrameOfInt32:
    """
    Assert that some ``data`` is a :py:const:`FrameOfInt32` and return it as such for ``mypy``.
    """
    assert is_frame_of_int32(data), f"expected a panda.DataFrame of int32, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of uint32 values.
FrameOfUInt32 = NewType("FrameOfUInt32", Annotated[PandasFrame, "uint32"])


def is_frame_of_uint32(data: Any) -> TypeGuard[FrameOfUInt32]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfInt32`.
    """
    return is_frame(data, ("uint32"))


def be_frame_of_uint32(data: Any) -> FrameOfUInt32:
    """
    Assert that some ``data`` is a :py:const:`FrameOfUInt32` and return it as such for ``mypy``.
    """
    assert is_frame_of_uint32(data), f"expected a panda.DataFrame of uint32, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of int64 values.
FrameOfInt64 = NewType("FrameOfInt64", Annotated[PandasFrame, "int64"])


def is_frame_of_int64(data: Any) -> TypeGuard[FrameOfInt64]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfInt64`.
    """
    return is_frame(data, ("int64"))


def be_frame_of_int64(data: Any) -> FrameOfInt64:
    """
    Assert that some ``data`` is a :py:const:`FrameOfInt64` and return it as such for ``mypy``.
    """
    assert is_frame_of_int64(data), f"expected a panda.DataFrame of int64, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of uint64 values.
FrameOfUInt64 = NewType("FrameOfUInt64", Annotated[PandasFrame, "uint64"])


def is_frame_of_uint64(data: Any) -> TypeGuard[FrameOfUInt64]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfInt64`.
    """
    return is_frame(data, ("uint64"))


def be_frame_of_uint64(data: Any) -> FrameOfUInt64:
    """
    Assert that some ``data`` is a :py:const:`FrameOfUInt64` and return it as such for ``mypy``.
    """
    assert is_frame_of_uint64(data), f"expected a panda.DataFrame of uint64, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of any integer values.
FrameOfInt = Union[
    FrameOfInt8,
    FrameOfUInt8,
    FrameOfInt16,
    FrameOfUInt16,
    FrameOfInt32,
    FrameOfUInt32,
    FrameOfInt64,
    FrameOfUInt64,
]


def is_frame_of_int(data: Any) -> TypeGuard[FrameOfInt]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfInt`.
    """
    return is_frame(data, ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"))


def be_frame_of_int(data: Any) -> FrameOfInt:
    """
    Assert that some ``data`` is a :py:const:`FrameOfInt` and return it as such for ``mypy``.
    """
    assert is_frame_of_int(data), f"expected a panda.DataFrame of [u]int*, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of float16 values.
FrameOfFloat16 = NewType("FrameOfFloat16", Annotated[PandasFrame, "float16"])


def is_frame_of_float16(data: Any) -> TypeGuard[FrameOfFloat16]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfFloat16`.
    """
    return is_frame(data, ("float16"))


def be_frame_of_float16(data: Any) -> FrameOfFloat16:
    """
    Assert that some ``data`` is a :py:const:`FrameOfFloat16` and return it as such for ``mypy``.
    """
    assert is_frame_of_float16(data), f"expected a panda.DataFrame of float16, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of float32 values.
FrameOfFloat32 = NewType("FrameOfFloat32", Annotated[PandasFrame, "float32"])


def is_frame_of_float32(data: Any) -> TypeGuard[FrameOfFloat32]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfFloat32`.
    """
    return is_frame(data, ("float32"))


def be_frame_of_float32(data: Any) -> FrameOfFloat32:
    """
    Assert that some ``data`` is a :py:const:`FrameOfFloat32` and return it as such for ``mypy``.
    """
    assert is_frame_of_float32(data), f"expected a panda.DataFrame of float32, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of float64 values.
FrameOfFloat64 = NewType("FrameOfFloat64", Annotated[PandasFrame, "float64"])


def is_frame_of_float64(data: Any) -> TypeGuard[FrameOfFloat64]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfFloat64`.
    """
    return is_frame(data, ("float64"))


def be_frame_of_float64(data: Any) -> FrameOfFloat64:
    """
    Assert that some ``data`` is a :py:const:`FrameOfFloat64` and return it as such for ``mypy``.
    """
    assert is_frame_of_float64(data), f"expected a panda.DataFrame of float64, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of any float values.
#:
#: .. todo::
#:
#:    If/when ``pandas`` supports things like ``bfloat16``, and/or ``float8``, add them as well.
FrameOfFloat = Union[FrameOfFloat16, FrameOfFloat32, FrameOfFloat64]


def is_frame_of_float(data: Any) -> TypeGuard[FrameOfFloat]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfFloat`.
    """
    return is_frame(data, ("float16", "float32", "float64"))


def be_frame_of_float(data: Any) -> FrameOfFloat:
    """
    Assert that some ``data`` is a :py:const:`FrameOfFloat` and return it as such for ``mypy``.
    """
    assert is_frame_of_float(data), f"expected a panda.DataFrame of float*, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame  of any numeric values.
FrameOfNum = Union[FrameOfInt, FrameOfFloat]


def is_frame_of_num(data: Any) -> TypeGuard[FrameOfNum]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfNum`.
    """
    return is_frame(
        data,
        ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64"),
    )


def be_frame_of_num(data: Any) -> FrameOfNum:
    """
    Assert that some ``data`` is a :py:const:`FrameOfNum` and return it as such for ``mypy``.
    """
    assert is_frame_of_num(data), f"expected a panda.DataFrame of [u]int* or float*, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame  of any string values.
FrameOfStr = NewType("FrameOfStr", Annotated[PandasFrame, "str"])


def is_frame_of_str(data: Any) -> TypeGuard[FrameOfStr]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfStr`.
    """
    if not is_frame(data):
        return False
    for dtype in data.dtypes:
        if dtype not in ("string", "category"):
            return False
    return True


def be_frame_of_str(data: Any) -> FrameOfStr:
    """
    Assert that some ``data`` is a :py:const:`FrameOfStr` and return it as such for ``mypy``.
    """
    assert is_frame_of_str(data), f"expected a panda.DataFrame of str, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of values of a single type in all columns.
FrameOfAny = Union[FrameOfBool, FrameOfNum, FrameOfStr]


def is_frame_of_any(data: Any) -> TypeGuard[FrameOfAny]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfAny`.
    """
    return is_frame(
        data,
        (
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
        ),
    ) or is_frame_of_str(data)


def be_frame_of_any(data: Any) -> FrameOfAny:
    """
    Assert that some ``data`` is a :py:const:`FrameOfAny` and return it as such for ``mypy``.
    """
    assert is_frame_of_any(data), f"expected a panda.DataFrame of any reasonable type, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame of values of many types in different columns.
FrameOfMany = NewType("FrameOfMany", Annotated[PandasFrame, "many"])


def is_frame_of_many(data: Any) -> TypeGuard[FrameOfMany]:
    """
    Check whether some ``data`` is a :py:const:`FrameOfMany`.
    """
    return is_frame(data)


def be_frame_of_many(data: Any) -> FrameOfMany:
    """
    Assert that some ``data`` is a :py:const:`FrameOfMany` and return it as such for ``mypy``.
    """
    assert is_frame_of_many(data), f"expected a panda.DataFrame of (m)any type, got {_array_description(data)}"
    return data


#: 2-dimensional ``pandas`` frame  of any (reasonable) data type.
Frame = Union[FrameOfAny, FrameOfMany]


def is_frame(data: Any, dtypes: Optional[Collection[str]] = None) -> TypeGuard[Frame]:
    """
    Check whether some ``data`` is a :py:const:`Frame`, optionally only of some ``dtypes``.

    .. note::

        This explicitly forbids the deprecated data type ``numpy.matrix`` which like a zombie keeps combing back from
        the grave and causes much havoc when it does.
    """
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
            "string",
            "category",
            "object",
        )
    if not isinstance(data, pd.DataFrame):
        return False
    for dtype in data.dtypes:
        if str(dtype) not in dtypes:
            return False
    return True


def be_frame(data: Any, dtypes: Optional[Collection[str]] = None) -> Frame:
    """
    Assert that some ``data`` is a :py:const:`Frame` optionally only of some ``dtypes``, and return it as such for
    ``mypy``.
    """
    if dtypes is None:
        assert is_frame(data), f"expected a panda.DataFrame of (m)any type, got {_array_description(data)}"
    else:
        assert is_frame(data), f"expected a pandas.DataFrame of {' or '.join(dtypes)}, got {_array_description(data)}"
    return data


@overload
def to_array2d(frame: FrameOfBool) -> Array2DofBool:
    ...


@overload
def to_array2d(frame: FrameOfInt8) -> Array2DofInt8:
    ...


@overload
def to_array2d(frame: FrameOfUInt8) -> Array2DofUInt8:
    ...


@overload
def to_array2d(frame: FrameOfInt16) -> Array2DofInt16:
    ...


@overload
def to_array2d(frame: FrameOfUInt16) -> Array2DofUInt16:
    ...


@overload
def to_array2d(frame: FrameOfInt32) -> Array2DofInt32:
    ...


@overload
def to_array2d(frame: FrameOfUInt32) -> Array2DofUInt32:
    ...


@overload
def to_array2d(frame: FrameOfInt64) -> Array2DofInt64:
    ...


@overload
def to_array2d(frame: FrameOfUInt64) -> Array2DofUInt64:
    ...


@overload
def to_array2d(frame: FrameOfFloat16) -> Array2DofFloat16:
    ...


@overload
def to_array2d(frame: FrameOfFloat32) -> Array2DofFloat32:
    ...


@overload
def to_array2d(frame: FrameOfFloat64) -> Array2DofFloat64:
    ...


@overload
def to_array2d(frame: FrameOfStr) -> Array2DofStr:
    ...


def to_array2d(frame: Any) -> Any:
    """
    Access the internal 2D ``numpy`` array of a ``pandas`` frame.

    You would think it is sufficient to just access the ``.values`` member. You'd be wrong in case the pandas frame
    contains categorical data.
    """
    array2d = frame.values

    if is_frame_of_str(frame):
        array2d = np.array(array2d, "U")
    elif not isinstance(array2d, np.ndarray) or isinstance(array2d, np.matrix):
        array2d = np.array(array2d)

    return array2d


def _array_description(data: Any) -> str:
    if isinstance(data, np.ndarray):
        return f"a {data.ndim}D numpy.ndarray of {data.dtype}"
    return f"an instance of {data.__class__.__module__}.{data.__class__.__qualname__}"


def _series_description(data: Any) -> str:
    if isinstance(data, pd.Series):
        return f"a pandas.Series of {data.dtype}"
    return f"an instance of {data.__class__.__module__}.{data.__class__.__qualname__}"


def _frame_description(data: Any) -> str:
    if isinstance(data, pd.DataFrame):
        return f"a pandas.DataFrame of {data.dtype}"
    return f"an instance of {data.__class__.__module__}.{data.__class__.__qualname__}"
