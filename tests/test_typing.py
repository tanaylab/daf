"""
Test typing.
"""

import numpy as np

from daf import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-function-docstring,too-many-locals,self-assigning-variable,too-many-statements


def test_array1d_types() -> None:
    array1d_of_bool: Array1DofBool = be_array1d_of_bool(np.zeros(1, dtype="bool"))

    array1d_of_int8: Array1DofInt8 = be_array1d_of_int8(np.zeros(1, dtype="int8"))
    array1d_of_uint8: Array1DofUInt8 = be_array1d_of_uint8(np.zeros(1, dtype="uint8"))
    array1d_of_int16: Array1DofInt16 = be_array1d_of_int16(np.zeros(1, dtype="int16"))
    array1d_of_uint16: Array1DofUInt16 = be_array1d_of_uint16(np.zeros(1, dtype="uint16"))
    array1d_of_int32: Array1DofInt32 = be_array1d_of_int32(np.zeros(1, dtype="int32"))
    array1d_of_uint32: Array1DofUInt32 = be_array1d_of_uint32(np.zeros(1, dtype="uint32"))
    array1d_of_int64: Array1DofInt64 = be_array1d_of_int64(np.zeros(1, dtype="int64"))
    array1d_of_uint64: Array1DofUInt64 = be_array1d_of_uint64(np.zeros(1, dtype="uint64"))

    array1d_of_int: Array1DofInt
    array1d_of_int = be_array1d_of_int(array1d_of_int8)
    array1d_of_int = be_array1d_of_int(array1d_of_uint8)
    array1d_of_int = be_array1d_of_int(array1d_of_int16)
    array1d_of_int = be_array1d_of_int(array1d_of_uint16)
    array1d_of_int = be_array1d_of_int(array1d_of_int32)
    array1d_of_int = be_array1d_of_int(array1d_of_uint32)
    array1d_of_int = be_array1d_of_int(array1d_of_int64)
    array1d_of_int = be_array1d_of_int(array1d_of_uint64)
    array1d_of_int = array1d_of_int

    array1d_of_float16: Array1DofFloat16 = be_array1d_of_float16(np.zeros(1, dtype="float16"))
    array1d_of_float32: Array1DofFloat32 = be_array1d_of_float32(np.zeros(1, dtype="float32"))
    array1d_of_float64: Array1DofFloat64 = be_array1d_of_float64(np.zeros(1, dtype="float64"))

    array1d_of_float: Array1DofFloat
    array1d_of_float = be_array1d_of_float(array1d_of_float16)
    array1d_of_float = be_array1d_of_float(array1d_of_float32)
    array1d_of_float = be_array1d_of_float(array1d_of_float64)
    array1d_of_float = array1d_of_float

    array1d_of_num: Array1DofNum
    array1d_of_num = be_array1d_of_num(array1d_of_int8)
    array1d_of_num = be_array1d_of_num(array1d_of_uint8)
    array1d_of_num = be_array1d_of_num(array1d_of_int16)
    array1d_of_num = be_array1d_of_num(array1d_of_uint16)
    array1d_of_num = be_array1d_of_num(array1d_of_int32)
    array1d_of_num = be_array1d_of_num(array1d_of_uint32)
    array1d_of_num = be_array1d_of_num(array1d_of_int64)
    array1d_of_num = be_array1d_of_num(array1d_of_uint64)
    array1d_of_num = be_array1d_of_num(array1d_of_float16)
    array1d_of_num = be_array1d_of_num(array1d_of_float32)
    array1d_of_num = be_array1d_of_num(array1d_of_float64)
    array1d_of_num = array1d_of_num

    array1d_of_str: Array1DofStr = be_array1d_of_str(np.array([""]))

    array1d: Array1D
    array1d = be_array1d(array1d_of_bool)
    array1d = be_array1d(array1d_of_int8)
    array1d = be_array1d(array1d_of_uint8)
    array1d = be_array1d(array1d_of_int16)
    array1d = be_array1d(array1d_of_uint16)
    array1d = be_array1d(array1d_of_int32)
    array1d = be_array1d(array1d_of_uint32)
    array1d = be_array1d(array1d_of_int64)
    array1d = be_array1d(array1d_of_uint64)
    array1d = be_array1d(array1d_of_float16)
    array1d = be_array1d(array1d_of_float32)
    array1d = be_array1d(array1d_of_float64)
    array1d = be_array1d(array1d_of_str)
    array1d = array1d


def test_array2d_types() -> None:
    array2d_of_bool: Array2DofBool = be_array2d_of_bool(np.zeros((1, 1), dtype="bool"))

    array2d_of_int8: Array2DofInt8 = be_array2d_of_int8(np.zeros((1, 1), dtype="int8"))
    array2d_of_uint8: Array2DofUInt8 = be_array2d_of_uint8(np.zeros((1, 1), dtype="uint8"))
    array2d_of_int16: Array2DofInt16 = be_array2d_of_int16(np.zeros((1, 1), dtype="int16"))
    array2d_of_uint16: Array2DofUInt16 = be_array2d_of_uint16(np.zeros((1, 1), dtype="uint16"))
    array2d_of_int32: Array2DofInt32 = be_array2d_of_int32(np.zeros((1, 1), dtype="int32"))
    array2d_of_uint32: Array2DofUInt32 = be_array2d_of_uint32(np.zeros((1, 1), dtype="uint32"))
    array2d_of_int64: Array2DofInt64 = be_array2d_of_int64(np.zeros((1, 1), dtype="int64"))
    array2d_of_uint64: Array2DofUInt64 = be_array2d_of_uint64(np.zeros((1, 1), dtype="uint64"))

    array2d_of_int: Array2DofInt
    array2d_of_int = be_array2d_of_int(array2d_of_int8)
    array2d_of_int = be_array2d_of_int(array2d_of_uint8)
    array2d_of_int = be_array2d_of_int(array2d_of_int16)
    array2d_of_int = be_array2d_of_int(array2d_of_uint16)
    array2d_of_int = be_array2d_of_int(array2d_of_int32)
    array2d_of_int = be_array2d_of_int(array2d_of_uint32)
    array2d_of_int = be_array2d_of_int(array2d_of_int64)
    array2d_of_int = be_array2d_of_int(array2d_of_uint64)
    array2d_of_int = array2d_of_int

    array2d_of_float16: Array2DofFloat16 = be_array2d_of_float16(np.zeros((1, 1), dtype="float16"))
    array2d_of_float32: Array2DofFloat32 = be_array2d_of_float32(np.zeros((1, 1), dtype="float32"))
    array2d_of_float64: Array2DofFloat64 = be_array2d_of_float64(np.zeros((1, 1), dtype="float64"))

    array2d_of_float: Array2DofFloat
    array2d_of_float = be_array2d_of_float(array2d_of_float16)
    array2d_of_float = be_array2d_of_float(array2d_of_float32)
    array2d_of_float = be_array2d_of_float(array2d_of_float64)
    array2d_of_float = array2d_of_float

    array2d_of_num: Array2DofNum
    array2d_of_num = be_array2d_of_num(array2d_of_int8)
    array2d_of_num = be_array2d_of_num(array2d_of_uint8)
    array2d_of_num = be_array2d_of_num(array2d_of_int16)
    array2d_of_num = be_array2d_of_num(array2d_of_uint16)
    array2d_of_num = be_array2d_of_num(array2d_of_int32)
    array2d_of_num = be_array2d_of_num(array2d_of_uint32)
    array2d_of_num = be_array2d_of_num(array2d_of_int64)
    array2d_of_num = be_array2d_of_num(array2d_of_uint64)
    array2d_of_num = be_array2d_of_num(array2d_of_float16)
    array2d_of_num = be_array2d_of_num(array2d_of_float32)
    array2d_of_num = be_array2d_of_num(array2d_of_float64)
    array2d_of_num = array2d_of_num

    array2d_of_str: Array2DofStr = be_array2d_of_str(np.array([[""], [""]]))

    array2d: Array2D
    array2d = be_array2d(array2d_of_bool)
    array2d = be_array2d(array2d_of_int8)
    array2d = be_array2d(array2d_of_uint8)
    array2d = be_array2d(array2d_of_int16)
    array2d = be_array2d(array2d_of_uint16)
    array2d = be_array2d(array2d_of_int32)
    array2d = be_array2d(array2d_of_uint32)
    array2d = be_array2d(array2d_of_int64)
    array2d = be_array2d(array2d_of_uint64)
    array2d = be_array2d(array2d_of_float16)
    array2d = be_array2d(array2d_of_float32)
    array2d = be_array2d(array2d_of_float64)
    array2d = be_array2d(array2d_of_str)
    array2d = array2d
