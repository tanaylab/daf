"""
Test typing.
"""

import numpy as np
import pandas as pd  # type: ignore

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


def test_series_types() -> None:
    series_of_bool: SeriesOfBool = be_series_of_bool(pd.Series([0], dtype="bool"))

    be_array1d_of_bool(to_array1d(series_of_bool))

    series_of_int8: SeriesOfInt8 = be_series_of_int8(pd.Series([0], dtype="int8"))
    series_of_uint8: SeriesOfUInt8 = be_series_of_uint8(pd.Series([0], dtype="uint8"))
    series_of_int16: SeriesOfInt16 = be_series_of_int16(pd.Series([0], dtype="int16"))
    series_of_uint16: SeriesOfUInt16 = be_series_of_uint16(pd.Series([0], dtype="uint16"))
    series_of_int32: SeriesOfInt32 = be_series_of_int32(pd.Series([0], dtype="int32"))
    series_of_uint32: SeriesOfUInt32 = be_series_of_uint32(pd.Series([0], dtype="uint32"))
    series_of_int64: SeriesOfInt64 = be_series_of_int64(pd.Series([0], dtype="int64"))
    series_of_uint64: SeriesOfUInt64 = be_series_of_uint64(pd.Series([0], dtype="uint64"))

    be_array1d_of_int8(to_array1d(series_of_int8))
    be_array1d_of_uint8(to_array1d(series_of_uint8))
    be_array1d_of_int16(to_array1d(series_of_int16))
    be_array1d_of_uint16(to_array1d(series_of_uint16))
    be_array1d_of_int32(to_array1d(series_of_int32))
    be_array1d_of_uint32(to_array1d(series_of_uint32))
    be_array1d_of_int64(to_array1d(series_of_int64))
    be_array1d_of_uint64(to_array1d(series_of_uint64))

    series_of_int: SeriesOfInt
    series_of_int = be_series_of_int(series_of_int8)
    series_of_int = be_series_of_int(series_of_uint8)
    series_of_int = be_series_of_int(series_of_int16)
    series_of_int = be_series_of_int(series_of_uint16)
    series_of_int = be_series_of_int(series_of_int32)
    series_of_int = be_series_of_int(series_of_uint32)
    series_of_int = be_series_of_int(series_of_int64)
    series_of_int = be_series_of_int(series_of_uint64)
    series_of_int = series_of_int

    be_array1d_of_int(to_array1d(series_of_int))

    series_of_float16: SeriesOfFloat16 = be_series_of_float16(pd.Series([0], dtype="float16"))
    series_of_float32: SeriesOfFloat32 = be_series_of_float32(pd.Series([0], dtype="float32"))
    series_of_float64: SeriesOfFloat64 = be_series_of_float64(pd.Series([0], dtype="float64"))

    be_array1d_of_float16(to_array1d(series_of_float16))
    be_array1d_of_float32(to_array1d(series_of_float32))
    be_array1d_of_float64(to_array1d(series_of_float64))

    series_of_float: SeriesOfFloat
    series_of_float = be_series_of_float(series_of_float16)
    series_of_float = be_series_of_float(series_of_float32)
    series_of_float = be_series_of_float(series_of_float64)
    series_of_float = series_of_float

    be_array1d_of_float(to_array1d(series_of_float))

    series_of_num: SeriesOfNum
    series_of_num = be_series_of_num(series_of_int8)
    series_of_num = be_series_of_num(series_of_uint8)
    series_of_num = be_series_of_num(series_of_int16)
    series_of_num = be_series_of_num(series_of_uint16)
    series_of_num = be_series_of_num(series_of_int32)
    series_of_num = be_series_of_num(series_of_uint32)
    series_of_num = be_series_of_num(series_of_int64)
    series_of_num = be_series_of_num(series_of_uint64)
    series_of_num = be_series_of_num(series_of_float16)
    series_of_num = be_series_of_num(series_of_float32)
    series_of_num = be_series_of_num(series_of_float64)
    series_of_num = series_of_num

    be_array1d_of_num(to_array1d(series_of_num))

    series_of_str: SeriesOfStr = be_series_of_str(pd.Series([""], dtype="string"))

    be_array1d_of_str(to_array1d(series_of_str))

    series_of_category: SeriesOfStr = be_series_of_str(pd.Series([""], dtype="category"))

    be_array1d_of_str(to_array1d(series_of_category))

    series: Series
    series = be_series(series_of_bool)
    series = be_series(series_of_int8)
    series = be_series(series_of_uint8)
    series = be_series(series_of_int16)
    series = series


def test_frame_types() -> None:
    frame_of_bool: FrameOfBool = be_frame_of_bool(pd.DataFrame(np.zeros((1, 1), dtype="bool")))

    be_array2d_of_bool(to_array2d(frame_of_bool))

    frame_of_int8: FrameOfInt8 = be_frame_of_int8(pd.DataFrame(np.zeros((1, 1), dtype="int8")))
    frame_of_uint8: FrameOfUInt8 = be_frame_of_uint8(pd.DataFrame(np.zeros((1, 1), dtype="uint8")))
    frame_of_int16: FrameOfInt16 = be_frame_of_int16(pd.DataFrame(np.zeros((1, 1), dtype="int16")))
    frame_of_uint16: FrameOfUInt16 = be_frame_of_uint16(pd.DataFrame(np.zeros((1, 1), dtype="uint16")))
    frame_of_int32: FrameOfInt32 = be_frame_of_int32(pd.DataFrame(np.zeros((1, 1), dtype="int32")))
    frame_of_uint32: FrameOfUInt32 = be_frame_of_uint32(pd.DataFrame(np.zeros((1, 1), dtype="uint32")))
    frame_of_int64: FrameOfInt64 = be_frame_of_int64(pd.DataFrame(np.zeros((1, 1), dtype="int64")))
    frame_of_uint64: FrameOfUInt64 = be_frame_of_uint64(pd.DataFrame(np.zeros((1, 1), dtype="uint64")))

    be_series_of_int8(frame_of_int8.iloc[:, 0])
    be_series_of_uint8(frame_of_uint8.iloc[:, 0])
    be_series_of_int16(frame_of_int16.iloc[:, 0])
    be_series_of_uint16(frame_of_uint16.iloc[:, 0])
    be_series_of_int32(frame_of_int32.iloc[:, 0])
    be_series_of_uint32(frame_of_uint32.iloc[:, 0])
    be_series_of_int64(frame_of_int64.iloc[:, 0])
    be_series_of_uint64(frame_of_uint64.iloc[:, 0])

    be_array2d_of_int8(to_array2d(frame_of_int8))
    be_array2d_of_uint8(to_array2d(frame_of_uint8))
    be_array2d_of_int16(to_array2d(frame_of_int16))
    be_array2d_of_uint16(to_array2d(frame_of_uint16))
    be_array2d_of_int32(to_array2d(frame_of_int32))
    be_array2d_of_uint32(to_array2d(frame_of_uint32))
    be_array2d_of_int64(to_array2d(frame_of_int64))
    be_array2d_of_uint64(to_array2d(frame_of_uint64))

    frame_of_int: FrameOfInt
    frame_of_int = be_frame_of_int(frame_of_int8)
    frame_of_int = be_frame_of_int(frame_of_uint8)
    frame_of_int = be_frame_of_int(frame_of_int16)
    frame_of_int = be_frame_of_int(frame_of_uint16)
    frame_of_int = be_frame_of_int(frame_of_int32)
    frame_of_int = be_frame_of_int(frame_of_uint32)
    frame_of_int = be_frame_of_int(frame_of_int64)
    frame_of_int = be_frame_of_int(frame_of_uint64)

    be_series_of_int(frame_of_int.iloc[:, 0])

    be_array2d_of_int(to_array2d(frame_of_int))

    frame_of_float16: FrameOfFloat16 = be_frame_of_float16(pd.DataFrame(np.zeros((1, 1), dtype="float16")))
    frame_of_float32: FrameOfFloat32 = be_frame_of_float32(pd.DataFrame(np.zeros((1, 1), dtype="float32")))
    frame_of_float64: FrameOfFloat64 = be_frame_of_float64(pd.DataFrame(np.zeros((1, 1), dtype="float64")))

    be_series_of_float16(frame_of_float16.iloc[:, 0])
    be_series_of_float32(frame_of_float32.iloc[:, 0])
    be_series_of_float64(frame_of_float64.iloc[:, 0])

    be_array2d_of_float16(to_array2d(frame_of_float16))
    be_array2d_of_float32(to_array2d(frame_of_float32))
    be_array2d_of_float64(to_array2d(frame_of_float64))

    frame_of_float: FrameOfFloat
    frame_of_float = be_frame_of_float(frame_of_float16)
    frame_of_float = be_frame_of_float(frame_of_float32)
    frame_of_float = be_frame_of_float(frame_of_float64)

    be_series_of_float(frame_of_float.iloc[:, 0])

    be_array2d_of_float(to_array2d(frame_of_float))

    frame_of_num: FrameOfNum
    frame_of_num = be_frame_of_num(frame_of_int8)
    frame_of_num = be_frame_of_num(frame_of_uint8)
    frame_of_num = be_frame_of_num(frame_of_int16)
    frame_of_num = be_frame_of_num(frame_of_uint16)
    frame_of_num = be_frame_of_num(frame_of_int32)
    frame_of_num = be_frame_of_num(frame_of_uint32)
    frame_of_num = be_frame_of_num(frame_of_int64)
    frame_of_num = be_frame_of_num(frame_of_uint64)
    frame_of_num = be_frame_of_num(frame_of_float16)
    frame_of_num = be_frame_of_num(frame_of_float32)
    frame_of_num = be_frame_of_num(frame_of_float64)

    be_series_of_num(frame_of_num.iloc[:, 0])

    be_array2d_of_num(to_array2d(frame_of_num))

    frame_of_str: FrameOfStr = be_frame_of_str(
        pd.DataFrame(dict(a=pd.Series([""], dtype="string"), b=pd.Series([""], dtype="category")))
    )

    be_series_of_str(frame_of_str["a"])
    be_series_of_str(frame_of_str["b"])

    be_array2d_of_str(to_array2d(frame_of_str))

    frame_of_any: FrameOfAny
    frame_of_any = be_frame_of_any(frame_of_bool)
    frame_of_any = be_frame_of_any(frame_of_int8)
    frame_of_any = be_frame_of_any(frame_of_uint8)
    frame_of_any = be_frame_of_any(frame_of_int16)
    frame_of_any = be_frame_of_any(frame_of_uint16)
    frame_of_any = be_frame_of_any(frame_of_int32)
    frame_of_any = be_frame_of_any(frame_of_uint32)
    frame_of_any = be_frame_of_any(frame_of_int64)
    frame_of_any = be_frame_of_any(frame_of_uint64)
    frame_of_any = be_frame_of_any(frame_of_float16)
    frame_of_any = be_frame_of_any(frame_of_float32)
    frame_of_any = be_frame_of_any(frame_of_float64)
    frame_of_any = be_frame_of_any(frame_of_str)
    frame_of_any = frame_of_any

    be_series(frame_of_any.iloc[:, 0])

    be_array2d(to_array2d(frame_of_any))

    frame_of_many: FrameOfMany = be_frame_of_many(
        pd.DataFrame(dict(a=pd.Series([0], dtype="int"), b=pd.Series([""], dtype="category")))
    )

    frame: Frame
    frame = be_frame(frame_of_bool)
    frame = be_frame(frame_of_int8)
    frame = be_frame(frame_of_uint8)
    frame = be_frame(frame_of_int16)
    frame = be_frame(frame_of_uint16)
    frame = be_frame(frame_of_int32)
    frame = be_frame(frame_of_uint32)
    frame = be_frame(frame_of_int64)
    frame = be_frame(frame_of_uint64)
    frame = be_frame(frame_of_float16)
    frame = be_frame(frame_of_float32)
    frame = be_frame(frame_of_float64)
    frame = be_frame(frame_of_str)
    frame = be_frame(frame_of_many)
    frame = frame

    be_series(frame.iloc[:, 0])
