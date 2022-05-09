"""
Test typing.
"""

import numpy as np

from daf import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-function-docstring,too-many-locals,self-assigning-variable,too-many-statements


def test_vector_types() -> None:
    vector_of_bool: VectorOfBool = be_vector_of_bool(np.zeros(1, dtype="bool"))

    vector_of_int8: VectorOfInt8 = be_vector_of_int8(np.zeros(1, dtype="int8"))
    vector_of_uint8: VectorOfUInt8 = be_vector_of_uint8(np.zeros(1, dtype="uint8"))
    vector_of_int16: VectorOfInt16 = be_vector_of_int16(np.zeros(1, dtype="int16"))
    vector_of_uint16: VectorOfUInt16 = be_vector_of_uint16(np.zeros(1, dtype="uint16"))
    vector_of_int32: VectorOfInt32 = be_vector_of_int32(np.zeros(1, dtype="int32"))
    vector_of_uint32: VectorOfUInt32 = be_vector_of_uint32(np.zeros(1, dtype="uint32"))
    vector_of_int64: VectorOfInt64 = be_vector_of_int64(np.zeros(1, dtype="int64"))
    vector_of_uint64: VectorOfUInt64 = be_vector_of_uint64(np.zeros(1, dtype="uint64"))

    vector_of_int: VectorOfInt
    vector_of_int = be_vector_of_int(vector_of_int8)
    vector_of_int = be_vector_of_int(vector_of_uint8)
    vector_of_int = be_vector_of_int(vector_of_int16)
    vector_of_int = be_vector_of_int(vector_of_uint16)
    vector_of_int = be_vector_of_int(vector_of_int32)
    vector_of_int = be_vector_of_int(vector_of_uint32)
    vector_of_int = be_vector_of_int(vector_of_int64)
    vector_of_int = be_vector_of_int(vector_of_uint64)
    vector_of_int = vector_of_int

    vector_of_float16: VectorOfFloat16 = be_vector_of_float16(np.zeros(1, dtype="float16"))
    vector_of_float32: VectorOfFloat32 = be_vector_of_float32(np.zeros(1, dtype="float32"))
    vector_of_float64: VectorOfFloat64 = be_vector_of_float64(np.zeros(1, dtype="float64"))

    vector_of_float: VectorOfFloat
    vector_of_float = be_vector_of_float(vector_of_float16)
    vector_of_float = be_vector_of_float(vector_of_float32)
    vector_of_float = be_vector_of_float(vector_of_float64)
    vector_of_float = vector_of_float

    vector_of_num: VectorOfNum
    vector_of_num = be_vector_of_num(vector_of_int8)
    vector_of_num = be_vector_of_num(vector_of_uint8)
    vector_of_num = be_vector_of_num(vector_of_int16)
    vector_of_num = be_vector_of_num(vector_of_uint16)
    vector_of_num = be_vector_of_num(vector_of_int32)
    vector_of_num = be_vector_of_num(vector_of_uint32)
    vector_of_num = be_vector_of_num(vector_of_int64)
    vector_of_num = be_vector_of_num(vector_of_uint64)
    vector_of_num = be_vector_of_num(vector_of_float16)
    vector_of_num = be_vector_of_num(vector_of_float32)
    vector_of_num = be_vector_of_num(vector_of_float64)
    vector_of_num = vector_of_num

    vector_of_str: VectorOfStr = be_vector_of_str(np.array([""]))

    vector: Vector
    vector = be_vector(vector_of_bool)
    vector = be_vector(vector_of_int8)
    vector = be_vector(vector_of_uint8)
    vector = be_vector(vector_of_int16)
    vector = be_vector(vector_of_uint16)
    vector = be_vector(vector_of_int32)
    vector = be_vector(vector_of_uint32)
    vector = be_vector(vector_of_int64)
    vector = be_vector(vector_of_uint64)
    vector = be_vector(vector_of_float16)
    vector = be_vector(vector_of_float32)
    vector = be_vector(vector_of_float64)
    vector = be_vector(vector_of_str)
    vector = vector


def test_matrix_types() -> None:
    matrix_of_bool: MatrixOfBool = be_matrix_of_bool(np.zeros((1, 1), dtype="bool"))

    matrix_of_int8: MatrixOfInt8 = be_matrix_of_int8(np.zeros((1, 1), dtype="int8"))
    matrix_of_uint8: MatrixOfUInt8 = be_matrix_of_uint8(np.zeros((1, 1), dtype="uint8"))
    matrix_of_int16: MatrixOfInt16 = be_matrix_of_int16(np.zeros((1, 1), dtype="int16"))
    matrix_of_uint16: MatrixOfUInt16 = be_matrix_of_uint16(np.zeros((1, 1), dtype="uint16"))
    matrix_of_int32: MatrixOfInt32 = be_matrix_of_int32(np.zeros((1, 1), dtype="int32"))
    matrix_of_uint32: MatrixOfUInt32 = be_matrix_of_uint32(np.zeros((1, 1), dtype="uint32"))
    matrix_of_int64: MatrixOfInt64 = be_matrix_of_int64(np.zeros((1, 1), dtype="int64"))
    matrix_of_uint64: MatrixOfUInt64 = be_matrix_of_uint64(np.zeros((1, 1), dtype="uint64"))

    matrix_of_int: MatrixOfInt
    matrix_of_int = be_matrix_of_int(matrix_of_int8)
    matrix_of_int = be_matrix_of_int(matrix_of_uint8)
    matrix_of_int = be_matrix_of_int(matrix_of_int16)
    matrix_of_int = be_matrix_of_int(matrix_of_uint16)
    matrix_of_int = be_matrix_of_int(matrix_of_int32)
    matrix_of_int = be_matrix_of_int(matrix_of_uint32)
    matrix_of_int = be_matrix_of_int(matrix_of_int64)
    matrix_of_int = be_matrix_of_int(matrix_of_uint64)
    matrix_of_int = matrix_of_int

    matrix_of_float16: MatrixOfFloat16 = be_matrix_of_float16(np.zeros((1, 1), dtype="float16"))
    matrix_of_float32: MatrixOfFloat32 = be_matrix_of_float32(np.zeros((1, 1), dtype="float32"))
    matrix_of_float64: MatrixOfFloat64 = be_matrix_of_float64(np.zeros((1, 1), dtype="float64"))

    matrix_of_float: MatrixOfFloat
    matrix_of_float = be_matrix_of_float(matrix_of_float16)
    matrix_of_float = be_matrix_of_float(matrix_of_float32)
    matrix_of_float = be_matrix_of_float(matrix_of_float64)
    matrix_of_float = matrix_of_float

    matrix_of_num: MatrixOfNum
    matrix_of_num = be_matrix_of_num(matrix_of_int8)
    matrix_of_num = be_matrix_of_num(matrix_of_uint8)
    matrix_of_num = be_matrix_of_num(matrix_of_int16)
    matrix_of_num = be_matrix_of_num(matrix_of_uint16)
    matrix_of_num = be_matrix_of_num(matrix_of_int32)
    matrix_of_num = be_matrix_of_num(matrix_of_uint32)
    matrix_of_num = be_matrix_of_num(matrix_of_int64)
    matrix_of_num = be_matrix_of_num(matrix_of_uint64)
    matrix_of_num = be_matrix_of_num(matrix_of_float16)
    matrix_of_num = be_matrix_of_num(matrix_of_float32)
    matrix_of_num = be_matrix_of_num(matrix_of_float64)
    matrix_of_num = matrix_of_num

    matrix_of_str: MatrixOfStr = be_matrix_of_str(np.array([[""], [""]]))

    matrix: Matrix
    matrix = be_matrix(matrix_of_bool)
    matrix = be_matrix(matrix_of_int8)
    matrix = be_matrix(matrix_of_uint8)
    matrix = be_matrix(matrix_of_int16)
    matrix = be_matrix(matrix_of_uint16)
    matrix = be_matrix(matrix_of_int32)
    matrix = be_matrix(matrix_of_uint32)
    matrix = be_matrix(matrix_of_int64)
    matrix = be_matrix(matrix_of_uint64)
    matrix = be_matrix(matrix_of_float16)
    matrix = be_matrix(matrix_of_float32)
    matrix = be_matrix(matrix_of_float64)
    matrix = be_matrix(matrix_of_str)
    matrix = matrix
