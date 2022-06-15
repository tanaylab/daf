"""
Test ``daf.access.operations``.
"""

import numpy as np
import scipy.sparse as sp  # type: ignore

from daf.access.operations import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-function-docstring


def check_element_wise(
    element_wise: ElementWise, *, canonical: str, input_dense: np.ndarray, expected_output: np.ndarray
) -> None:
    assert element_wise.canonical == canonical

    input_vector = input_dense[0, :]
    expected_vector = expected_output[0, :]
    actual_vector = element_wise.vector_to_vector(input_vector)
    assert actual_vector.dtype == expected_vector.dtype
    assert np.allclose(actual_vector, expected_vector)

    if element_wise.sparsifies:
        actual_output = be_dense_in_rows(
            be_sparse_in_rows(element_wise.dense_to_sparse(be_dense_in_rows(input_dense))).toarray(order="C")
        )
    else:
        actual_output = be_dense_in_rows(np.empty(expected_output.shape, dtype=expected_output.dtype))
        element_wise.dense_to_dense(be_dense_in_rows(input_dense), be_dense_in_rows(actual_output))
    assert actual_output.dtype == expected_output.dtype
    assert np.allclose(actual_output, expected_output)

    input_sparse = sp.csr_matrix(input_dense)
    if element_wise.densifies:
        actual_output = be_dense_in_rows(np.empty(expected_output.shape, dtype=expected_output.dtype))
        element_wise.sparse_to_dense(input_sparse, be_dense_in_rows(actual_output))
    else:
        actual_output = be_dense_in_rows(
            be_sparse_in_rows(element_wise.sparse_to_sparse(input_sparse)).toarray(order="C")
        )
    assert actual_output.dtype == expected_output.dtype
    assert np.allclose(actual_output, expected_output)


def test_abs() -> None:
    check_element_wise(
        Abs(_input_dtype="float32", dtype="float64"),
        canonical="Abs,dtype=float64",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.array([[1.8, 0], [0.8, 1.8]], dtype="float64"),
    )


def test_floor() -> None:
    check_element_wise(
        Floor(_input_dtype="float32", dtype="int8"),
        canonical="Floor,dtype=int8",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.array([[-2, 0], [0, 1]], dtype="int8"),
    )


def test_round() -> None:
    check_element_wise(
        Round(_input_dtype="float32", dtype="int8"),
        canonical="Round,dtype=int8",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.array([[-2, 0], [1, 2]], dtype="int8"),
    )


def test_ceil() -> None:
    check_element_wise(
        Ceil(_input_dtype="float32", dtype="int8"),
        canonical="Ceil,dtype=int8",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.array([[-1, 0], [1, 2]], dtype="int8"),
    )


def test_clip() -> None:
    check_element_wise(
        Clip(_input_dtype="float32", min="-1", max="1"),
        canonical="Clip,min=-1.0,max=1.0,dtype=float32",
        input_dense=np.array([[-1.8, 0.8], [0.8, 1.8]], dtype="float32"),
        expected_output=np.array([[-1, 0.8], [0.8, 1]], dtype="float32"),
    )


def test_log() -> None:
    check_element_wise(
        Log(_input_dtype="float32", base="e", factor="1"),
        canonical="Log,base=e,factor=1.0,dtype=float32",
        input_dense=np.array([[0, 0.8], [0.8, 1.8]], dtype="float32"),
        expected_output=np.log(np.array([[1, 1.8], [1.8, 2.8]], dtype="float32")),
    )
    check_element_wise(
        Log(_input_dtype="float32", base="2", factor="1"),
        canonical="Log,base=2.0,factor=1.0,dtype=float32",
        input_dense=np.array([[0, 0.8], [0.8, 1.8]], dtype="float32"),
        expected_output=np.log2(np.array([[1, 1.8], [1.8, 2.8]], dtype="float32")),
    )
    check_element_wise(
        Log(_input_dtype="float3", base="10", factor="1"),
        canonical="Log,base=10.0,factor=1.0,dtype=float32",
        input_dense=np.array([[0, 0.8], [0.8, 1.8]], dtype="float32"),
        expected_output=np.log10(np.array([[1, 1.8], [1.8, 2.8]], dtype="float32")),
    )
    check_element_wise(
        Log(_input_dtype="float3", base="4", factor="1"),
        canonical="Log,base=4.0,factor=1.0,dtype=float32",
        input_dense=np.array([[0, 0.8], [0.8, 1.8]], dtype="float32"),
        expected_output=np.log2(np.array([[1, 1.8], [1.8, 2.8]], dtype="float32")) / 2,
    )


def test_reformat() -> None:
    check_element_wise(
        Densify(_input_dtype="float32"),
        canonical="Densify,dtype=float32",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
    )
    check_element_wise(
        Sparsify(_input_dtype="float32"),
        canonical="Sparsify,dtype=float32",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
    )


def test_significant() -> None:
    check_element_wise(
        Significant(_input_dtype="float32", low="1", high="2", abs="false"),
        canonical="Significant,low=1.0,high=2.0,abs=False,dtype=float32",
        input_dense=np.array([[-2.5, -1.5, -0.5], [2.5, 1.5, 0.5], [-0.5, 0.5, 1.5]], dtype="float32"),
        expected_output=np.array([[0, 0, 0], [2.5, 1.5, 0], [0, 0, 0]], dtype="float32"),
    )
    check_element_wise(
        Significant(_input_dtype="float32", low="1", high="2"),
        canonical="Significant,low=1.0,high=2.0,abs=True,dtype=float32",
        input_dense=np.array([[-2.5, -1.5, -0.5], [2.5, 1.5, 0.5], [-0.5, 0.5, 1.5]], dtype="float32"),
        expected_output=np.array([[-2.5, -1.5, 0], [2.5, 1.5, 0], [0, 0, 0]], dtype="float32"),
    )


def check_reduction(
    reduction: Reduction, *, canonical: str, input_dense: np.ndarray, expected_output: np.ndarray
) -> None:
    assert reduction.canonical == canonical

    input_vector = input_dense[0, :]
    expected_scalar = expected_output[0]
    actual_scalar = reduction.vector_to_scalar(input_vector)
    assert np.allclose(actual_scalar, expected_scalar)

    actual_output = reduction.dense_to_vector(be_dense_in_rows(input_dense))
    assert actual_output.dtype == expected_output.dtype
    assert np.allclose(actual_output, expected_output)

    actual_output = reduction.sparse_to_vector(sp.csr_matrix(input_dense))
    assert actual_output.dtype == expected_output.dtype
    assert np.allclose(actual_output, expected_output)


def test_sum() -> None:
    check_reduction(
        Sum(_input_dtype="float32"),
        canonical="Sum,dtype=float32",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.array([-1.8, 2.6], dtype="float32"),
    )


def test_min() -> None:
    check_reduction(
        Min(_input_dtype="float32"),
        canonical="Min,dtype=float32",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.array([-1.8, 0.8], dtype="float32"),
    )


def test_mean() -> None:
    check_reduction(
        Mean(_input_dtype="float32"),
        canonical="Mean,dtype=float32",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.array([-0.9, 1.3], dtype="float32"),
    )


def test_max() -> None:
    check_reduction(
        Max(_input_dtype="float32"),
        canonical="Max,dtype=float32",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.array([0, 1.8], dtype="float32"),
    )


def test_var() -> None:
    check_reduction(
        Var(_input_dtype="float32"),
        canonical="Var,dtype=float32",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.var(np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"), axis=1),
    )


def test_std() -> None:
    check_reduction(
        Std(_input_dtype="float32"),
        canonical="Std,dtype=float32",
        input_dense=np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"),
        expected_output=np.std(np.array([[-1.8, 0], [0.8, 1.8]], dtype="float32"), axis=1),
    )
