"""
Test ``daf.typing.memory_mapping``.
"""

# pylint: disable=duplicate-code

from resource import RLIMIT_NOFILE
from resource import getrlimit
from tempfile import TemporaryDirectory

import numpy as np
import scipy.sparse as sp  # type: ignore

from daf.storage.memory_mapping import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-function-docstring


def test_allow_maximal_open_files() -> None:
    _soft, hard = getrlimit(RLIMIT_NOFILE)
    assert allow_maximal_open_files() == hard


def test_memory_mapping_vector() -> None:
    with TemporaryDirectory() as directory:
        path = f"{directory}/vector"
        assert not exists_memory_mapped_array(path)
        create_memory_mapped_array(path, 10, "int32")

        assert exists_memory_mapped_array(path)
        vector = open_memory_mapped_array(path, "r+")
        assert is_vector(vector)
        assert not is_frozen(vector)
        assert vector.dtype == "int32"
        assert np.all(vector == np.zeros(10))
        vector[:] = np.arange(10)

        vector = open_memory_mapped_array(path, "r")
        assert is_vector(vector)
        assert is_frozen(vector)
        assert vector.dtype == "int32"
        assert np.all(vector == np.arange(10))

        remove_memory_mapped_array(path)
        assert not exists_memory_mapped_array(path)


def test_memory_mapping_dense() -> None:
    with TemporaryDirectory() as directory:
        path = f"{directory}/dense"
        assert not exists_memory_mapped_array(path)
        create_memory_mapped_array(path, (10, 20), "float32")

        assert exists_memory_mapped_array(path)
        dense = open_memory_mapped_array(path, "r+")
        assert is_dense_in_rows(dense)
        assert not is_frozen(dense)
        assert dense.dtype == "float32"
        assert np.all(dense == np.zeros((10, 20)))

        np.random.seed(123456)
        rand2d = np.random.rand(10, 20).astype("float32")
        dense[:, :] = rand2d

        dense = open_memory_mapped_array(path, "r")
        assert is_dense_in_rows(dense)
        assert is_frozen(dense)
        assert dense.dtype == "float32"
        assert np.all(dense == rand2d)

        remove_memory_mapped_array(path)
        assert not exists_memory_mapped_array(path)


def test_memory_mapping_sparse() -> None:
    with TemporaryDirectory() as directory:
        path = f"{directory}/sparse"
        assert not exists_memory_mapped_sparse(path)

        np.random.seed(123456)
        rand2d = sp.random(10, 20, density=0.25, format="csr", dtype="float32")
        write_memory_mapped_sparse(path, rand2d)

        sparse = open_memory_mapped_sparse(path, "r")
        assert is_sparse_in_rows(sparse)
        assert is_frozen(sparse)
        assert sparse.data.dtype == "float32"
        assert fast_all_close(sparse, rand2d)

        remove_memory_mapped_sparse(path)
        assert not exists_memory_mapped_sparse(path)
