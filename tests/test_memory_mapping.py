"""
Test ``daf.typing.memory_mapping``.
"""

# pylint: disable=duplicate-code

from resource import RLIMIT_NOFILE
from resource import getrlimit
from tempfile import TemporaryDirectory

import numpy as np

from daf.storage.memory_mapping import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-function-docstring


def test_allow_maximal_open_files() -> None:
    _soft, hard = getrlimit(RLIMIT_NOFILE)
    assert allow_maximal_open_files() == hard


def test_memory_mapping_array1d() -> None:
    with TemporaryDirectory() as directory:
        path = f"{directory}/array1d"
        assert not exists_memory_mapped_dense(path)
        create_memory_mapped_dense(path, 10, "int32")

        assert exists_memory_mapped_dense(path)
        array1d = open_memory_mapped_dense(path, "w")
        assert is_array1d(array1d)
        assert not is_frozen(array1d)
        assert array1d.dtype == "int32"
        assert np.all(array1d == np.zeros(10))
        array1d[:] = np.arange(10)

        array1d = open_memory_mapped_dense(path, "r")
        assert is_array1d(array1d)
        assert is_frozen(array1d)
        assert array1d.dtype == "int32"
        assert np.all(array1d == np.arange(10))

        remove_memory_mapped_dense(path)
        assert not exists_memory_mapped_dense(path)
