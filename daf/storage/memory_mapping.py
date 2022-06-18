"""
Functions to implement memory-mapped files of 1D/2D data.

The format used here was chosen for simplicity, making it easy for "any" (even non-Python) systems to access the data.
It is so trivial it can hardly be called a "format" at all, and is explicitly **not** the format used by
``numpy.memmap``, which is terribly complicated and can only be accessed using ``numpy`` in Python (in order to support
many use cases we don't care about in ``daf``).

.. note::

    The code here assumes all the machines accessing memory-mapped data use the same (little-endian) byte order and IEEE
    floating point formats, which is true for all modern CPUs.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from functools import reduce
from mmap import ALLOCATIONGRANULARITY
from mmap import PROT_READ
from mmap import PROT_WRITE
from mmap import mmap
from operator import mul
from os import remove as remove_path
from os.path import exists as exists_path
from resource import RLIMIT_NOFILE
from resource import getrlimit
from resource import setrlimit
from typing import Collection
from typing import Optional
from typing import Tuple
from typing import Union
from weakref import WeakValueDictionary

import numpy as np
import scipy.sparse as sp  # type: ignore
from yaml import dump as dump_yaml  # type: ignore
from yaml import safe_load as load_yaml

from ..typing import FIXED_DTYPES
from ..typing import AnyMajor
from ..typing import DenseInRows
from ..typing import DType
from ..typing import SparseInRows
from ..typing import Vector
from ..typing import assert_data
from ..typing import be_dense_in_rows
from ..typing import be_vector
from ..typing import is_dtype
from ..typing import is_optimal
from ..typing import is_sparse_in_rows

# pylint: enable=duplicate-code,cyclic-import


__all__ = [
    "exists_memory_mapped_array",
    "remove_memory_mapped_array",
    "create_memory_mapped_array",
    "open_memory_mapped_array",
    "exists_memory_mapped_sparse",
    "remove_memory_mapped_sparse",
    "write_memory_mapped_sparse",
    "open_memory_mapped_sparse",
    "allow_maximal_open_files",
    "MMAP_CACHE",
    "mmap_file",
    "bytes_as_ndarray",
]


#: Cache memory-mapped files so they are not mapped twice.
#:
#: The key is the path, the mode, offset and size (size=0 means the whole file), the value is the ``mmap`` object.
MMAP_CACHE: WeakValueDictionary[Tuple[str, str, int, int], mmap] = WeakValueDictionary()


_PROT_OF_MODE = {"r": PROT_READ, "r+": PROT_READ | PROT_WRITE}


def mmap_file(*, path: str, mode: str, fd: int, offset: int = 0, size: int = 0) -> Union[bytes, bytearray]:
    """
    Memory-map a whole file at some ``path`` opened using some ``fd`` using some ``mode`` (``r``, ``r+``).

    If both the ``offset`` and the ``size`` are zero, maps the whole file.

    If there already exists a compatible mapping, return it instead of creating a new one. That is:

    * A mapping of the whole file will be used (sliced as needed) to return any mapping for the same file.

    * A mapping for "r+" data will be used to return any mapping for both "r" and "r+" data.

    If the mode is ``r`` returns an immutable ``bytes`` object, if it is ``r+`` returns a mutable ``bytearray`` object.


    .. note::

        Since we are caching the ``mmap``, if the file was changed in the file system, this need not be reflected in the
        result of a following call to ``mmap_file``. E.g., if the file was deleted, the ``mmap`` may survive (the OS
        will actually delete the data only once the final ``mmap`` is garbage collected, or the Python process exits).
        However writing into the file (updating the content) will immediately update the content of the ``mmap`` (and
        any ``numpy.ndarray`` built from it), which may cause subtle problems issues if the code assumes the data is
        immutable.
    """
    assert mode in ("r", "r+")
    assert size >= 0
    assert offset >= 0

    prefix = offset % ALLOCATIONGRANULARITY
    offset -= prefix
    if size > 0:
        size += prefix

    memory = MMAP_CACHE.get((path, mode, 0, 0))
    if memory is None and mode == "r":
        memory = MMAP_CACHE.get((path, "r+", 0, 0))
    if memory is not None:
        prefix += offset

    if memory is None and (offset > 0 or size > 0):
        memory = MMAP_CACHE.get((path, mode, offset, size))
        if memory is None and mode == "r":
            memory = MMAP_CACHE.get((path, "r+", offset, size))

    if memory is None:
        MMAP_CACHE[(path, mode, offset, size)] = memory = mmap(fd, size, prot=_PROT_OF_MODE[mode], offset=offset)

    if mode == "r+":
        return memoryview(memory)[prefix:]

    return memory[prefix:]


def bytes_as_ndarray(memory: Union[bytes, bytearray], *, name: str, shape: Collection[int], dtype: DType) -> np.ndarray:
    """
    View the bytes in ``memory`` as a ``numpy.ndarray`` with some ``shape`` and ``dtype``.

    If the bytes array has the wrong size, complain using the ``name``.
    """
    size = reduce(mul, shape, 1) * np.dtype(dtype).itemsize
    assert len(memory) == size, f"wrong size for the memory-mapped data in the {name}"

    array = np.frombuffer(memory, dtype=dtype)
    if len(shape) > 1:
        array = array.reshape(*shape)

    return array


def exists_memory_mapped_array(path: str) -> bool:
    """
    Returns whether the disk files for a memory-mapped ``numpy.ndarray`` exist.
    """
    return exists_path(f"{path}.yaml") and exists_path(f"{path}.array")


def remove_memory_mapped_array(path: str) -> None:
    """
    Remove the disk files (which must exist) for a memory-mapped ``numpy.ndarray``.
    """
    remove_path(f"{path}.yaml")
    remove_path(f"{path}.array")


def create_memory_mapped_array(path: str, shape: Union[int, Tuple[int, int]], dtype: DType) -> None:
    """
    Create new disk files for a memory-mapped ``numpy.ndarray`` of some ``shape`` and ``dtype`` in some ``path``.

    This will silently overwrite existing files. In particular, it will delete ``<path>.data``, ``<path>.indices``
    and/or ``<path>.indptr`` files if the exists.

    The array element type must be one of `.FIXED_DTYPES`, that is, one can't create a memory-mapped file of strings or
    objects of strange types.

    This creates two disk files:

    * ``<path>.array`` which contains just the data elements. For 2D data, this is always in `.ROW_MAJOR` layout.

    * ``<path>.yaml`` which contains a mapping with three keys:

      * ``version`` is a list of two integers, the major and minor format version numbers, to protect against
        future extensions of the format. This version of the library will generate ``[1, 0]`` files and will accept any
        files with a major version of ``1``.

      * ``dtype`` specifies the array element data type.

      * ``shape`` specifies the number of elements (as a sequence of one or two integers).

    This simple representation makes it easy for other systems to directly access the data. However, it basically makes
    it impossible to automatically report the type of the files (e.g., using the Linux ``file`` command).

    .. note::

        This just creates the files, filled with zeros. To access the data, you'll need to call
        `.open_memory_mapped_array`. Also, actual disk space for the data file is not allocated yet; all of the file
        except is a "hole". Actual disk pages are only created when the data is actually written for the first time.
        This makes creating large data files very quick, and even filling them with data is quick as long as the
        operating system doesn't need to actually flush the pages to the disk, which might even be deferred until after
        the program exits. In fact, if the file is deleted before the program exits, the data need not touch the disk at
        all.
    """
    assert is_dtype(dtype, FIXED_DTYPES), f"unsupported memory-mapped array dtype: {dtype}"

    if isinstance(shape, tuple):
        shape_list = list(shape)
    else:
        shape_list = [shape]

    assert 1 <= len(shape_list) <= 2 and all(
        isinstance(size, int) and size > 0 for size in shape_list
    ), f"invalid memory-mapped array shape: {shape}"

    with open(f"{path}.yaml", "w", encoding="utf-8") as yaml_file:
        dump_yaml(dict(version=[1, 0], dtype=str(dtype), shape=shape_list), yaml_file)

    with open(f"{path}.array", "wb") as data_file:
        data_file.truncate(reduce(mul, shape_list, 1) * np.dtype(dtype).itemsize)
        data_file.flush()
        data_file.close()

    for suffix in (".data", ".indices", ".indptr"):
        if exists_path(f"{path}.{suffix}"):
            remove_path(f"{path}.{suffix}")


def open_memory_mapped_array(path: str, mode: str) -> Union[Vector, DenseInRows]:
    """
    Open memory-mapped ``numpy.ndarray`` disk files.

    The ``mode`` must be one of ``r+`` for read-write access, or ``r`` for read-only access (which returns `.is_frozen`
    data).

    .. note::

        This only maps the data to memory, it does not actually read it from the disk. This makes opening large data
        files very quick, and even accessing the data may be fast as long as the operating system doesn't need to
        actually get the specific used pages from the disk (e.g. the pages were previously read, or were just written,
        so they are already in RAM). Therefore mapping a large array and only accessing small parts of it would be much
        faster than reading all the array to memory in advance.

        This does consume virtual address space to cover the whole data, but because the data is memory-mapped to a disk
        file, it allows accessing data that is larger than the physical RAM; the operating system brings in disk pages
        as necessary when data is accessed, and is free to flush/forget them to release space, so only a small subset of
        them must exist in RAM at any given time.
    """
    assert mode in ("r", "r+"), f"invalid memory-mapped mode: {mode}"

    # pylint: disable=duplicate-code

    with open(f"{path}.yaml", "r", encoding="utf-8") as yaml_file:
        metadata = load_yaml(yaml_file)
        assert (
            isinstance(metadata, dict)
            and isinstance(metadata.get("version"), list)
            and len(metadata["version"]) == 2
            and all(isinstance(version, int) and version >= 0 for version in metadata["version"])
            and isinstance(metadata.get("dtype"), str)
            and metadata["dtype"] in FIXED_DTYPES
            and isinstance(metadata.get("shape"), list)
            and 1 <= len(metadata["shape"]) <= 2
            and all(isinstance(size, int) and size > 0 for size in metadata["shape"])
        ), f"invalid YAML format for the memory-mapped array data: {path}"

    assert metadata["version"][0] == 1, (
        f"unsupported version: {metadata['version'][0]}.{metadata['version'][1]} "
        f"for the memory-mapped array metadata: {path}"
    )

    # pylint: enable=duplicate-code

    shape = metadata["shape"]
    array = _mmap_array(path=path + ".array", mode=mode, shape=shape, dtype=metadata["dtype"])
    if len(shape) == 1:
        return be_vector(array, dtype=metadata["dtype"])
    return be_dense_in_rows(array, dtype=metadata["dtype"])


def _mmap_array(*, path: str, fd: Optional[int] = None, mode: str, shape: Collection[int], dtype: DType) -> np.ndarray:
    if fd is None:
        # TRICKY: The ``file`` object has to live while ``mmap_file`` is called.
        file = open(path, mode=mode + "b")  # pylint: disable=unspecified-encoding,consider-using-with
        fd = file.fileno()  # pylint: disable=unspecified-encoding,consider-using-with

    return bytes_as_ndarray(
        mmap_file(path=path, fd=fd, mode=mode),
        name=f"file: {path}",
        shape=shape,
        dtype=dtype,
    )


def exists_memory_mapped_sparse(path: str) -> bool:
    """
    Returns whether the disk files for a memory-mapped `.SparseInRows` matrix exist.
    """
    return (
        exists_path(f"{path}.yaml")
        and exists_path(f"{path}.data")
        and exists_path(f"{path}.indices")
        and exists_path(f"{path}.indptr")
    )


def remove_memory_mapped_sparse(path: str) -> None:
    """
    Remove the disk files for a memory-mapped `.SparseInRows` matrix, if they exist.
    """
    remove_path(f"{path}.yaml")
    remove_path(f"{path}.data")
    remove_path(f"{path}.indices")
    remove_path(f"{path}.indptr")


def write_memory_mapped_sparse(path: str, sparse: SparseInRows) -> None:
    """
    Write the disk files for a memory-mapped `.SparseInRows` matrix, if they exist.

    This will silently overwrite existing files. In particular, it will delete a ``<path>.array`` file if one exists.

    This creates four disk files:

    * ``<path>.data`` which contains just the non-zero data elements.

    * ``<path>.indices`` which contains the column indices of the non-zero data elements.

    * ``<path>.indptr`` which contains the ranges of the entries of the non-zero data elements of the rows.

    * ``<path>.yaml`` which contains a mapping with six keys:

      * ``version`` is a list of two integers, the major and minor version numbers, to protect against
        future extensions of the format. This version of the library will generate ``[1, 0]`` files and will accept any
        files with a major version of ``1``.

      * ``data_dtype`` specifies the non-zero data array element data type.

      * ``indices_dtype`` specifies the non-zero column indices array element data type.

      * ``indptr_dtype`` specifies the rows entries ranges indptr array element data type.

      * ``shape`` specifies the number of elements (as a sequence of one or two integers).

      * ``nnz`` specifies the number of non-zero data elements.

    This simple representation makes it easy for other systems to directly access the data. However, it basically makes
    it impossible to automatically report the type of the files (e.g., using the Linux ``file`` command).
    """
    assert_data(is_sparse_in_rows(sparse, dtype=FIXED_DTYPES), AnyMajor.sparse_class_name, sparse, dtype=FIXED_DTYPES)
    assert_data(is_optimal(sparse), f"optimal {AnyMajor.sparse_class_name}", sparse, dtype=FIXED_DTYPES)

    with open(f"{path}.yaml", "w", encoding="utf-8") as yaml_file:
        dump_yaml(
            dict(
                version=[1, 0],
                data_dtype=str(sparse.data.dtype),
                indices_dtype=str(sparse.indices.dtype),
                indptr_dtype=str(sparse.indptr.dtype),
                nnz=sparse.nnz,
                shape=list(sparse.shape),
            ),
            yaml_file,
        )

    for (suffix, array) in (("data", sparse.data), ("indices", sparse.indices), ("indptr", sparse.indptr)):
        with open(f"{path}.{suffix}", "wb") as data_file:
            data_file.write(array.data)

    if exists_path(f"{path}.array"):
        remove_path(f"{path}.array")


def open_memory_mapped_sparse(path: str, mode: str) -> SparseInRows:
    """
    Open memory-mapped `.SparseInRows` matrix disk files.

    The ``mode`` must be one of ``w`` for read-write access, or ``r`` for read-only access.

    .. note:

        Take *great* care when modifying memory-mapped sparse matrices. In particular, since the memory mapped disk
        files have a fixed size, it *not* safe to modify the structure of the non-zero values.
    """
    assert exists_memory_mapped_sparse(path), f"missing memory-mapped sparse matrix: {path}"

    with open(f"{path}.yaml", "r", encoding="utf-8") as yaml_file:
        metadata = load_yaml(yaml_file)
        assert (
            isinstance(metadata, dict)
            and isinstance(metadata.get("version"), list)
            and len(metadata["version"]) == 2
            and all(isinstance(version, int) and version >= 0 for version in metadata["version"])
            and isinstance(metadata.get("data_dtype"), str)
            and metadata["data_dtype"] in FIXED_DTYPES
            and isinstance(metadata.get("indices_dtype"), str)
            and metadata["indices_dtype"] in FIXED_DTYPES
            and isinstance(metadata.get("indptr_dtype"), str)
            and metadata["indptr_dtype"] in FIXED_DTYPES
            and isinstance(metadata.get("nnz"), int)
            and metadata["nnz"] >= 0
            and isinstance(metadata.get("shape"), list)
            and 1 <= len(metadata["shape"]) <= 2
            and all(isinstance(size, int) and size > 0 for size in metadata["shape"])
        ), f"invalid YAML format for the memory-mapped sparse data: {path}"

    data_array = _mmap_array(path=f"{path}.data", mode=mode, shape=[metadata["nnz"]], dtype=metadata["data_dtype"])
    indices_array = _mmap_array(
        path=f"{path}.indices", mode=mode, shape=[metadata["nnz"]], dtype=metadata["indices_dtype"]
    )
    indptr_array = _mmap_array(
        path=f"{path}.indptr", mode=mode, shape=[metadata["shape"][0] + 1], dtype=metadata["indptr_dtype"]
    )

    return sp.csr_matrix((data_array, indices_array, indptr_array))


def allow_maximal_open_files() -> int:
    """
    Increase the maximal number of open files as much as possible, and return the updated limit.

    Every time you `.open_memory_mapped_array` or `.open_memory_mapped_sparse`, the relevant file(s) are memory-mapped
    which counts as "open files". The operating system restricts the maximal number of such open files per process. When
    you reach this limit you will see an error complaining about "too many open files" or "running out of file
    descriptors".

    Luckily, modern operating system allow for a large number of open files so this isn't a problem for common usage.

    If you do reach this limit, call this function which will use ``resource.setrlimit(resource.RLIMIT_NOFILE, ...)`` to
    increase the maximal number of open files to the maximum ("hard") limit allowed by the operating system, as opposed
    of to the lower "soft" limit used by default. This higher hard limit (the return value) is even higher in modern
    operating systems, and should be enough for most "uncommon" usage.

    If even that isn't enough, you should probably reflect on whether what you are trying to do makes sense in the first
    place. If you are certain it does, then most operating systems provide a way to raise the hard limit of open files
    to "any" value. This requires administrator privileges and is beyond the scope of this package.
    """
    _soft_limit, hard_limit = getrlimit(RLIMIT_NOFILE)
    setrlimit(RLIMIT_NOFILE, (hard_limit, hard_limit))
    return hard_limit
