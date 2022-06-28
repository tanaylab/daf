"""
Store ``daf`` data in an ``h5fs`` file.

The intent here is **not** to define a "new format", but to use ``h5fs`` as simply as possible with a transparent naming
convention.

An ``h5fs`` file contains "groups" and "data sets", where a group contains groups and/or data sets, and each data set is
a single 1D/2D data array, both allowing for additional arbitrary metadata.

For maximal flexibility, the code here does not deal with creating or opening the ``h5fs`` file. Instead, given a file
opened or created using the ``h5py`` package, it allows using an arbitrary group in the file to hold all data for some
``daf`` storage. This allows multiple ``daf`` data sets to co-exist in the same ``h5fs`` file; the downside is that
given an ``h5fs`` file, you need to know the name of the group that contains the ``daf`` data set. Therefore, **by
convention**, if you name a file ``.h5df``, you are saying that it contains just one ``daf`` data set at the root group
of the file (that is, viewing the file object as a group). In contrast, if you name the file ``.h5fs``, you are saying
it may contain "anything", and you need to provide additional information such as "the group ``/foo/bar`` contains some
``daf`` data set, the group ``/foo/baz`` contains another ``daf`` data set, and the group ``/vaz`` contains non-``daf``
data".

.. note::

    Do **not** directly modify the content of a ``daf`` group in the ``h5fs`` after creating a `.H5fsReader` or
    `.H5fsWriter` for it. External modifications may or may not become visible, causing subtle problems.

**Group Structure**

A ``daf`` group inside an ``h5fs`` file will contain the following data sets:

* A single ``__daf__`` data set whose value is an array of two integers, the major and minor format version numbers, to
  protect against future extensions of the format. . This version of the library will generate ``[1, 0]`` files and will
  accept any files with a major version of ``1``.

* Every 0D data will be stored as an attribute of the ``__daf__`` data set.

* For axes, there will be an ``axis;`` data set containing the unique names of the entries along the axis.

* For 1D data, there will be an ``axis;property`` data set containing the data.

* For dense 2D data, there will be a ``row_axis,column_axis;property`` data set containing the data.

.. note::

    Storing 1D/2D data of strings in ``h5fs`` is built around the concept of a fixed number of bytes per element. This
    requires us to convert all strings to byte arrays before passing them on to ``h5fs`` (and the reverse when accessing
    the data). But ``numpy`` can't encode ``None`` in a byte array; instead it silently converts it to the 6-character
    string ``'None'``. To work around this, in ``h5fs``, we store the magic string value ``\\001`` to indicate the
    ``None`` value. So do **not** use this magic string value in arrays of strings you pass to ``daf``, and don't be
    surprised if you see this value if you access the data directly from ``h5fs``. Sigh.

* For sparse 2D data, there will be a group ``row_axis,column_axis;property`` which will contain three data sets:
  ``data``, ``indices`` and ``indptr``, needed to construct the sparse ``scipy.sparse.csr_matrix``. The group will have
  a ``shape`` attribute whose value is an array of two integers, the rows count and the columns count of the matrix.

Other data sets and/or groups, if any, are silently ignored.

.. note::

    Even though ``AnnData`` can also be used to access data in ``h5fs`` files, these files must be in a specific format
    (``h5ad``) which is **not** compatible with the format used here, and is much more restrictive; it isn't even
    possible to store multiple ``AnnData`` in a single ``h5ad`` file, because "reasons". See the `.anndata` module if
    you need to read or write ``h5ad`` files with ``daf``.

Using ``h5fs`` as a storage format has some advantages over using simple `.files` storage:

* The data is contained in a single file, making it easier to send it across a network.

* Using an ``h5fs`` file only consumes a single file descriptor, as opposed to one per memory-mapped 1D/2D data for the
  `.files` storage.

There are of course also downsides to this approach:

* All access to the data must be via the ``h5py`` API. This means that you can't apply any of the multitude of
  file-based tools to the data. Putting aside the loss of the convenience of using ``bash`` or the Windows file explorer
  to simply see and manipulate the data, this also rules out the possibility of using build tools like ``make`` to
  create complex reproducible multi-program computation pipelines, and automatically re-run just the necessary steps
  if/when some input data or control parameters are changed.

* Accessing data from ``h5fs`` creates an in-memory copy. To clarify, the ``h5py`` API does lazily load data only
  when it is accessed, and does allow to only access a slice of the data, but it **will** create an in-memory copy of
  that slice.

  When using ``daf`` to access ``h5fs`` data, you can't even ask it for just a slice, since ``daf`` always asks for the
  whole thing (in theory we could do something clever with views - we don't). If you are accessing large data, this will
  hurt performance; in extreme cases, when the data is bigger than the available RAM, the program will crash.

  All that said, the implementation here uses the low-level ``h5py`` APIs to memory-map 1D/2D data, so the above applies
  only to using ``h5fs`` through the "normal" ``h5py`` high-level API, which does not support memory-mapping (at least
  such time that https://github.com/h5py/h5py/issues/1607 is resolved).

.. note::

    The ``h5py`` API provides advanced storage features (e.g., chunking, compression). While ``daf`` doesn't support
    creating data using these features, it will happily read them. You can therefore either manually create ``daf`` data
    using these advanced features (following the above naming convention), or you can create the data using ``daf`` and
    then run a post-processing step that optimizes the storage format of the data as you see fit. However, if you do so,
    ``daf`` will no longer be able to memory-map the data, so for large data you may end up losing rather than gaining
    performance.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from typing import Collection
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import scipy.sparse as sp  # type: ignore
from h5py import AttributeManager  # type: ignore
from h5py import Dataset
from h5py import Group
from h5py import h5d
from h5py import h5p
from h5py import h5s
from h5py import h5t

from ..typing import INT_DTYPES
from ..typing import STR_DTYPE
from ..typing import DenseInRows
from ..typing import DType
from ..typing import Known1D
from ..typing import Known2D
from ..typing import MatrixInRows
from ..typing import Vector
from ..typing import be_dense_in_rows
from ..typing import has_dtype
from ..typing import is_vector
from . import interface as _interface
from . import memory_mapping as _memory_mapping

# pylint: enable=duplicate-code,cyclic-import
# pylint: disable=c-extension-no-member

__all__ = [
    "H5fsReader",
    "H5fsWriter",
]

# We store this to indicate a string is actually ``None``.
NONE_STRING = "\001"
NONE_BYTES = b"\001"


class H5fsReader(_interface.StorageReader):
    """
    Implement the `.StorageReader` interface for a ``group`` in an ``h5fs`` file.

    If the name ends with ``#``, we append the object id to it to make it unique.
    """

    def __init__(self, group: Group, *, name: str = "h5fs#") -> None:
        if name.endswith("#"):
            name += str(id(self))

        super().__init__(name=name)

        #: The ``h5fs`` group containing the data.
        self.group = group

        # The 0D data by name.
        self._items: AttributeManager

        # The axis entries, by axis name.
        self._axes: Dict[str, Dataset] = {}

        # The 1D data, by axis and name.
        self._vectors: Dict[str, Dict[str, Dataset]] = {}

        # The 2D data, by axis and name.
        self._matrices: Dict[Tuple[str, str], Dict[str, Union[Dataset, Group]]] = {}

        self._scan_data()

    def _scan_data(self) -> None:  # pylint: disable=too-many-branches
        vector_data: List[Tuple[str, str, Dataset]] = []
        matrix_data: List[Tuple[Tuple[str, str], str, Union[Dataset, Group]]] = []

        assert (
            "__daf__" in self.group
        ), f"the group: {self.group.name} of the h5fs file: {self.group.file.filename} does not contain daf data"
        items = self.group["__daf__"]
        self._items = items.attrs

        version = items[:]
        assert len(version) == 2 and is_vector(version, dtype=INT_DTYPES), (
            f"invalid daf version: {list(version)} "
            f"for the group: {self.group.name} "
            f"of the h5fs file: {self.group.file.filename}"
        )

        assert version[0] == 1, (
            f"unsupported daf version: {list(version)} "
            f"for the group: {self.group.name} "
            f"of the h5fs file: {self.group.file.filename}"
        )

        for name, data in sorted(self.group.items()):
            if name == "__daf__":
                continue

            if ";" not in name:
                continue

            if name.endswith(";"):
                axis = name[:-1]
                if isinstance(data, Dataset):
                    self._axes[axis] = data
                    self._vectors[axis] = {}
                    for other_axis in self._axes:
                        self._matrices[(axis, other_axis)] = {}
                        self._matrices[(other_axis, axis)] = {}
                continue

            axis = _interface.prefix(name, ";")
            if "," not in axis:
                if isinstance(data, Dataset):
                    vector_data.append((axis, name, data))
                continue

            axes = tuple(axis.split(","))
            if len(axes) == 2:
                if isinstance(data, Dataset) or (
                    isinstance(data, Group) and "data" in data and "indices" in data and "indptr" in data
                ):
                    matrix_data.append((axes, name, data))  # type: ignore
                continue

        for axis, name, data in vector_data:
            self._vectors[axis][name] = data

        for axes, name, data in matrix_data:
            self._matrices[axes][name] = data

    # pylint: disable=duplicate-code

    def _item_names(self) -> Collection[str]:
        return self._items.keys()

    def _has_item(self, name: str) -> bool:
        return name in self._items

    # pylint: enable=duplicate-code

    def _get_item(self, name: str) -> Any:
        return self._items[name]

    # pylint: disable=duplicate-code

    def _axis_names(self) -> Collection[str]:
        return self._axes.keys()

    def _has_axis(self, axis: str) -> bool:
        return axis in self._axes

    def _axis_size(self, axis: str) -> int:
        return self._axes[axis].size

    # pylint: enable=duplicate-code

    def _axis_entries(self, axis: str) -> Known1D:
        return self._axes[axis][:].astype("U")

    # pylint: disable=duplicate-code

    def _data1d_names(self, axis: str) -> Collection[str]:
        return self._vectors[axis].keys()

    def _has_data1d(self, axis: str, name: str) -> bool:
        return name in self._vectors[axis]

    # pylint: enable=duplicate-code

    def _get_data1d(self, axis: str, name: str) -> Known1D:
        dataset = self._vectors[axis][name]
        vector = self._get_ndarray(dataset, name)

        if "S" in str(vector.dtype):
            vector = vector.astype("U").astype("object")
            vector[vector == NONE_STRING] = None

        return vector

    # pylint: disable=duplicate-code

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:
        return self._matrices[axes].keys()

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:
        return name in self._matrices[axes]

    # pylint: enable=duplicate-code

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Known2D:
        data = self._matrices[axes][name]
        if isinstance(data, Dataset):
            matrix = self._get_ndarray(data, name)

            if "S" in str(matrix.dtype):
                matrix = matrix.astype("U").astype("object")
                matrix[matrix == NONE_STRING] = None

            return matrix

        assert isinstance(data, Group)

        data_dataset = data["data"]
        indices_dataset = data["indices"]
        indptr_dataset = data["indptr"]

        assert isinstance(data_dataset, Dataset)
        assert isinstance(indices_dataset, Dataset)
        assert isinstance(indptr_dataset, Dataset)

        data_ndarray = self._get_ndarray(data_dataset, name + ".data")
        indices_ndarray = self._get_ndarray(indices_dataset, name + ".indices")
        indptr_ndarray = self._get_ndarray(indptr_dataset, name + ".indptr")
        shape = tuple(data.attrs["shape"])

        return sp.csr_matrix((data_ndarray, indices_ndarray, indptr_ndarray), shape=shape)

    def _get_ndarray(self, dataset: Dataset, name: str) -> np.ndarray:
        if (
            dataset.id.get_create_plist().get_layout() != h5d.CONTIGUOUS
            or dataset.id.get_space_status() != h5d.SPACE_STATUS_ALLOCATED
        ):
            return dataset[:]

        dataset.flush()

        memory = _memory_mapping.mmap_file(
            path=self.group.file.filename,
            mode=self.group.file.mode,
            fd=self.group.file.id.get_vfd_handle(),
            offset=dataset.id.get_offset(),
            size=dataset.id.get_storage_size(),
        )

        return _memory_mapping.bytes_as_ndarray(
            memory,
            name=f"dataset for: {name} in the group: {self.group.name} in the h5fs file: {self.group.file.filename}",
            shape=dataset.shape,
            dtype=dataset.dtype,
        )


class H5fsWriter(H5fsReader, _interface.StorageWriter):
    """
    Implement the `.StorageWriter` interface for simple files storage inside an empty ``group`` in an ``h5fs`` file.

    If the name ends with ``#``, we append the object id to it to make it unique.
    """

    def __init__(self, group: Group, *, name: str = "h5fs#") -> None:
        if "__daf__" not in group:
            group.create_dataset("__daf__", (2,), dtype="int8")[:] = [1, 0]
        super().__init__(group, name=name)

    def _set_item(self, name: str, item: Any) -> None:
        self._items[name] = item

    # pylint: disable=duplicate-code

    def _create_axis(self, axis: str, entries: Vector) -> None:
        entries = entries.astype("S")  # type: ignore
        self._vectors[axis] = {}
        self._axes[axis] = dataset = self.group.create_dataset(f"{axis};", entries.shape, dtype=entries.dtype)
        dataset[:] = entries
        for other_axis in self._axes:
            self._matrices[(axis, other_axis)] = {}
            self._matrices[(other_axis, axis)] = {}

    # pylint: enable=duplicate-code

    def _set_vector(self, axis: str, name: str, vector: Vector) -> None:
        if name in self.group:
            del self.group[name]

        if has_dtype(vector, STR_DTYPE):
            none_mask = vector == None  # pylint: disable=singleton-comparison
            vector = vector.astype("S")  # type: ignore
            vector[none_mask] = NONE_BYTES
        self._vectors[axis][name] = dataset = self.group.create_dataset(name, vector.shape, dtype=vector.dtype)
        dataset[:] = vector[:]

    def _set_matrix(self, axes: Tuple[str, str], name: str, matrix: MatrixInRows) -> None:
        if name in self.group:
            del self.group[name]

        if isinstance(matrix, sp.spmatrix):
            data = self.group.create_group(name)
            data.attrs["shape"] = matrix.shape
            data.create_dataset("data", matrix.data.shape, dtype=matrix.data.dtype)[:] = matrix.data[:]
            data.create_dataset("indices", matrix.indices.shape, dtype=matrix.indices.dtype)[:] = matrix.indices[:]
            data.create_dataset("indptr", matrix.indptr.shape, dtype=matrix.indptr.dtype)[:] = matrix.indptr[:]
        else:
            if has_dtype(matrix, STR_DTYPE):
                none_mask = matrix == None  # pylint: disable=singleton-comparison
                matrix = matrix.astype("S")  # type: ignore
                matrix[none_mask] = NONE_BYTES
            data = self.group.create_dataset(name, matrix.shape, dtype=matrix.dtype)
            data[:] = matrix[:]

        self._matrices[axes][name] = data

    @contextmanager
    def _create_dense_in_rows(
        self, name: str, *, axes: Tuple[str, str], shape: Tuple[int, int], dtype: DType
    ) -> Generator[DenseInRows, None, None]:
        space_id = h5s.create_simple(shape)
        plist = h5p.create(h5p.DATASET_CREATE)
        plist.set_layout(h5d.CONTIGUOUS)
        plist.set_fill_time(h5d.FILL_TIME_NEVER)
        plist.set_alloc_time(h5d.ALLOC_TIME_EARLY)
        dataset_id = h5d.create(
            self.group.id,
            bytes(name, encoding="utf-8"),
            h5t.py_create(np.dtype(dtype)),
            space_id,
            plist,
        )
        dataset = Dataset(dataset_id)

        matrix = be_dense_in_rows(self._get_ndarray(dataset, name), shape=shape)
        yield matrix
        self._matrices[axes][name] = dataset
