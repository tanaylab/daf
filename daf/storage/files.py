"""
This stores all the data as simple files in a trivial format in a single directory.

The intent here is **not** to define a "new format", but to use the trivial/standard existing formats to store the data
in files in a directory in the simplest way possible.

.. note::

    Do **not** directly modify the storage files after creating a `.FilesReader` or `.FilesWriter`. External
    modifications may or may not become visible, causing subtle problems.

    The only exception is that it is safe to create new axis/data files in the directory; these will not be reflected in
    any existing `.FilesReader` or `.FilesWriter` object. To access the new data, you will need to create a new
    `.FilesReader` or `.FilesWriter` object.

**Directory Structure**

A ``daf`` storage directory will contain the following files:

* A single ``__daf__.yaml`` identifies the directory as containing ``daf`` data. This should contain a mapping with a
  single ``version`` key whose value must be a sequence of two integers, the major and minor format version numbers, to
  protect against future extensions of the format. . This version of the library will generate ``[1, 0]`` files and will
  accept any files with a major version of ``1``.

* Every 0D data will be stored as a separate ``name.yaml`` file, to maximize human-readability of the data.

* For axes, there will be an ``axis;.csv`` file with a single column with the axis name header, containing the unique
  names of the entries along the axis.

* For 1D string data, there will be an ``axis;name.csv`` file with two columns, with the axis and the data name header
  (if the name is identical to the axis, we suffix it with ``.value``). Any missing entries will be set to ``None``. The
  entries may be in any order, but `.FilesWriter` always writes them in the axis order (skiupping writing of ``None``
  values).

* For 1D binary data, there will be an ``axis;name.yaml`` file and an ``axis;name.array`` containing the data
  (always in the axis entries order). See `.create_memory_mapped_array` for details in the (trivial) format of these
  files.

* For 2D string data, there will be a ``row_axis,column_axis;name.csv`` file with three columns, with the rows axis,
  columns axis, and data name header (if the axis names are identical we suffix them with ``.row`` and ``.column``, and
  if the name is identical to either we suffix it with ``.value``). Any missing entries will be set to ``None``. The
  entries may be in any order, but `.FilesWriter` always writes them in `.ROW_MAJOR` order (skipping writing of ``None``
  values).

* For 2D `.Dense` binary data, there will be a ``row_axis,column_axis;name.yaml`` file accompanied by a
  ``row_axis,column_axis;name.array`` file (always in `.ROW_MAJOR` order based on the axis entries order). See
  `.create_memory_mapped_array` for details on the (trivial) format of these files.

* For 2D `.Sparse` binary data, there will be a ``row_axis,column_axis;name.yaml`` file accompanied by three files:
  ``row_axis,column_axis;name.data``, ``row_axis,column_axis;name.indices`` and ``row_axis,column_axis;name.indptr``
  (always in `.ROW_MAJOR`, that is, CSR order, based on the axis entries order). See `.write_memory_mapped_sparse` for
  details on the (trivial) format of these files.

Other files, if any, are silently ignored.

.. note::

    The formats of non-binary (0D data and 1D/2D strings) data were chosen to maximize robustness as opposed to
    maximizing performance. This is an explicit design choice made since this performance has almost no impact on the
    data sets we created ``daf`` for (single-cell RNA sequencing data), and we saw the advantage of using simple
    self-describing text files to maximize the direct accessibility of the data for non-``daf`` tools. If/when this
    becomes an issue, these formats can be replaced (at least as an option) with more efficient (but more opaque)
    formats.

    Similarly, we currently only support the most trivial format for binary data, to maximize their accessibility to
    non-``daf`` tools . In particular, no compressed format is available, which may be important for some data sets.

    If these restrictions are an issue for your data, you can use the `.h5fs` storage instead (even then, using
    compression will require some effort).

Using a directory of separate files for separate data instead of a complex single-file format such as ``h5fs`` has some
advantages:

* One can apply the multitude of file-based tools to the data. Putting aside the convenience of using ``bash`` or the
  Windows file explorer to simply see and manipulate the data, this allows using build tools like ``make`` to create
  complex reproducible multi-program computation pipelines, and automatically re-run just the necessary steps if/when
  some input data or control parameters are changed.

* Using memory-mapped files **never** creates an in-memory copy when accessing data, which is faster, and allows you to
  access data files larger than the available RAM (thanks to the wonders of paged virtual address spaces). You would
  need to **always** use `.StorageWriter.create_dense_in_rows` to create your data, though.

.. note::

    We have improved the implementation of ``daf`` storage for the `.h5fs` format, using the low-level ``h5py`` APIs, to
    also use memory mapping, which avoids copies "almost all the time". You still need to **always** use
    `.StorageWriter.create_dense_in_rows` to create your data, though.

There are of course also downsides to this approach:

* It requires you create an archive (using ``tar`` or ``zip`` or the like) if you want to send the data across the
  network. This isn't much of a hardship, as typically a data set consists of multiple files anyway. Using an archive
  also allows for compression, which is important when sending files across the network.

* It uses one file descriptor per memory-mapped file (that is, any actually accessed 1D/2D data). If you access "too
  many" such data files at the same time, you may see an error saying something like "too many open files". This isn't
  typically a problem for normal usage. If you do encounter such an error, try calling `.allow_maximal_open_files` which
  will increase the limit as much as possible without requiring changing operating system settings.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from contextlib import contextmanager
from os import listdir
from os import mkdir
from os.path import exists
from os.path import isdir
from typing import Any
from typing import Collection
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import overload

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore
from yaml import dump as dump_yaml  # type: ignore
from yaml import safe_load as load_yaml

from ..typing import FIXED_DTYPES
from ..typing import STR_DTYPE
from ..typing import DenseInRows
from ..typing import DType
from ..typing import Known1D
from ..typing import Known2D
from ..typing import MatrixInRows
from ..typing import SparseInRows
from ..typing import Vector
from ..typing import as_vector
from ..typing import be_dense_in_rows
from ..typing import be_sparse_in_rows
from ..typing import be_vector
from ..typing import has_dtype
from ..typing import is_matrix_in_rows
from ..typing import is_vector
from . import interface as _interface
from . import memory_mapping as _memory_mapping

# pylint: enable=duplicate-code,cyclic-import


__all__ = [
    "FilesReader",
    "FilesWriter",
]


class NotLoadedYet:  # pylint: disable=too-few-public-methods
    """
    Indicate some data was not loaded from the disk yet.
    """

    def __init__(self, kind: str) -> None:
        #: The format of the data, one of ``YAML`` (for 0D data), ``csv`` for 1D/2D string data, ``array`` for 1D/2D
        #: numeric/Boolean data, or ``sparse`` for 2D numeric/Boolean data.
        self.kind = kind


class FilesReader(_interface.StorageReader):
    """
    Implement the `.StorageReader` interface for simple files storage inside a ``path`` directory.

    If ``name`` is not specified, the ``path`` is used instead, adding a trailing ``/`` to ensure no ambiguity if/when
    the name is suffixed later. If the name ends with ``#``, we append the object id to it to make it unique.
    """

    def __init__(self, path: str, *, name: Optional[str] = None) -> None:
        while path.endswith("/"):
            path = path[:-1]
        name = name or path
        if name.endswith("#"):
            name += str(id(self))

        super().__init__(name=name)

        #: The path of the directory containing the data files.
        self.path = path

        assert exists(path), f"missing storage directory: {path}"
        assert isdir(path), f"storage path is not a directory: {path}"

        # The 0D data by name. This will contain `.NotLoadedYet` for data that hasn't been loaded yet.
        self._items: Dict[str, Any] = {}

        # The axis entries, by axis name. This will contain `.NotLoadedYet` for data that hasn't been loaded yet.
        self._axes: Dict[str, Union[Vector, NotLoadedYet]] = {}

        # The 1D data, by axis and name. This will contain `.NotLoadedYet` for data that hasn't been loaded yet.
        self._vectors: Dict[str, Dict[str, Union[Vector, NotLoadedYet]]] = {}

        # The 2D data, by axis and name. This will contain `.NotLoadedYet` for data that hasn't been loaded yet.
        self._matrices: Dict[Tuple[str, str], Dict[str, Union[MatrixInRows, NotLoadedYet]]] = {}

        self._scan_data()

    def _scan_data(self) -> None:  # pylint: disable=too-many-branches
        vector_data: List[Tuple[str, str, str]] = []
        dense_data: List[Tuple[Tuple[str, str], str, str]] = []

        # pylint: disable=duplicate-code

        with open(f"{self.path}/__daf__.yaml", "r", encoding="utf-8") as yaml_file:
            metadata = load_yaml(yaml_file)
            assert (
                isinstance(metadata, dict)
                and isinstance(metadata.get("version"), list)
                and len(metadata["version"]) == 2
                and all(isinstance(version, int) and version >= 0 for version in metadata["version"])
            ), f"invalid YAML format for the daf metadata in: {self.path}/__daf__.yaml"

        assert metadata["version"][0] == 1, (
            f"unsupported version: {metadata['version'][0]}.{metadata['version'][1]} "
            f"for the for the daf metadata in: {self.path}/__daf__.yaml"
        )

        # pylint: enable=duplicate-code

        for path in listdir(self.path):
            if path == "__daf__.yaml":
                continue

            if path.endswith(";.csv"):
                self._axes[path[:-5]] = NotLoadedYet("csv")
                continue

            if path.endswith(".yaml"):
                kind = "yaml"
            elif path.endswith(".csv"):
                kind = "csv"
            else:
                continue

            if ";" not in path:
                if kind == "yaml":
                    self._items[path[: -1 - len(kind)]] = NotLoadedYet(kind)
                continue

            axis = _interface.prefix(path, ";")
            if "," not in axis:
                vector_data.append((axis, path, kind))
                continue

            axes = tuple(axis.split(","))
            if len(axes) == 2:
                dense_data.append((axes, path, kind))  # type: ignore
                continue

        for axis in self._axes:
            self._vectors[axis] = {}
            for other_axis in self._axes:
                self._matrices[(axis, other_axis)] = {}

        for axis, path, kind in vector_data:
            self._scan_array(axis, path, kind)

        for axes, path, kind in dense_data:
            self._scan_matrix(axes, path, kind)

    def _scan_array(self, axis: str, path: str, kind: str) -> None:
        assert (
            axis in self._axes
        ), f"missing the axis entries CSV file: {self.path}/{axis}.csv for the 1D data file: {self.path}/{path}"

        name = path[: -1 - len(kind)]
        if kind == "yaml":
            kind = self._memory_mapped_kind(name)
            assert kind != "sparse", f"unsupported sparse 1D data file: {self.path}/{name}.*"
        self._vectors[axis][name] = NotLoadedYet(kind)

    def _scan_matrix(self, axes: Tuple[str, str], path: str, kind: str) -> None:
        rows_axis, columns_axis = axes
        assert rows_axis in self._axes, (
            f"missing the rows axis entries CSV file: {self.path}/{rows_axis};.csv "
            f"for the 2D data file: {self.path}/{path}"
        )
        assert columns_axis in self._axes, (
            f"missing the columns axis entries CSV file: {self.path}/{columns_axis};.csv "
            f"for the 2D data file: {self.path}/{path}"
        )

        name = path[: -len(kind) - 1]
        if kind == "yaml":
            kind = self._memory_mapped_kind(name)
        self._matrices[axes][name] = NotLoadedYet(kind)

    # pylint: disable=duplicate-code

    def _item_names(self) -> Collection[str]:
        return self._items.keys()

    def _has_item(self, name: str) -> bool:
        return name in self._items

    # pylint: enable=duplicate-code

    def _get_item(self, name: str) -> Any:
        item = self._items[name]
        if isinstance(item, NotLoadedYet):
            assert item.kind == "yaml"
            with open(f"{self.path}/{name}.yaml", "r", encoding="utf-8") as yaml_file:
                self._items[name] = item = load_yaml(yaml_file)
        return item

    # pylint: disable=duplicate-code

    def _axis_names(self) -> Collection[str]:
        return self._axes.keys()

    def _has_axis(self, axis: str) -> bool:
        return axis in self._axes

    def _axis_size(self, axis: str) -> int:
        return len(self._axis_entries(axis))

    # pylint: enable=duplicate-code

    def _axis_entries(self, axis: str) -> Known1D:
        entries = self._axes[axis]
        if isinstance(entries, NotLoadedYet):
            assert entries.kind == "csv"
            frame = pd.read_csv(f"{self.path}/{axis};.csv", converters={0: str})
            assert (
                len(frame.columns) == 1 and frame.columns[0] == axis and has_dtype(frame, STR_DTYPE)
            ), f"invalid axis entries CSV file: {self.path}/{axis};.csv"
            self._axes[axis] = entries = as_vector(frame)
        return entries

    # pylint: disable=duplicate-code

    def _data1d_names(self, axis: str) -> Collection[str]:
        return self._vectors[axis].keys()

    def _has_data1d(self, axis: str, name: str) -> bool:
        return name in self._vectors[axis]

    # pylint: enable=duplicate-code

    def _get_data1d(self, axis: str, name: str) -> Known1D:
        array = self._vectors[axis][name]
        if isinstance(array, NotLoadedYet):
            if array.kind == "csv":
                array = self._read_1d_csv(axis, name)
            else:
                assert array.kind in ("array", "csv")
                array = self._memory_map_data(name, "array")  # type: ignore
            assert is_vector(array), f"invalid 1D data in the files: {self.path}/{name}.*"
            self._vectors[axis][name] = array
        return array

    # pylint: disable=duplicate-code

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:
        return self._matrices[axes].keys()

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:
        return name in self._matrices[axes]

    # pylint: enable=duplicate-code

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Known2D:
        matrix = self._matrices[axes][name]
        if isinstance(matrix, NotLoadedYet):
            if matrix.kind == "csv":
                matrix = self._read_2d_csv(axes, name)
            else:
                assert matrix.kind in ("array", "sparse", "csv")
                matrix = self._memory_map_data(name, matrix.kind)  # type: ignore
            assert is_matrix_in_rows(matrix), f"invalid 2D data in the files: {self.path}/{name}.*"
            self._matrices[axes][name] = matrix

        return matrix

    def _memory_mapped_kind(self, name: str) -> str:
        is_array = _memory_mapping.exists_memory_mapped_array(f"{self.path}/{name}")
        is_sparse = _memory_mapping.exists_memory_mapped_sparse(f"{self.path}/{name}")

        assert (
            is_array or is_sparse
        ), f"the {self.path}/{name}.* files contain neither dense nor sparse memory mapped data"
        assert (
            not is_array or not is_sparse
        ), f"the {self.path}/{name}.* files contain either dense or sparse memory mapped data"

        return "array" if is_array else "sparse"

    def _memory_map_data(self, name: str, kind: str) -> Union[Vector, DenseInRows, SparseInRows]:
        if kind == "array":
            return _memory_mapping.open_memory_mapped_array(f"{self.path}/{name}", "r")
        assert kind in ("sparse", "array")
        return _memory_mapping.open_memory_mapped_sparse(f"{self.path}/{name}", "r")

    def _read_1d_csv(self, axis: str, name: str) -> Vector:
        axis_header, name_header = _1d_csv_header(axis, name)
        csv_data = pd.read_csv(f"{self.path}/{name}.csv", converters={0: str, 1: str})
        assert list(csv_data.columns) == [axis_header, name_header], f"invalid 1D data CSV file: {self.path}/{name}.csv"
        array = np.full(self.axis_size(axis), None, dtype="object")
        series = pd.Series(array, index=self.axis_entries(axis))
        series[csv_data[axis_header]] = csv_data[name_header].values
        return be_vector(array, dtype=STR_DTYPE)

    def _read_2d_csv(self, axes: Tuple[str, str], name: str) -> DenseInRows:
        rows_axis_header, columns_axis_header, name_header = _2d_csv_header(axes, name)

        csv_data = pd.read_csv(f"{self.path}/{name}.csv", converters={0: str, 1: str, 2: str})
        assert list(csv_data.columns) == [
            rows_axis_header,
            columns_axis_header,
            name_header,
        ], f"invalid 2D csv_data CSV file: {self.path}/{name}.csv"
        frame = csv_data.pivot_table(
            columns=rows_axis_header,
            index=columns_axis_header,
            values=name_header,
            fill_value="__no_value_in_csv__",
            aggfunc=lambda series: str(series.values[0]),
        )
        array = frame.values.T
        array[array == "__no_value_in_csv__"] = None
        return be_dense_in_rows(array, dtype=STR_DTYPE)


class FilesWriter(FilesReader, _interface.StorageWriter):
    """
    Implement the `.StorageWriter` interface for simple files storage inside a ``path`` directory.

    If ``name`` is not specified, the ``path`` is used instead, adding a trailing ``/`` to ensure no ambiguity if/when
    the name is suffixed later. If the name ends with ``#``, we append the object id to it to make it unique.
    """

    def __init__(self, path: str, *, name: Optional[str] = None) -> None:
        if not isdir(path):
            mkdir(path)
            with open(f"{path}/__daf__.yaml", "w", encoding="utf-8") as yaml_file:
                dump_yaml(dict(version=[1, 0]), yaml_file)

        super().__init__(path, name=name)

    def _set_item(self, name: str, item: Any) -> None:
        with open(f"{self.path}/{name}.yaml", "w", encoding="utf-8") as yaml_file:
            dump_yaml(item, yaml_file)
        self._items[name] = NotLoadedYet("yaml")

    # pylint: disable=duplicate-code

    def _create_axis(self, axis: str, entries: Vector) -> None:
        self._vectors[axis] = {}
        for other_axis in self._axes:
            self._matrices[(axis, other_axis)] = {}
            self._matrices[(other_axis, axis)] = {}
        self._axes[axis] = entries
        self._vectors[axis] = {}
        self._matrices[(axis, axis)] = {}
        pd.DataFrame({axis: entries}).to_csv(f"{self.path}/{axis};.csv", index=False)

    # pylint: enable=duplicate-code

    def _set_vector(self, axis: str, name: str, vector: Vector) -> None:
        if has_dtype(vector, STR_DTYPE):
            self._write_1d_csv(axis, name, vector)
            self._vectors[axis][name] = vector
        else:
            self._write_array(name, vector)
            self._vectors[axis][name] = NotLoadedYet("array")

    def _set_matrix(self, axes: Tuple[str, str], name: str, matrix: MatrixInRows) -> None:
        if isinstance(matrix, sp.spmatrix):
            _memory_mapping.write_memory_mapped_sparse(
                f"{self.path}/{name}", be_sparse_in_rows(matrix, dtype=FIXED_DTYPES)
            )
            self._matrices[axes][name] = NotLoadedYet("sparse")
            return

        array = be_dense_in_rows(matrix)
        if has_dtype(array, STR_DTYPE):
            self._write_2d_csv(axes, name, array)
            self._matrices[axes][name] = array
        else:
            self._write_array(name, array)
            self._matrices[axes][name] = NotLoadedYet("array")

    @contextmanager
    def _create_dense_in_rows(
        self, name: str, *, axes: Tuple[str, str], shape: Tuple[int, int], dtype: DType
    ) -> Generator[DenseInRows, None, None]:
        path = f"{self.path}/{name}"
        _memory_mapping.create_memory_mapped_array(path, shape, dtype)
        dense = be_dense_in_rows(_memory_mapping.open_memory_mapped_array(path, "r+"))
        yield dense
        self._matrices[axes][name] = NotLoadedYet("array")

    def _write_1d_csv(self, axis: str, name: str, vector: Vector) -> None:
        entries = as_vector(self.axis_entries(axis))

        mask = vector != None  # pylint: disable=singleton-comparison
        vector = vector[mask]
        entries = entries[mask]

        axis_header, name_header = _1d_csv_header(axis, name)
        frame = pd.DataFrame({name_header: vector}, index=entries)
        frame.to_csv(f"{self.path}/{name}.csv", index_label=axis_header)

    def _write_2d_csv(self, axes: Tuple[str, str], name: str, dense: DenseInRows) -> None:
        rows_axis_header, columns_axis_header, name_header = _2d_csv_header(axes, name)
        with open(f"{self.path}/{name}.csv", "w", encoding="utf-8") as csv_file:
            csv_file.write(_csv_quote(rows_axis_header))
            csv_file.write(",")
            csv_file.write(_csv_quote(columns_axis_header))
            csv_file.write(",")
            csv_file.write(_csv_quote(name_header))
            csv_file.write("\n")

            column_entries = self.axis_entries(axes[1])
            for row_index, row_name in enumerate(self.axis_entries(axes[0])):
                values = dense[row_index, :]
                mask = values != None  # pylint: disable=singleton-comparison
                values = values[mask]
                if len(values) > 0:
                    entries = column_entries[mask]
                    frame = pd.DataFrame(dict(rows=np.full(len(values), row_name), columns=entries, value=values))
                    frame.to_csv(csv_file, index=False, header=False)

    @overload
    def _write_array(self, name: str, array: Vector) -> None:
        ...

    @overload
    def _write_array(self, name: str, array: DenseInRows) -> None:
        ...

    def _write_array(self, name: str, array: Union[Vector, DenseInRows]) -> None:
        path = f"{self.path}/{name}"
        _memory_mapping.create_memory_mapped_array(path, shape=array.shape, dtype=array.dtype)  # type: ignore
        mapped_array = _memory_mapping.open_memory_mapped_array(path, mode="r+")
        mapped_array[:] = array[:]  # TRICKY: Somehow this also works for 2D data.


def _1d_csv_header(axis: str, name: str) -> Tuple[str, str]:
    name = _interface.suffix(name, ";")
    if name == axis:
        name += ".value"
    return axis, name


def _2d_csv_header(axes: Tuple[str, str], name: str) -> Tuple[str, str, str]:
    name = _interface.suffix(name, ";")
    rows_axis, columns_axis = axes
    if rows_axis == columns_axis:
        rows_axis += ".row"
        columns_axis += ".column"
    if name in (rows_axis, columns_axis):
        name += ".value"
    return rows_axis, columns_axis, name


def _csv_quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"' if '"' in value else value
