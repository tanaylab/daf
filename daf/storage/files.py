"""
This stores all the data as simple files in a trivial format in a single directory.

.. note::

    Do **not** directly modify the storage files after creating a `.FilesReader` or `.FilesWriter`. External
    modifications may or may not become visible, causing subtle problems.

    The only exception is that it is safe to create new axis/data files in the directory; these will not be reflected in
    any existing `.FilesReader` or `.FilesWriter` object. To access the new data, you will need to create a new
    `.FilesReader` or `.FilesWriter` object.

Directory Structure
-------------------

A ``daf`` storage directory will contain the following files:

* A single ``__daf__.yaml`` identifies the directory as containing ``daf`` data. This should contain a mapping with a
  single key ``version`` whose value must be a sequence of two integers, the major and minor format version numbers, to
  protect against future extensions of the format. . This version of the library will generate ``[1, 0]`` files and will
  accept any files with a major version of ``1``.

* Every 0D ("blob") data will be stored as a separate ``name.yaml`` file, to maximize human-readability of the data.

* For axes, there will be an ``axis.csv`` file with a single column with the axis name, containing the unique
  names of the entries along the axis.

* For 1D string data, there will be an ``axis;name.csv`` file with two columns, with the axis and the data name
  (if the name is identical to the axis, we suffix it with ``.value``). Any missing entries will be set to ``None``. The
  entries may be in any order, but `.FilesWriter` always writes them in the axis order.

* For 1D binary data, there will be an ``axis;name.yaml`` file and an ``axis;name.array`` containing the data
  (always in the axis entries order). See `.create_memory_mapped_array` for details in the (trivial) format of these
  files.

* For 2D string data, there will be a ``row,column;name.csv`` file with three columns, with the rows axis, columns
  axis, and data name (if the axis names are identical we suffix them with ``.row`` and ``.column``, and if the name is
  identical to either we suffix it with ``.value``). Any missing entries will be set to ``None``. The entries may be in
  any order, but `.FilesWriter` always writes them in `.ROW_MAJOR` order (skipping writing of ``None`` values).

* For 2D dense binary data, there will be a ``row,column;name.yaml`` file accompanied by a ``row,column;name.array``
  file (always in `.ROW_MAJOR` order based on the axis entries order). See `.create_memory_mapped_array` for details on
  the (trivial) format of these files.

* For 2D sparse binary data, there will be a ``row,column;name.yaml`` file accompanied by three files:
  ``row,column;name.sparse``, ``row,column;name.indices`` and ``row,column;name.indptr`` (always in `.ROW_MAJOR`,
  that is, CSR order, based on the axis entries order). See `.write_memory_mapped_sparse` for details on the (trivial)
  format of these files.

Other files, if any, are silently ignored.

.. note::

    The formats of non-binary (0D "blobs" and 1D/2D strings) data were chosen to maximize robustness as opposed to
    maximizing performance. This is an explicit design choice made since this performance has almost no impact on the
    data sets we created ``daf`` for (single-cell RNA sequencing data), and we saw the advantage of using simple
    self-describing text files to maximize the direct accessibility of the data for other system. If/when this becomes
    an issue, these formats can be replaced (at least as an option) with more efficient (but more opaque) formats (e.g.,
    it is even possible to memory-map a large 1D/2D string data file by using an accompanying array of offsets indices).
    Pull requests are welcome :-)

Motivation
----------

The intent here is **not** to define a "new format", but to use the trivial/standard existing formats to store the data
in files in a directory in the simplest way possible. Using a directory of separate files for separate data instead of a
complex single-file format such as ``H5AD`` has some advantages:

* It lazily loads the data, which allows "instantaneously" opening large data sets and quickly access just the data you
  need out of them.

* It uses memory-mapping to access 1D/2D binary (numeric/boolean) data, which allows efficient accessing very large data
  sets. This even allows accessing data sets larger than the physical RAM, though of course this will be slower. See
  `.memory_mapping` for further details on the (trivial) format of the memory-mapped files.

* It allows using standard file system tools to list the contents of the data set, delete data, and (if you know what
  you are doing and are *very* careful), copy (or share using hard or symbolic links) data between "compatible" data
  sets.

* It allows using advanced tools like ``make`` to automate computing annotations, which opens up many possibilities for
  writing and combining complex computation pipelines in a reproducible way.

* It allows multiple processes to write into the same directory, that is, parellelize and distribute computations, as
  long as they don't try to write the same data files.

There are of course also downsides to this approach:

* It requires you create an archive (using ``tar`` or ``zip`` or the like) if you want to send the data across the
  network. This isn't much of a hardship, as typically a data set consists of multiple files anyway, and it allows it to

* It uses one file descriptor per memory-mapped file (that is, any actually accessed 1D/2D data). If you access "too
  many" such data files at the same time, you may see an error saying something like "too many open files". This isn't
  typically a problem for normal usage. If you do encounter such an error, try calling `.allow_maximal_open_files` which
  will increase the limit as much as possible without requiring changing the operating system systems.

* When creating very large 2D data, you need to religiously use `.StorageWriter.create_array_in_rows` to ensure you
  do not lock up very large amounts of RAM. The error messages you will get when running out of RAM aren't great, but
  ideally you should see something like "failed to allocate <some very large amount> of memory". We assume here that
  creating new 1D data, even for very large data sets, is never a problem.
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
from ..typing import Array1D
from ..typing import ArrayInRows
from ..typing import Data2D
from ..typing import GridInRows
from ..typing import SparseInRows
from ..typing import as_array1d
from ..typing import be_array1d
from ..typing import be_array_in_rows
from ..typing import be_sparse_in_rows
from ..typing import freeze
from ..typing import is_array1d
from ..typing import is_dtype
from ..typing import is_grid_in_rows
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
        #: The format of the data, one of ``YAML`` (for 0D "blob" data), ``csv`` for 1D/2D string data, ``array`` for
        #: 1D/2D numeric/boolean data, or ``sparse`` for 2D numeric/boolean data.
        self.kind = kind


class FilesReader(_interface.StorageReader):
    """
    Implement the `.StorageReader` interface for simple files storage inside a ``path`` directory.

    Opens an existing ``daf`` storage directory.
    """

    def __init__(self, path: str, *, name: Optional[str] = None) -> None:
        super().__init__(name=name or path)

        #: The path of the directory containing the data files.
        self.path = path

        assert exists(path), f"missing storage directory: {path}"
        assert isdir(path), f"storage path is not a directory: {path}"

        # The 0D ("blob") data by name This will contain `.NotLoadedYet` for data that hasn't been loaded yet.
        self._data: Dict[str, Any] = {}

        # The axis entries, by axis name. This will contain `.NotLoadedYet` for data that hasn't been loaded yet.
        self._axes: Dict[str, Union[Array1D, NotLoadedYet]] = {}

        # The 1D data, by axis and name. This will contain `.NotLoadedYet` for data that hasn't been loaded yet.
        self._arrays: Dict[str, Dict[str, Union[Array1D, NotLoadedYet]]] = {}

        # The 2D data, by axis and name. This will contain `.NotLoadedYet` for data that hasn't been loaded yet.
        self._grids: Dict[Tuple[str, str], Dict[str, Union[GridInRows, NotLoadedYet]]] = {}

        self._scan_data()

    def _scan_data(self) -> None:  # pylint: disable=too-many-branches
        array1d_data: List[Tuple[str, str, str]] = []
        array2d_data: List[Tuple[Tuple[str, str], str, str]] = []

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

            if path.endswith(".yaml"):
                kind = "yaml"
            elif path.endswith(".csv"):
                kind = "csv"
            else:
                continue

            if ";" not in path:
                if kind == "yaml":
                    self._data[path[:-5]] = NotLoadedYet(kind)
                else:
                    self._axes[path[:-4]] = NotLoadedYet(kind)
                continue

            axis = path.split(";")[0]
            if "," not in axis:
                array1d_data.append((axis, path, kind))
                continue

            axes = tuple(axis.split(","))
            if len(axes) == 2:
                array2d_data.append((axes, path, kind))  # type: ignore
                continue

        for axis in self._axes:
            self._arrays[axis] = {}
            for other_axis in self._axes:
                self._grids[(axis, other_axis)] = {}

        for axis, path, kind in array1d_data:
            self._scan_array(axis, path, kind)

        for axes, path, kind in array2d_data:
            self._scan_grid(axes, path, kind)

    def _scan_array(self, axis: str, path: str, kind: str) -> None:
        assert (
            axis in self._axes
        ), f"missing the axis entries CSV file: {self.path}/{axis}.csv {axis} for the 1D data file: {self.path}/{path}"

        name = path[: -len(kind) - 1]
        if kind == "yaml":
            kind = self._memory_mapped_kind(name)
            assert kind != "sparse", f"unsupported sparse 1D data file: {self.path}/{name}.*"
        self._arrays[axis][name] = NotLoadedYet(kind)

    def _scan_grid(self, axes: Tuple[str, str], path: str, kind: str) -> None:
        rows_axis, columns_axis = axes
        assert rows_axis in self._axes, (
            f"missing the rows axis entries CSV file: {self.path}/{rows_axis}.csv "
            f"for the 2D data file: {self.path}/{path}"
        )
        assert columns_axis in self._axes, (
            f"missing the columns axis entries CSV file: {self.path}/{columns_axis}.csv "
            f"for the 2D data file: {self.path}/{path}"
        )

        name = path[: -len(kind) - 1]
        if kind == "yaml":
            kind = self._memory_mapped_kind(name)
        self._grids[axes][name] = NotLoadedYet(kind)

    # pylint: disable=duplicate-code

    def _datum_names(self) -> Collection[str]:
        return self._data.keys()

    def _has_datum(self, name: str) -> bool:
        return name in self._data

    # pylint: enable=duplicate-code

    def _get_datum(self, name: str) -> Any:
        datum = self._data[name]
        if isinstance(datum, NotLoadedYet):
            assert datum.kind == "yaml"
            with open(f"{self.path}/{name}.yaml", "r", encoding="utf-8") as yaml_file:
                self._data[name] = datum = load_yaml(yaml_file)
        return datum

    def _axis_names(self) -> Collection[str]:
        return self._axes.keys()

    def _has_axis(self, axis: str) -> bool:
        return axis in self._axes

    def _axis_size(self, axis: str) -> int:
        return len(self._axis_entries(axis))

    def _axis_entries(self, axis: str) -> Array1D:
        entries = self._axes[axis]
        if isinstance(entries, NotLoadedYet):
            assert entries.kind == "csv"
            frame = pd.read_csv(f"{self.path}/{axis}.csv", converters={0: str})
            assert (
                len(frame.columns) == 1 and frame.columns[0] == axis and is_dtype(frame.values.dtype, STR_DTYPE)
            ), f"invalid axis entries CSV file: {self.path}/{axis}.csv"
            self._axes[axis] = entries = freeze(as_array1d(frame))
        return entries

    # pylint: disable=duplicate-code

    def _array1d_names(self, axis: str) -> Collection[str]:
        return self._arrays[axis].keys()

    def _has_array1d(self, axis: str, name: str) -> bool:
        return name in self._arrays[axis]

    # pylint: enable=duplicate-code

    def _get_array1d(self, axis: str, name: str) -> Array1D:
        array = self._arrays[axis][name]
        if isinstance(array, NotLoadedYet):
            if array.kind == "csv":
                array = self._read_1d_csv(axis, name)
            else:
                assert array.kind in ("array", "csv")
                array = self._memory_map_data(name, "array")  # type: ignore
            assert is_array1d(array), f"invalid 1D data in the files: {self.path}/{name}.*"
            self._arrays[axis][name] = array
        return array

    # pylint: disable=duplicate-code

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:
        return self._grids[axes].keys()

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:
        return name in self._grids[axes]

    # pylint: enable=duplicate-code

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Data2D:
        grid = self._grids[axes][name]
        if isinstance(grid, NotLoadedYet):
            if grid.kind == "csv":
                grid = self._read_2d_csv(axes, name)
            else:
                assert grid.kind in ("array", "sparse", "csv")
                grid = self._memory_map_data(name, grid.kind)  # type: ignore
            assert is_grid_in_rows(grid), f"invalid 2D data in the files: {self.path}/{name}.*"
            self._grids[axes][name] = grid

        return grid

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

    def _memory_map_data(self, name: str, kind: str) -> Union[Array1D, ArrayInRows, SparseInRows]:
        if kind == "array":
            return _memory_mapping.open_memory_mapped_array(f"{self.path}/{name}", "r")
        assert kind in ("sparse", "array")
        return _memory_mapping.open_memory_mapped_sparse(f"{self.path}/{name}", "r")

    def _read_1d_csv(self, axis: str, name: str) -> Array1D:
        axis_header, name_header = _1d_csv_header(axis, name)
        csv_data = pd.read_csv(f"{self.path}/{name}.csv", converters={0: str, 1: str})
        assert list(csv_data.columns) == [axis_header, name_header], f"invalid 1D data CSV file: {self.path}/{name}.csv"
        array = np.full(self.axis_size(axis), None, dtype="object")
        series = pd.Series(array, index=self.axis_entries(axis))
        series[csv_data[axis_header]] = csv_data[name_header].values
        return freeze(be_array1d(array, dtype=STR_DTYPE))

    def _read_2d_csv(self, axes: Tuple[str, str], name: str) -> ArrayInRows:
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
        return be_array_in_rows(freeze(array), dtype=STR_DTYPE)


class FilesWriter(FilesReader, _interface.StorageWriter):
    """
    Implement the `.StorageWriter` interface for simple files storage inside a ``path`` directory.

    Opens an existing ``daf`` storage directory, or, if it does not exist, create a new empty one.
    """

    def __init__(self, path: str, *, name: Optional[str] = None) -> None:
        if not isdir(path):
            mkdir(path)
            with open(f"{path}/__daf__.yaml", "w", encoding="utf-8") as yaml_file:
                dump_yaml(dict(version=[1, 0]), yaml_file)

        super().__init__(path, name=name)

    def _set_datum(self, name: str, datum: Any) -> None:
        with open(f"{self.path}/{name}.yaml", "w", encoding="utf-8") as yaml_file:
            dump_yaml(datum, yaml_file)
        self._data[name] = NotLoadedYet("yaml")

    # pylint: disable=duplicate-code

    def _create_axis(self, axis: str, entries: Array1D) -> None:
        self._arrays[axis] = {}
        for other_axis in self._axes:
            self._grids[(axis, other_axis)] = {}
            self._grids[(other_axis, axis)] = {}
        self._axes[axis] = entries
        self._arrays[axis] = {}
        self._grids[(axis, axis)] = {}
        pd.DataFrame({axis: entries}).to_csv(f"{self.path}/{axis}.csv", index=False)

    # pylint: enable=duplicate-code

    def _set_array1d(self, axis: str, name: str, array1d: Array1D) -> None:
        if is_dtype(array1d.dtype, STR_DTYPE):
            self._write_1d_csv(axis, name, array1d)
            self._arrays[axis][name] = array1d
        else:
            self._write_array(name, array1d)
            self._arrays[axis][name] = NotLoadedYet("array")

    def _set_grid(self, axes: Tuple[str, str], name: str, grid: GridInRows) -> None:
        if isinstance(grid, sp.spmatrix):
            _memory_mapping.write_memory_mapped_sparse(
                f"{self.path}/{name}", be_sparse_in_rows(grid, dtype=FIXED_DTYPES)
            )
            self._grids[axes][name] = NotLoadedYet("sparse")
            return

        array = be_array_in_rows(grid)
        if is_dtype(array.dtype, STR_DTYPE):
            self._write_2d_csv(axes, name, array)
            self._grids[axes][name] = array
        else:
            self._write_array(name, array)
            self._grids[axes][name] = NotLoadedYet("array")

    @contextmanager
    def _create_array_in_rows(self, axes: Tuple[str, str], name: str, dtype: str) -> Generator[ArrayInRows, None, None]:
        path = f"{self.path}/{name}"
        shape = (self.axis_size(axes[0]), self.axis_size(axes[1]))
        _memory_mapping.create_memory_mapped_array(path, shape, dtype)
        array2d = be_array_in_rows(_memory_mapping.open_memory_mapped_array(path, "r+"))
        yield array2d
        self._grids[axes][name] = NotLoadedYet("array")

    def _write_1d_csv(self, axis: str, name: str, array1d: Array1D) -> None:
        entries = self.axis_entries(axis)

        mask = array1d != None  # pylint: disable=singleton-comparison
        array1d = array1d[mask]
        entries = entries[mask]

        axis_header, name_header = _1d_csv_header(axis, name)
        frame = pd.DataFrame({name_header: array1d}, index=entries)
        frame.to_csv(f"{self.path}/{name}.csv", index_label=axis_header)

    def _write_2d_csv(self, axes: Tuple[str, str], name: str, array2d: ArrayInRows) -> None:
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
                values = array2d[row_index, :]
                mask = values != None  # pylint: disable=singleton-comparison
                values = values[mask]
                if len(values) > 0:
                    entries = column_entries[mask]
                    frame = pd.DataFrame(dict(rows=np.full(len(values), row_name), columns=entries, value=values))
                    frame.to_csv(csv_file, index=False, header=False)

    @overload
    def _write_array(self, name: str, array: Array1D) -> None:
        ...

    @overload
    def _write_array(self, name: str, array: ArrayInRows) -> None:
        ...

    def _write_array(self, name: str, array: Union[Array1D, ArrayInRows]) -> None:
        path = f"{self.path}/{name}"
        _memory_mapping.create_memory_mapped_array(path, shape=array.shape, dtype=array.dtype)  # type: ignore
        mapped_array = _memory_mapping.open_memory_mapped_array(path, mode="r+")
        mapped_array[:] = array[:]  # TRICKY: Somehow this also works for 2D data.


def _1d_csv_header(axis: str, name: str) -> Tuple[str, str]:
    name = name.split(";")[1]
    if name == axis:
        name += ".value"
    return axis, name


def _2d_csv_header(axes: Tuple[str, str], name: str) -> Tuple[str, str, str]:
    name = name.split(";")[1]
    rows_axis, columns_axis = axes
    if rows_axis == columns_axis:
        rows_axis += ".row"
        columns_axis += ".column"
    if name in (rows_axis, columns_axis):
        name += ".value"
    return rows_axis, columns_axis, name


def _csv_quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"' if '"' in value else value
