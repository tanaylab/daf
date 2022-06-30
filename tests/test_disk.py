"""
Test ``daf.storage.files`` and ``daf.storage.h5fs``.
"""

from tempfile import TemporaryDirectory
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import scipy.sparse as sp  # type: ignore
from h5py import File  # type: ignore

from daf.storage.files import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.h5fs import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.storage.interface import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_description
from . import expect_raise

# pylint: disable=missing-function-docstring


def writers(directory: str) -> List[Tuple[StorageWriter, Callable[[], StorageReader]]]:
    h5fs = File(directory + "/test.h5fs", "w")
    h5fs_writer = H5fsWriter(h5fs, name="test")
    files_writer = FilesWriter(directory + "/test", name="test")
    return [
        (files_writer, lambda: FilesReader(directory + "/test", name="test")),
        (h5fs_writer, lambda: H5fsReader(File(directory + "/test.h5fs", "r+"))),
    ]


def test_disk_item() -> None:
    with TemporaryDirectory() as directory:
        for writer, make_reader in writers(directory):

            assert not writer.has_item("description")
            assert len(writer.item_names()) == 0
            with expect_raise("missing item: description in the storage: test"):
                writer.get_item("description")

            writer.set_item("description", "test disk storage")

            for reader in (make_reader(), writer):
                assert set(reader.item_names()) == set(["description"])
                assert reader.has_item("description")
                assert reader.get_item("description") == "test disk storage"

            writer.set_item("description", "retest writer storage", overwrite=True)
            assert writer.get_item("description") == "retest writer storage"

            expect_description(
                writer,
                detail=True,
                expected=f"""
                test:
                  class: {writer.__class__.__module__}.{writer.__class__.__qualname__}
                  axes: {{}}
                  data:
                    description: builtins.str = retest writer storage
                """,
            )


def test_disk_axis() -> None:
    with TemporaryDirectory() as directory:
        for writer, make_reader in writers(directory):

            assert not writer.has_axis("cell")
            assert len(writer.axis_names()) == 0
            with expect_raise("missing axis: cell in the storage: test"):
                writer.axis_size("cell")
            with expect_raise("missing axis: cell in the storage: test"):
                writer.axis_entries("cell")

            cell_names = freeze(as_vector(["cell0", "cell1"]))
            writer.create_axis("cell", cell_names)

            for reader in (make_reader(), writer):
                reader = FilesWriter(directory + "/test", name="test")

                assert reader.has_axis("cell")
                assert set(reader.axis_names()) == set(["cell"])
                assert len(reader.axis_entries("cell")) == len(cell_names)
                assert np.all(reader.axis_entries("cell") == cell_names)

            with expect_raise("refuse to recreate the axis: cell in the storage: test"):
                writer.create_axis("cell", cell_names)
            assert np.all(writer.axis_entries("cell") == cell_names)

            frozen = "frozen " if isinstance(writer, FilesWriter) else ""
            expect_description(
                writer,
                detail=True,
                expected=f"""
                test:
                  class: {writer.__class__.__module__}.{writer.__class__.__qualname__}
                  axes:
                    cell: {frozen}1D numpy.ndarray of 2 of <U5
                  data: {{}}
                """,
            )


def test_disk_vector_of_str() -> None:
    with TemporaryDirectory() as directory:
        for writer, make_reader in writers(directory):

            assert not writer.has_data1d("cell#type")
            with expect_raise("missing axis: cell in the storage: test"):
                writer.data1d_names("cell")
            with expect_raise("missing axis: cell in the storage: test"):
                writer.get_data1d("cell#type")

            cell_names = freeze(as_vector(["cell0", "cell1"]))
            writer.create_axis("cell", cell_names)

            assert not writer.has_data1d("cell#type")
            assert len(writer.data1d_names("cell")) == 0
            with expect_raise("missing 1D data: cell#type in the storage: test"):
                writer.get_data1d("cell#type")

            cell_types = freeze(as_vector(["T", "B"]))
            writer.set_vector("cell#type", cell_types)

            for reader in (make_reader(), writer):
                assert reader.has_data1d("cell#type")
                assert set(reader.data1d_names("cell")) == set(["cell#type"])
                assert is_vector(reader.get_data1d("cell#type"), dtype=STR_DTYPE)
                assert np.all(reader.get_data1d("cell#type") == cell_types)

            new_cell_types = freeze(as_vector(["B", "T"]))
            writer.set_vector("cell#type", new_cell_types, overwrite=True)
            assert np.all(writer.get_data1d("cell#type") == new_cell_types)

            # pylint: disable=duplicate-code

            frozen, dtype = ("frozen ", "<U1") if isinstance(writer, FilesWriter) else ("", "object")
            expect_description(
                writer,
                detail=True,
                expected=f"""
                test:
                  class: {writer.__class__.__module__}.{writer.__class__.__qualname__}
                  axes:
                    cell: {frozen}1D numpy.ndarray of 2 of <U5
                  data:
                    cell#type: {frozen}1D numpy.ndarray of 2 of {dtype}
                """,
            )

            # pylint: enable=duplicate-code


def test_disk_vector_of_bool() -> None:
    with TemporaryDirectory() as directory:
        for writer, make_reader in writers(directory):

            assert not writer.has_data1d("gene#significant")
            with expect_raise("missing axis: gene in the storage: test"):
                writer.data1d_names("gene")
            with expect_raise("missing axis: gene in the storage: test"):
                writer.get_data1d("gene#significant")

            gene_names = freeze(as_vector(["gene0", "gene1", "gene2"]))
            writer.create_axis("gene", gene_names)

            assert not writer.has_data1d("gene#significant")
            assert len(writer.data1d_names("gene")) == 0
            with expect_raise("missing 1D data: gene#significant in the storage: test"):
                writer.get_data1d("gene#significant")

            significant_genes_mask = freeze(as_vector([False, True, False]))
            writer.set_vector("gene#significant", significant_genes_mask)

            for reader in (make_reader(), writer):
                assert reader.has_data1d("gene#significant")
                assert set(reader.data1d_names("gene")) == set(["gene#significant"])
                assert is_vector(reader.get_data1d("gene#significant"), dtype="bool")
                assert np.all(reader.get_data1d("gene#significant") == significant_genes_mask)
                if isinstance(writer, FilesWriter):
                    assert is_frozen(be_vector(reader.get_data1d("gene#significant")))

            new_significant_genes_mask = freeze(as_vector([False, False, True]))
            writer.set_vector("gene#significant", new_significant_genes_mask, overwrite=True)
            assert np.all(writer.get_data1d("gene#significant") == new_significant_genes_mask)

            frozen = "frozen " if isinstance(writer, FilesWriter) else ""
            expect_description(
                writer,
                detail=True,
                expected=f"""
                test:
                  class: {writer.__class__.__module__}.{writer.__class__.__qualname__}
                  axes:
                    gene: {frozen}1D numpy.ndarray of 3 of <U5
                  data:
                    gene#significant: {frozen}1D numpy.ndarray of 3 of bool (1 true, 33.33%)
                """,
            )


def test_disk_matrix_of_int() -> None:
    with TemporaryDirectory() as directory:
        for writer, make_reader in writers(directory):

            assert not writer.has_data2d("cell,gene#UMIs")
            with expect_raise("missing axis: cell in the storage: test"):
                writer.data2d_names("cell,gene")
            with expect_raise("missing axis: cell in the storage: test"):
                writer.get_data2d("cell,gene#UMIs")

            cell_names = freeze(as_vector(["cell0", "cell1"]))
            writer.create_axis("cell", cell_names)

            assert not writer.has_data2d("cell,gene#UMIs")
            with expect_raise("missing axis: gene in the storage: test"):
                writer.data2d_names("cell,gene")
            with expect_raise("missing axis: gene in the storage: test"):
                writer.get_data2d("cell,gene#UMIs")

            gene_names = freeze(as_vector(["gene0", "gene1", "gene2"]))
            writer.create_axis("gene", gene_names)

            assert not writer.has_data2d("cell,gene#UMIs")
            assert len(writer.data2d_names("cell,gene")) == 0
            with expect_raise("missing 2D data: cell,gene#UMIs in the storage: test"):
                writer.get_data2d("cell,gene#UMIs")

            umis = freeze(be_dense_in_rows(as_dense([[0, 10, 90], [190, 10, 0]])))
            writer.set_matrix("cell,gene#UMIs", umis)

            for reader in (make_reader(), writer):
                assert reader.has_data2d("cell,gene#UMIs")
                assert set(reader.data2d_names("cell,gene")) == set(["cell,gene#UMIs"])
                assert is_dense_in_rows(reader.get_data2d("cell,gene#UMIs"))
                assert fast_all_close(reader.get_data2d("cell,gene#UMIs"), umis)
                if isinstance(writer, FilesWriter):
                    assert is_frozen(be_matrix(reader.get_data2d("cell,gene#UMIs")))

            new_umis = freeze(sp.csr_matrix([[90, 0, 10], [10, 0, 190]]))
            writer.set_matrix("cell,gene#UMIs", new_umis, overwrite=True)
            assert fast_all_close(writer.get_data2d("cell,gene#UMIs"), new_umis)

            # pylint: disable=duplicate-code

            frozen = "frozen " if isinstance(writer, FilesWriter) else ""
            expect_description(
                writer,
                detail=True,
                expected=f"""
                test:
                  class: {writer.__class__.__module__}.{writer.__class__.__qualname__}
                  axes:
                    cell: {frozen}1D numpy.ndarray of 2 of <U5
                    gene: {frozen}1D numpy.ndarray of 3 of <U5
                  data:
                    cell,gene#UMIs: {frozen}scipy.sparse.csr_matrix of 2x3 of int64 with 66.67% nnz
                """,
            )

            # pylint: enable=duplicate-code


def test_disk_mmap_of_int() -> None:
    with TemporaryDirectory() as directory:
        for writer, make_reader in writers(directory):

            assert not writer.has_data2d("cell,gene#UMIs")
            with expect_raise("missing axis: cell in the storage: test"):
                writer.data2d_names("cell,gene")
            with expect_raise("missing axis: cell in the storage: test"):
                writer.get_data2d("cell,gene#UMIs")

            cell_names = freeze(as_vector(["cell0", "cell1"]))
            writer.create_axis("cell", cell_names)

            assert not writer.has_data2d("cell,gene#UMIs")
            with expect_raise("missing axis: gene in the storage: test"):
                writer.data2d_names("cell,gene")
            with expect_raise("missing axis: gene in the storage: test"):
                writer.get_data2d("cell,gene#UMIs")

            gene_names = freeze(as_vector(["gene0", "gene1", "gene2"]))
            writer.create_axis("gene", gene_names)

            assert not writer.has_data2d("cell,gene#UMIs")
            assert len(writer.data2d_names("cell,gene")) == 0
            with expect_raise("missing 2D data: cell,gene#UMIs in the storage: test"):
                writer.get_data2d("cell,gene#UMIs")

            with writer.create_dense_in_rows("cell,gene#UMIs", dtype="int16") as umis:
                umis[:] = np.array([[0, 10, 90], [190, 10, 0]])

            for reader in (make_reader(), writer):
                assert reader.has_data2d("cell,gene#UMIs")
                assert set(reader.data2d_names("cell,gene")) == set(["cell,gene#UMIs"])
                assert is_dense_in_rows(reader.get_data2d("cell,gene#UMIs"))
                assert fast_all_close(reader.get_data2d("cell,gene#UMIs"), umis)
                if isinstance(writer, FilesWriter):
                    assert is_frozen(be_matrix(reader.get_data2d("cell,gene#UMIs")))

            new_umis = freeze(sp.csr_matrix([[90, 0, 10], [10, 0, 190]]))
            writer.set_matrix("cell,gene#UMIs", new_umis, overwrite=True)
            assert fast_all_close(writer.get_data2d("cell,gene#UMIs"), new_umis)

            frozen = "frozen " if isinstance(writer, FilesWriter) else ""
            expect_description(
                writer,
                detail=True,
                expected=f"""
                test:
                  class: {writer.__class__.__module__}.{writer.__class__.__qualname__}
                  axes:
                    cell: {frozen}1D numpy.ndarray of 2 of <U5
                    gene: {frozen}1D numpy.ndarray of 3 of <U5
                  data:
                    cell,gene#UMIs: {frozen}scipy.sparse.csr_matrix of 2x3 of int64 with 66.67% nnz
                """,
            )


def test_disk_matrix_of_str() -> None:
    with TemporaryDirectory() as directory:
        for writer, make_reader in writers(directory):

            assert not writer.has_data2d("cell,gene#UMIs")
            with expect_raise("missing axis: cell in the storage: test"):
                writer.data2d_names("cell,gene")
            with expect_raise("missing axis: cell in the storage: test"):
                writer.get_data2d("cell,gene#UMIs")

            cell_names = freeze(as_vector(["cell0", "cell1"]))
            writer.create_axis("cell", cell_names)

            assert not writer.has_data2d("cell,gene#UMIs")
            with expect_raise("missing axis: gene in the storage: test"):
                writer.data2d_names("cell,gene")
            with expect_raise("missing axis: gene in the storage: test"):
                writer.get_data2d("cell,gene#UMIs")

            gene_names = freeze(as_vector(["gene0", "gene1", "gene2"]))
            writer.create_axis("gene", gene_names)

            assert not writer.has_data2d("cell,gene#UMIs")
            assert len(writer.data2d_names("cell,gene")) == 0
            with expect_raise("missing 2D data: cell,gene#UMIs in the storage: test"):
                writer.get_data2d("cell,gene#UMIs")

            levels = freeze(be_dense_in_rows(as_dense([[None, "Low", "High"], ["High", "Low", None]])))
            writer.set_matrix("cell,gene#level", levels)

            for reader in (make_reader(), writer):
                assert reader.has_data2d("cell,gene#level")
                assert set(reader.data2d_names("cell,gene")) == set(["cell,gene#level"])
                assert is_dense_in_rows(reader.get_data2d("cell,gene#level"))
                assert np.all(reader.get_data2d("cell,gene#level") == levels)

            new_levels = freeze(be_dense_in_rows(as_dense([["High", None, "Low"], ["Low", None, "High"]])))
            writer.set_matrix("cell,gene#level", new_levels, overwrite=True)
            assert np.all(writer.get_data2d("cell,gene#level") == new_levels)

            frozen = "frozen " if isinstance(writer, FilesWriter) else ""
            expect_description(
                writer,
                detail=True,
                expected=f"""
                test:
                  class: {writer.__class__.__module__}.{writer.__class__.__qualname__}
                  axes:
                    cell: {frozen}1D numpy.ndarray of 2 of <U5
                    gene: {frozen}1D numpy.ndarray of 3 of <U5
                  data:
                    cell,gene#level: {frozen}row-major numpy.ndarray of 2x3 of object
                """,
            )
