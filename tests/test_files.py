"""
Test ``daf.storage.files``.
"""

from tempfile import TemporaryDirectory

import numpy as np
import scipy.sparse as sp  # type: ignore

from daf.storage.files import *  # pylint: disable=wildcard-import,unused-wildcard-import
from daf.typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from . import expect_raise

# pylint: disable=missing-function-docstring


def test_files_datum() -> None:
    with TemporaryDirectory() as directory:
        writer = FilesWriter(directory + "/test", name="test")

        assert not writer.has_datum("description")
        assert len(writer.datum_names()) == 0
        with expect_raise("missing datum: description in the storage: test"):
            writer.get_datum("description")

        writer.set_datum("description", "test files storage")

        for files in (FilesReader(directory + "/test", name="test"), writer):
            assert set(files.datum_names()) == set(["description"])
            assert files.has_datum("description")
            assert files.get_datum("description") == "test files storage"

            if not isinstance(files, FilesWriter):
                continue

            files.set_datum("description", "retest files storage", overwrite=True)
            assert files.get_datum("description") == "retest files storage"


def test_files_axis() -> None:
    with TemporaryDirectory() as directory:
        writer = FilesWriter(directory + "/test", name="test")

        assert not writer.has_axis("cell")
        assert len(writer.axis_names()) == 0
        with expect_raise("missing axis: cell in the storage: test"):
            writer.axis_size("cell")
        with expect_raise("missing axis: cell in the storage: test"):
            writer.axis_entries("cell")

        cell_names = freeze(as_array1d(["cell0", "cell1"]))
        writer.create_axis("cell", cell_names)

        for files in (FilesReader(directory + "/test", name="test"), writer):
            files = FilesWriter(directory + "/test", name="test")

            assert files.has_axis("cell")
            assert set(files.axis_names()) == set(["cell"])
            assert is_frozen(files.axis_entries("cell"))
            assert len(files.axis_entries("cell")) == len(cell_names)
            assert np.all(files.axis_entries("cell") == cell_names)

            if not isinstance(files, FilesWriter):
                continue

            with expect_raise("refuse to recreate the axis: cell in the storage: test"):
                files.create_axis("cell", cell_names)
            assert np.all(files.axis_entries("cell") == cell_names)


def test_files_array1d_of_str() -> None:
    with TemporaryDirectory() as directory:
        writer = FilesWriter(directory + "/test", name="test")

        assert not writer.has_array1d("cell;type")
        with expect_raise("missing axis: cell in the storage: test"):
            writer.array1d_names("cell")
        with expect_raise("missing axis: cell in the storage: test"):
            writer.get_array1d("cell;type")

        cell_names = freeze(as_array1d(["cell0", "cell1"]))
        writer.create_axis("cell", cell_names)

        assert not writer.has_array1d("cell;type")
        assert len(writer.array1d_names("cell")) == 0
        with expect_raise("missing 1D data: cell;type in the storage: test"):
            writer.get_array1d("cell;type")

        cell_types = freeze(as_array1d(["T", "B"]))
        writer.set_array1d("cell;type", cell_types)

        for files in (FilesReader(directory + "/test", name="test"), writer):
            assert files.has_array1d("cell;type")
            assert set(files.array1d_names("cell")) == set(["cell;type"])
            assert is_array1d(files.get_array1d("cell;type"), dtype=STR_DTYPE)
            assert is_frozen(files.get_array1d("cell;type"))
            assert np.all(files.get_array1d("cell;type") == cell_types)

            if not isinstance(files, FilesWriter):
                continue

            new_cell_types = freeze(as_array1d(["B", "T"]))
            files.set_array1d("cell;type", new_cell_types, overwrite=True)
            assert np.all(files.get_array1d("cell;type") == new_cell_types)


def test_files_array1d_of_bool() -> None:
    with TemporaryDirectory() as directory:
        writer = FilesWriter(directory + "/test", name="test")

        assert not writer.has_array1d("gene;significant")
        with expect_raise("missing axis: gene in the storage: test"):
            writer.array1d_names("gene")
        with expect_raise("missing axis: gene in the storage: test"):
            writer.get_array1d("gene;significant")

        gene_names = freeze(as_array1d(["gene0", "gene1", "gene2"]))
        writer.create_axis("gene", gene_names)

        assert not writer.has_array1d("gene;significant")
        assert len(writer.array1d_names("gene")) == 0
        with expect_raise("missing 1D data: gene;significant in the storage: test"):
            writer.get_array1d("gene;significant")

        significant_genes_mask = freeze(as_array1d([False, True, False]))
        writer.set_array1d("gene;significant", significant_genes_mask)

        for files in (FilesReader(directory + "/test", name="test"), writer):
            assert files.has_array1d("gene;significant")
            assert set(files.array1d_names("gene")) == set(["gene;significant"])
            assert is_array1d(files.get_array1d("gene;significant"), dtype="bool")
            assert is_frozen(files.get_array1d("gene;significant"))
            assert np.all(files.get_array1d("gene;significant") == significant_genes_mask)

            if not isinstance(files, FilesWriter):
                continue

            new_significant_genes_mask = freeze(as_array1d([False, False, True]))
            files.set_array1d("gene;significant", new_significant_genes_mask, overwrite=True)
            assert np.all(files.get_array1d("gene;significant") == new_significant_genes_mask)


def test_files_grid_of_int() -> None:
    with TemporaryDirectory() as directory:
        writer = FilesWriter(directory + "/test", name="test")

        assert not writer.has_data2d("cell,gene;UMIs")
        with expect_raise("missing rows axis: cell in the storage: test"):
            writer.data2d_names("cell,gene")
        with expect_raise("missing rows axis: cell in the storage: test"):
            writer.get_data2d("cell,gene;UMIs")

        cell_names = freeze(as_array1d(["cell0", "cell1"]))
        writer.create_axis("cell", cell_names)

        assert not writer.has_data2d("cell,gene;UMIs")
        with expect_raise("missing columns axis: gene in the storage: test"):
            writer.data2d_names("cell,gene")
        with expect_raise("missing columns axis: gene in the storage: test"):
            writer.get_data2d("cell,gene;UMIs")

        gene_names = freeze(as_array1d(["gene0", "gene1", "gene2"]))
        writer.create_axis("gene", gene_names)

        assert not writer.has_data2d("cell,gene;UMIs")
        assert len(writer.data2d_names("cell,gene")) == 0
        with expect_raise("missing 2D data: cell,gene;UMIs in the storage: test"):
            writer.get_data2d("cell,gene;UMIs")

        umis = freeze(be_array_in_rows(as_array2d([[0, 10, 90], [190, 10, 0]])))
        writer.set_grid("cell,gene;UMIs", umis)

        for files in (FilesReader(directory + "/test", name="test"), writer):
            assert files.has_data2d("cell,gene;UMIs")
            assert set(files.data2d_names("cell,gene")) == set(["cell,gene;UMIs"])
            assert is_array_in_rows(files.get_data2d("cell,gene;UMIs"))
            assert is_frozen(be_grid(files.get_data2d("cell,gene;UMIs")))
            assert fast_all_close(files.get_data2d("cell,gene;UMIs"), umis)

            if not isinstance(files, FilesWriter):
                continue

            new_umis = freeze(sp.csr_matrix([[90, 0, 10], [10, 0, 190]]))
            files.set_grid("cell,gene;UMIs", new_umis, overwrite=True)
            assert fast_all_close(files.get_data2d("cell,gene;UMIs"), new_umis)


def test_files_mmap_of_int() -> None:
    with TemporaryDirectory() as directory:
        writer = FilesWriter(directory + "/test", name="test")

        assert not writer.has_data2d("cell,gene;UMIs")
        with expect_raise("missing rows axis: cell in the storage: test"):
            writer.data2d_names("cell,gene")
        with expect_raise("missing rows axis: cell in the storage: test"):
            writer.get_data2d("cell,gene;UMIs")

        cell_names = freeze(as_array1d(["cell0", "cell1"]))
        writer.create_axis("cell", cell_names)

        assert not writer.has_data2d("cell,gene;UMIs")
        with expect_raise("missing columns axis: gene in the storage: test"):
            writer.data2d_names("cell,gene")
        with expect_raise("missing columns axis: gene in the storage: test"):
            writer.get_data2d("cell,gene;UMIs")

        gene_names = freeze(as_array1d(["gene0", "gene1", "gene2"]))
        writer.create_axis("gene", gene_names)

        assert not writer.has_data2d("cell,gene;UMIs")
        assert len(writer.data2d_names("cell,gene")) == 0
        with expect_raise("missing 2D data: cell,gene;UMIs in the storage: test"):
            writer.get_data2d("cell,gene;UMIs")

        with writer.create_array_in_rows("cell,gene;UMIs", dtype="int16") as umis:
            umis[:] = np.array([[0, 10, 90], [190, 10, 0]])

        for files in (FilesReader(directory + "/test", name="test"), writer):
            assert files.has_data2d("cell,gene;UMIs")
            assert set(files.data2d_names("cell,gene")) == set(["cell,gene;UMIs"])
            assert is_array_in_rows(files.get_data2d("cell,gene;UMIs"))
            assert is_frozen(be_grid(files.get_data2d("cell,gene;UMIs")))
            assert fast_all_close(files.get_data2d("cell,gene;UMIs"), umis)

            if not isinstance(files, FilesWriter):
                continue

            new_umis = freeze(sp.csr_matrix([[90, 0, 10], [10, 0, 190]]))
            files.set_grid("cell,gene;UMIs", new_umis, overwrite=True)
            assert fast_all_close(files.get_data2d("cell,gene;UMIs"), new_umis)


def test_files_grid_of_str() -> None:
    with TemporaryDirectory() as directory:
        writer = FilesWriter(directory + "/test", name="test")

        assert not writer.has_data2d("cell,gene;UMIs")
        with expect_raise("missing rows axis: cell in the storage: test"):
            writer.data2d_names("cell,gene")
        with expect_raise("missing rows axis: cell in the storage: test"):
            writer.get_data2d("cell,gene;UMIs")

        cell_names = freeze(as_array1d(["cell0", "cell1"]))
        writer.create_axis("cell", cell_names)

        assert not writer.has_data2d("cell,gene;UMIs")
        with expect_raise("missing columns axis: gene in the storage: test"):
            writer.data2d_names("cell,gene")
        with expect_raise("missing columns axis: gene in the storage: test"):
            writer.get_data2d("cell,gene;UMIs")

        gene_names = freeze(as_array1d(["gene0", "gene1", "gene2"]))
        writer.create_axis("gene", gene_names)

        assert not writer.has_data2d("cell,gene;UMIs")
        assert len(writer.data2d_names("cell,gene")) == 0
        with expect_raise("missing 2D data: cell,gene;UMIs in the storage: test"):
            writer.get_data2d("cell,gene;UMIs")

        levels = freeze(be_array_in_rows(as_array2d([[None, "Low", "High"], ["High", "Low", None]])))
        writer.set_grid("cell,gene;level", levels)

        for files in (FilesReader(directory + "/test", name="test"), writer):
            assert files.has_data2d("cell,gene;level")
            assert set(files.data2d_names("cell,gene")) == set(["cell,gene;level"])
            assert is_array_in_rows(files.get_data2d("cell,gene;level"))
            assert is_frozen(be_grid(files.get_data2d("cell,gene;level")))
            assert np.all(files.get_data2d("cell,gene;level") == levels)

            if not isinstance(files, FilesWriter):
                continue

            new_levels = freeze(be_array_in_rows(as_array2d([["High", None, "Low"], ["Low", None, "High"]])))
            files.set_grid("cell,gene;level", new_levels, overwrite=True)
            assert np.all(files.get_data2d("cell,gene;level") == new_levels)
