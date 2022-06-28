"""
This stores the data inside an ``AnnData`` `AnnData <https://pypi.org/project/anndata>`_ object.

Since ``AnnData`` is not really powerful enough to satisfy our needs (this was the main motivation for creationg
``daf``), this is mainly used to interface with other systems.

When accessing existing ``AnnData`` objects, you can either wrap it with an `.AnnDataWriter` to allow modifying it
through ``daf``, or with an `.AnnDataReader` object to provide just read-only access. In both cases, you will see the
hard-wired ``obs``, ``var`` and ``X`` names. As a convenience, you can call `.anndata_as_storage` instead, which will
create an `.AnnDataReader` and wrap it with a `.StorageView` to rename these to more meaningful names.

To create a new ``AnnData`` object, call `.storage_as_anndata`, giving it a `.StorageReader` that exposes only the data
you wish to place in the result; this is typically done using a `.StorageView`, which also renames some meaningful axes
and data names to ``obs``, ``var`` and ``X``.

**Representation**

We use the following scheme to map between ``daf`` data and ``AnnData`` fields:

* 0D data is easy, it is stored in the ``uns`` field of ``AnnData``.

* Axes other than ``obs`` and ``var`` require us to store their entries, which we do by using an ``uns`` entry with the
  name ``axis;``.

* 1D data other than per-``obs`` and per-``var`` data is stored in an ``uns`` entry named ``axis;property``.

* 2D data for ``obs`` and ``var`` axes is stored in the ``X``, ``layers``, ``obsp`` or ``varp`` ``AnnData`` fields, as
  appropriate.

* 2D data for either ``obs`` or ``var`` and another axis is stored as a set of 1D annotations in the ``obs`` or ``var``
  ``AnnData`` fields, one for each axis entry, named ``other_axis=entry;property``. It is debatable whether this makes
  it easier or harder to access this data in systems that directly use ``AnnData``, but it is at least "technically
  correct".

* 2D data where neither axis is ``obs`` or ``var`` is stored in an ``uns`` entry named
  ``row_axis,column_axis;property``.
"""

# pylint: disable=duplicate-code

from typing import Any
from typing import Collection
from typing import Tuple

import numpy as np
from anndata import AnnData  # type: ignore

from ..typing import COLUMN_MAJOR
from ..typing import Known1D
from ..typing import Known2D
from ..typing import MatrixInRows
from ..typing import Vector
from ..typing import as_layout
from ..typing import as_vector
from ..typing import dtype_of
from ..typing import freeze
from ..typing import is_matrix
from ..typing import optimize
from . import interface as _interface
from . import views as _views

# pylint: enable=duplicate-code

__all__ = ["AnnDataReader", "AnnDataWriter", "storage_as_anndata", "anndata_as_storage"]


class AnnDataReader(_interface.StorageReader):
    """
    Implement the `.StorageReader` interface for ``AnnData``.

    If the ``name`` ends with ``#``, we append the object id to it to make it unique.

    .. note::

        Do **not** modify the wrapped ``AnnData`` after creating a reader. Modifications may or may not be visible in
        the reader, causing subtle problems.
    """

    def __init__(self, adata: AnnData, *, name: str = "anndata#") -> None:
        if name.endswith("#"):
            name += str(id(self))
        super().__init__(name=name)

        #: The wrapped ``AnnData`` object.
        self.adata = adata

    def _item_names(self) -> Collection[str]:
        return [name for name in self.adata.uns if ";" not in name]

    def _has_item(self, name: str) -> bool:
        return name in self.adata.uns

    def _get_item(self, name: str) -> Any:
        return self.adata.uns[name]

    def _axis_names(self) -> Collection[str]:
        names = set(name[:-1] for name in self.adata.uns if name.endswith(";"))
        names.add("obs")
        names.add("var")
        return names

    def _has_axis(self, axis: str) -> bool:
        return axis in ("obs", "var") or f"{axis};" in self.adata.uns

    def _axis_size(self, axis: str) -> int:
        if axis == "obs":
            return self.adata.n_obs
        if axis == "var":
            return self.adata.n_vars
        return len(self.adata.uns[f"{axis};"])

    def _axis_entries(self, axis: str) -> Known1D:
        if axis == "obs":
            return self.adata.obs_names
        if axis == "var":
            return self.adata.var_names
        return self.adata.uns[f"{axis};"]

    def _data1d_names(self, axis: str) -> Collection[str]:
        if axis == "obs":
            return [f"obs;{property}" for property in self.adata.obs if ";" not in property]
        if axis == "var":
            return [f"var;{property}" for property in self.adata.var if ";" not in property]
        return [
            property for property in self.adata.uns if property.startswith(f"{axis};") and len(property) > len(axis) + 1
        ]

    def _has_data1d(self, axis: str, name: str) -> bool:
        if axis == "obs":
            return _interface.suffix(name, ";") in self.adata.obs
        if axis == "var":
            return _interface.suffix(name, ";") in self.adata.var
        return name in self.adata.uns

    def _get_data1d(self, axis: str, name: str) -> Known1D:
        if axis == "obs":
            return self.adata.obs[_interface.suffix(name, ";")]
        if axis == "var":
            return self.adata.var[_interface.suffix(name, ";")]
        return self.adata.uns[name]

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:  # pylint: disable=too-many-return-statements
        if axes == ("obs", "var"):
            names = set(f"obs,var;{property}" for property in self.adata.layers)
            names.add("obs,var;X")
            return names

        if axes == ("obs", "obs"):
            return set(f"obs,obs;{property}" for property in self.adata.obsp.keys())

        if axes == ("var", "var"):
            return set(f"var,var;{property}" for property in self.adata.varp.keys())

        if axes == ("var", "obs") or axes[1] in ("var", "obs"):
            return set()

        if axes[0] == "obs":
            return set(
                f"obs,{axes[1]};{_interface.suffix(name, ';')}"
                for name in self.adata.obs.keys()
                if ";" in name and name.startswith(f"{axes[1]}=")
            )

        if axes[0] == "var":
            return set(
                f"var,{axes[1]};{_interface.suffix(name, ';')}"
                for name in self.adata.var.keys()
                if ";" in name and name.startswith(f"{axes[1]}=")
            )

        return set(name for name in self.adata.uns if name.startswith(f"{axes[0]},{axes[1]};"))

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:  # pylint: disable=too-many-return-statements
        if axes == ("obs", "var"):
            return name == "obs,var;X" or _interface.suffix(name, ";") in self.adata.layers

        if axes == ("obs", "obs"):
            return _interface.suffix(name, ";") in self.adata.obsp.keys()

        if axes == ("var", "var"):
            return _interface.suffix(name, ";") in self.adata.varp.keys()

        if axes[1] in ("var", "obs"):
            return False

        if axes[0] == "obs":
            first_entry = self.axis_entries(axes[1])[0]
            return f"{axes[1]}={first_entry};{_interface.suffix(name, ';')}" in self.adata.obs

        if axes[0] == "var":
            first_entry = self.axis_entries(axes[1])[0]
            return f"{axes[1]}={first_entry};{_interface.suffix(name, ';')}" in self.adata.var

        return name in self.adata.uns

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Known2D:  # pylint: disable=too-many-return-statements
        if axes == ("obs", "var"):
            if name == "obs,var;X":
                return self.adata.X
            return self.adata.layers[_interface.suffix(name, ";")]

        if axes == ("obs", "obs"):
            return self.adata.obsp[_interface.suffix(name, ";")]

        if axes == ("var", "var"):
            return self.adata.varp[_interface.suffix(name, ";")]

        if axes[0] == "obs":
            return np.array(
                [
                    as_vector(self.adata.obs[f"{axes[1]}={entry};{_interface.suffix(name, ';')}"])
                    for entry in self.axis_entries(axes[1])
                ]
            ).transpose()

        if axes[0] == "var":
            return np.array(
                [
                    as_vector(self.adata.var[f"{axes[1]}={entry};{_interface.suffix(name, ';')}"])
                    for entry in self.axis_entries(axes[1])
                ]
            ).transpose()

        return self.adata.uns[name]


class AnnDataWriter(AnnDataReader, _interface.StorageWriter):
    """
    Implement the `.StorageWriter` interface for ``AnnData``.

    .. note::

        Do **not** modify the wrapped ``AnnData`` after creating a writer (other than through the writer object).
        Modifications may or may not be visible in the writer, causing subtle problems.

        Setting large 2D data for axes other than ``obs`` and ``var`` will be inefficient.
    """

    def _set_item(self, name: str, item: Any) -> None:
        self.adata.uns[name] = item

    def _create_axis(self, axis: str, entries: Vector) -> None:
        self.adata.uns[f"{axis};"] = entries

    def _set_vector(self, axis: str, name: str, vector: Vector) -> None:
        if axis == "obs":
            self.adata.obs[_interface.suffix(name, ";")] = vector
        elif axis == "var":
            self.adata.var[_interface.suffix(name, ";")] = vector
        else:
            self.adata.uns[name] = vector

    def _set_matrix(self, axes: Tuple[str, str], name: str, matrix: MatrixInRows) -> None:
        if axes == ("var", "obs") or (axes[0] not in ("var", "obs") and axes[1] in ("var", "obs")):
            name = f"{axes[1]},{axes[0]};{_interface.suffix(name, ';')}"
            matrix = freeze(optimize(matrix.transpose()))
            axes = (axes[1], axes[0])

        if axes == ("obs", "var"):
            if name == "obs,var;X":
                self.adata.X = matrix
            else:
                self.adata.layers[_interface.suffix(name, ";")] = matrix
            return

        if axes == ("obs", "obs"):
            self.adata.obsp[_interface.suffix(name, ";")] = matrix
            return

        if axes == ("var", "var"):
            self.adata.varp[_interface.suffix(name, ";")] = matrix
            return

        if axes[0] == "obs":
            matrix_in_columns = as_layout(matrix, COLUMN_MAJOR)
            for index, entry in enumerate(self.axis_entries(axes[1])):
                self.adata.obs[f"{axes[1]}={entry};{_interface.suffix(name, ';')}"] = as_vector(
                    matrix_in_columns[:, index]
                )
            return

        if axes[0] == "var":
            matrix_in_columns = as_layout(matrix, COLUMN_MAJOR)
            for index, entry in enumerate(self.axis_entries(axes[1])):
                self.adata.var[f"{axes[1]}={entry};{_interface.suffix(name, ';')}"] = as_vector(
                    matrix_in_columns[:, index]
                )
            return

        self.adata.uns[name] = matrix


def anndata_as_storage(
    adata: AnnData, obs: str, var: str, X: str, name: str = "anndata#"  # pylint: disable=invalid-name
) -> _interface.StorageReader:
    """
    View some ``adata`` (``AnnData`` object) as a `.StorageReader` to allow accessing it using ``daf``.

    This creates a `.StorageView` which maps the hard-wired ``obs`` and ``var`` axes names and the hard-wired ``X`` data
    to hopefully more meaningful names.

    If the ``name`` ends with ``#`` we append the object id to it to make it unique.

    .. note::

        It is often necessary to split the same data set into multiple ``AnnData`` objects, e.g. when the data contains
        both cells and cell-clusters/types, it is best to split it into two ``AnnData`` objects, one for the cells data
        and another for the clusters/types data. To merge these back into a single storage, create a `.StorageChain` by
        writing something like ``StorageChain([anndata_as_storage(..), anndata_as_storage(...), ...])``.
    """
    return _views.StorageView(
        AnnDataReader(adata),
        name=name,
        axes=dict(obs=obs, var=var),
        data={"obs,var;X": X},
    )


def storage_as_anndata(storage: _interface.StorageReader) -> AnnData:
    """
    Convert some ``storage`` into an ``AnnData`` object.

    The storage needs to include an ``obs`` and a ``var`` axis, and an ``obs,var;X`` matrix. In general it should
    contain the data one wishes to store in the ``AnnData`` object. Typically this is achieved by creating a
    `.StorageView` ``hide_implicit=True`` for the full data; this view is also used to rename the meaningful axes and
    data names to ``obs``, ``var`` and ``X``.

    .. note::

        It is often necessary to create several such views to extract several ``AnnData`` objects from the same
        ``storage``, e.g. when the data contains both cells and cell-clusters/types, it is best to split it into two
        ``AnnData`` objects, one for the cells data and another for the clusters/types data. Other data (e.g. per gene
        data) can be replicated in both ``AnnData`` objects or stored in only one of them, as needed.
    """
    data2d = storage.get_data2d("obs,var;X")
    if is_matrix(data2d):
        adata = AnnData(data2d, dtype=dtype_of(data2d))
    else:
        adata = AnnData(data2d)

    adata.obs_names = storage.axis_entries("obs")
    adata.var_names = storage.axis_entries("var")
    writer = AnnDataWriter(adata)
    writer.update(storage, overwrite=True)
    return writer.adata
