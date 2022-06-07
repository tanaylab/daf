"""
This stores the data inside an ``AnnData`` `AnnData <https://pypi.org/project/anndata>`_ object.

**Use cases***

Since ``AnnData`` is not really powerful enough to satisfy our needs (this was the main motivation for creationg
``daf``), this is mainly used to interface with other systems.

That is, if you have ``AnnData`` created by another system (such as `scanpy <https://pypi.org/project/scanpy/>`_), it is
technically possible to directly open it in ``daf`` using `.AnnDataReader` or even `.AnnDataWriter`. However, this would
be inconvenient, as ``AnnData`` mandates the two "main" axes to be called ``obs`` and ``var`` and the "main" 2D data to
be called ``X``; and inefficient, as ``AnnData`` isn't friendly to 2D data other than per-obs-per-var.

Instead, the recommended way is to wrap the `.AnnDataReader` with a `.StorageView` to properly rename the ``obs`` and
``var`` axes and the ``obs,var:X`` 2D data. If you have several ``AnnData`` objects with shared axes (e.g., one
per-gene-per-cell and another per-gene-per-cell-cluster/type), you can use `.StorageChain` to merge them into a single
coherent ``daf`` storage.

This suffices for read-only access (e.g. visualizations). You can also copy the results into a `.FilesWriter` (using
`.StorageWriter.update`) if you wish to convert from ``AnnData`` to native ``daf`` storage. Or, for running a
computation pipeline on the data, wrap it with a ``TODOL-DafDelta`` using `.MemoryStorage` or `.FilesWriter` to store
the additional computed data.

To export ``daf`` results as ``AnnData``, create a `.StorageView` which renames the two main axes to ``obs`` and ``var``
and the main 2D data to ``X``, and exposes only the data it makes sense to include in the ``AnnData`` object (typically
using ``hide_implicit=True``). Then use this view to create a new `.AnnDataWriter`, which will give you an ``AnnData``
object containing just the exposed data. You can directly use this object in ``scanpy``, or write it to a file (e.g. in
``h5ad`` format). You may need to create a set of such ``AnnData`` objects and/or files for different combinations of
"main" axes (e.g., if you have both per-gene-per-cell data and per-gene-per-cell-cluster/type data). These could be
re-merged when imported back, as described above.

As this process is tedious and error-prone, two utility functions `.anndata_as_daf` and `.daf_as_anndata` are provided,
which automate most of these steps.

**Representation**

Scalar (0D "blob") data is easy, it is stored in the ``uns`` field of ``AnnData``.

Axes other than ``obs`` and ``var`` require us to store their entries, which we do by using an ``uns`` entry with the
name ``axis;__entries__``.

1D data other than per-``obs`` and per-``var`` data is stored in an ``uns`` entry named ``axis;name``.

2D data for ``obs`` and ``var`` axes is stored in the ``X``, ``layers``, ``obsp`` or ``varp`` ``AnnData`` fields, as
appropriate.

2D data for either ``obs`` or ``var`` and another axis is stored as a set of 1D annotations in the ``obs`` or ``var``
``AnnData`` fields, one for each axis entry, named ``axis=entry;name``. This makes it slightly less unfriendly to access
the data if/when exported to non-``daf`` systems.

2D data where neither axis is ``obs`` or ``var`` is stored in an ``uns`` entry named ``row_axis,column_axis;name``.
"""


from typing import Any
from typing import Collection
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

from ..typing import COLUMN_MAJOR
from ..typing import Array1D
from ..typing import Data2D
from ..typing import GridInRows
from ..typing import as_array1d
from ..typing import as_array2d
from ..typing import as_layout
from ..typing import freeze
from . import chains as _chains
from . import interface as _interface
from . import views as _views

__all__ = ["AnnDataView", "daf_as_anndata", "anndata_as_daf", "AnnDataReader", "AnnDataWriter"]


class AnnDataReader(_interface.StorageReader):
    """
    Implement the `.StorageReader` interface for ``AnnData``.

    .. note::

        Do **not** modify the wrapped ``AnnData`` after creating a reader. Modifications may or may not be visible in
        the reader, causing subtle problems.
    """

    def __init__(self, adata: AnnData, *, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        #: The wrapped ``AnnData`` object.
        self.adata = adata

    def _datum_names(self) -> Collection[str]:
        return [name for name in self.adata.uns if ";" not in name]

    def _has_datum(self, name: str) -> bool:
        return name in self.adata.uns

    def _get_datum(self, name: str) -> Any:
        return self.adata.uns[name]

    def _axis_names(self) -> Collection[str]:
        names = set(name.split(";")[0] for name in self.adata.uns if name.endswith(";__entries__"))
        names.add("obs")
        names.add("var")
        return names

    def _has_axis(self, axis: str) -> bool:
        return axis in ("obs", "var") or f"{axis};__entries__" in self.adata.uns

    def _axis_size(self, axis: str) -> int:
        if axis == "obs":
            return self.adata.n_obs
        if axis == "var":
            return self.adata.n_vars
        return len(self.adata.uns[f"{axis};__entries__"])

    def _axis_entries(self, axis: str) -> Array1D:
        if axis == "obs":
            return freeze(as_array1d(self.adata.obs_names))
        if axis == "var":
            return freeze(as_array1d(self.adata.var_names))
        return freeze(as_array1d(self.adata.uns[f"{axis};__entries__"]))

    def _array1d_names(self, axis: str) -> Collection[str]:
        if axis == "obs":
            return [f"obs;{name}" for name in self.adata.obs if ";" not in name]
        if axis == "var":
            return [f"var;{name}" for name in self.adata.var if ";" not in name]
        return [name for name in self.adata.uns if name.startswith(f"{axis};") and not name.endswith(";__entries__")]

    def _has_array1d(self, axis: str, name: str) -> bool:
        if axis == "obs":
            return name.split(";")[1] in self.adata.obs
        if axis == "var":
            return name.split(";")[1] in self.adata.var
        return name in self.adata.uns

    def _get_array1d(self, axis: str, name: str) -> Array1D:
        if axis == "obs":
            return freeze(as_array1d(self.adata.obs[name.split(";")[1]]))
        if axis == "var":
            return freeze(as_array1d(self.adata.var[name.split(";")[1]]))
        return freeze(as_array1d(self.adata.uns[name]))

    def _data2d_names(self, axes: Tuple[str, str]) -> Collection[str]:  # pylint: disable=too-many-return-statements
        if axes == ("obs", "var"):
            names = set(f"obs,var;{name}" for name in self.adata.layers)
            names.add("obs,var;X")
            return names

        if axes == ("obs", "obs"):
            return set(f"obs,obs;{name}" for name in self.adata.obsp.keys())

        if axes == ("var", "var"):
            return set(f"var,var;{name}" for name in self.adata.varp.keys())

        if axes == ("var", "obs") or axes[1] in ("var", "obs"):
            return set()

        if axes[0] == "obs":
            return set(
                f"obs,{axes[1]};{name.split(';')[1]}"
                for name in self.adata.obs.keys()
                if ";" in name and name.startswith(f"{axes[1]}=")
            )

        if axes[0] == "var":
            return set(
                f"var,{axes[1]};{name.split(';')[1]}"
                for name in self.adata.var.keys()
                if ";" in name and name.startswith(f"{axes[1]}=")
            )

        return set(name for name in self.adata.uns if name.startswith(f"{axes[0]},{axes[1]};"))

    def _has_data2d(self, axes: Tuple[str, str], name: str) -> bool:  # pylint: disable=too-many-return-statements
        if axes == ("obs", "var"):
            return name == "obs,var;X" or name.split(";")[1] in self.adata.layers

        if axes == ("obs", "obs"):
            return name.split(";")[1] in self.adata.obsp.keys()

        if axes == ("var", "var"):
            return name.split(";")[1] in self.adata.varp.keys()

        if axes[1] in ("var", "obs"):
            return False

        if axes[0] == "obs":
            entry = self.axis_entries(axes[1])[0]
            return f"{axes[1]}={entry};{name.split(';')[1]}" in self.adata.obs

        if axes[0] == "var":
            entry = self.axis_entries(axes[1])[0]
            return f"{axes[1]}={entry};{name.split(';')[1]}" in self.adata.var

        return name in self.adata.uns

    def _get_data2d(self, axes: Tuple[str, str], name: str) -> Data2D:  # pylint: disable=too-many-return-statements
        if axes == ("obs", "var"):
            if name == "obs,var;X":
                return self.adata.X
            return self.adata.layers[name.split(";")[1]]

        if axes == ("obs", "obs"):
            return self.adata.obsp[name.split(";")[1]]

        if axes == ("var", "var"):
            return self.adata.varp[name.split(";")[1]]

        if axes[0] == "obs":
            return np.array(
                [
                    as_array1d(self.adata.obs[f"{axes[1]}={entry};{name.split(';')[1]}"])
                    for entry in self.axis_entries(axes[1])
                ]
            ).transpose()

        if axes[0] == "var":
            return np.array(
                [
                    as_array1d(self.adata.var[f"{axes[1]}={entry};{name.split(';')[1]}"])
                    for entry in self.axis_entries(axes[1])
                ]
            ).transpose()

        return as_array2d(self.adata.uns[name])


class AnnDataWriter(AnnDataReader, _interface.StorageWriter):
    """
    Implement the `.StorageWriter` interface for ``AnnData``.

    .. note::

        Do **not** modify the wrapped ``AnnData`` after creating a writer (other than through the writer object).
        Modifications may or may not be visible in the writer, causing subtle problems.

        Setting large 2D data for axes other than ``obs`` and ``var`` will be inefficient.
    """

    def _set_datum(self, name: str, datum: Any) -> None:
        self.adata.uns[name] = datum

    def _create_axis(self, axis: str, entries: Array1D) -> None:
        self.adata.uns[f"{axis};__entries__"] = entries

    def _set_array1d(self, axis: str, name: str, array1d: Array1D) -> None:
        if axis == "obs":
            self.adata.obs[name.split(";")[1]] = array1d
        elif axis == "var":
            self.adata.var[name.split(";")[1]] = array1d
        else:
            self.adata.uns[name] = array1d

    def _set_grid(self, axes: Tuple[str, str], name: str, grid: GridInRows) -> None:
        if axes == ("var", "obs") or (axes[0] not in ("var", "obs") and axes[1] in ("var", "obs")):
            name = f"{axes[1]},{axes[0]};{name.split(';')[1]}"
            grid = grid.transpose()
            axes = (axes[1], axes[0])

        if axes == ("obs", "var"):
            if name == "obs,var;X":
                self.adata.X = grid
            else:
                self.adata.layers[name.split(";")[1]] = grid
            return

        if axes == ("obs", "obs"):
            self.adata.obsp[name.split(";")[1]] = grid
            return

        if axes == ("var", "var"):
            self.adata.varp[name.split(";")[1]] = grid
            return

        if axes[0] == "obs":
            grid_in_columns = as_layout(grid, COLUMN_MAJOR)
            for index, entry in enumerate(self.axis_entries(axes[1])):
                self.adata.obs[f"{axes[1]}={entry};{name.split(';')[1]}"] = as_array1d(grid_in_columns[:, index])
            return

        if axes[0] == "var":
            grid_in_columns = as_layout(grid, COLUMN_MAJOR)
            for index, entry in enumerate(self.axis_entries(axes[1])):
                self.adata.var[f"{axes[1]}={entry};{name.split(';')[1]}"] = as_array1d(grid_in_columns[:, index])
            return

        self.adata.uns[name] = grid


class AnnDataView:  # pylint: disable=too-few-public-methods
    """
    Describe the real names for some ``AnnData``.
    """

    def __init__(self, adata: AnnData, *, obs: str, var: str, X: str, name: Optional[str] = None) -> None:
        #: The described ``AnnData``.
        self.adata = adata

        #: The true name of the observations axis (e.g. "cell").
        self.obs = obs

        #: The true name of the variables axis (e.g. "gene").
        self.var = var

        #: The true name of the main (``X``) 2D data (e.g. "UMIs").
        self.X = X  # pylint: disable=invalid-name

        #: An optional name to assign to the ``daf`` view of the ``AnnData``.
        self.name = name


def anndata_as_daf(
    adata: Union[AnnDataView, Sequence[AnnDataView]], *, name: Optional[str] = None
) -> _interface.StorageReader:
    """
    View some ``adata`` (one or more described ``AnnData`` objects) as a `.StorageReader` to allow accessing it using
    ``daf``.

    If more than one ``adata`` object is given, then they are placed in a `.StorageChain` (first one wins).
    """
    if isinstance(adata, AnnDataView):
        adata = [adata]

    assert len(adata) > 0, "no AnnData to wrap"

    if len(adata) == 1:
        aview = adata[0]
        return _views.StorageView(
            AnnDataReader(aview.adata),
            name=name or aview.name,
            axes=dict(obs=aview.obs, var=aview.var),
            data={"obs,var;X": aview.X},
        )

    return _chains.StorageChain(
        [
            _views.StorageView(
                AnnDataReader(aview.adata),
                name=aview.name,
                axes=dict(obs=aview.obs, var=aview.var),
                data={"obs,var;X": aview.X},
            )
            for aview in adata
        ],
        name=name,
    )


def daf_as_anndata(storage: _interface.StorageReader) -> AnnData:
    """
    View some ``daf`` ``storage`` as an ``AnnData`` object.

    The ``storage`` need to include an ``obs`` and a ``var`` axis, and an ``obs,var;X`` grid. In general it should
    contain the data one wishes to store in the ``AnnData`` object. Typically this is achieved by creating a
    `.StorageView` ``hide_implicit=True`` for the full data.

    .. note::

        It is often necessary to create several such views to extract several ``AnnData`` objects from the same
        ``storage``, e.g. when the data contains both cells and cell-clusters/types, it is best to split it into two
        ``AnnData`` objects, one for the cells data and another for the clusters/types data. Other data (e.g. per gene
        data) can be replicated in both ``AnnData`` objects or stored in only one of them, as needed.
    """
    adata = AnnData(storage.get_data2d("obs,var;X"))
    adata.obs_names = storage.axis_entries("obs")
    adata.var_names = storage.axis_entries("var")
    writer = AnnDataWriter(adata)
    writer.update(storage, overwrite=True)
    return writer.adata
