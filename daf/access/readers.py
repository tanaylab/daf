"""
Read-only interface for ``daf`` data sets.

In ``daf``, data access uses a string name in the format described below. Even though each name uniquely identifies
whether the data is 0D, 1D or 2D, there are separate functions for accessing the data based on its dimension. This
both makes the code more readable, and also allows ``mypy`` to provide some semblence of effective type checking (if
you choose to use it).

.. note::

    To avoid ambiguities and to ensure that storing ``daf`` data in files works as expected, do **not** use ``,``,
    ``#``, ``=`` or ``|`` characters in axis, property or entry names. In addition, since axis and property names are
    used as part of file names in certain storage formats, also avoid characters that are invalid in file names, most
    importantly ``/``, but also ``"``, ``:``, and ``\\``. If you want to be friendly to interactive shell usage, try to
    avoid characters used by shell such as ``'``, ``"``, ``*``, ``?``, ``&``, ``$`` and ``;``, even though these can be
    quoted.

The following data is used in all the examples below:

.. doctest::

    >>> import daf
    >>> data = daf.DafReader(daf.FilesReader(daf.DAF_EXAMPLE_PATH), name="example")

.. _2d_names:

**2D Names**

All 2D names start with ``rows_axis,columns_axis#``.

* | *rows_axis* ``,`` *columns_axis* ``#`` *property*
    [ ``|`` ``!``? `.ElementWise` [ ``,`` *param* ``=`` *value* ]* ]*

  The name of a property with a value per each combination of two axes entries, optionally processed by a series of
  `.ElementWise` `.operations`. For example:

.. doctest::

    >>> data.get_matrix("metacell,gene#UMIs")
    array([[  5.,   6.,  66.,   1.,   1.,   1.,   0., 110.,  13.,   1.],
           [ 13.,   2.,   1.,   2.,   1.,   3.,   2.,   3.,   7.,   1.],
           [211.,   1.,   2.,   0.,  91.,   0.,   0.,   1.,   2.,   4.],
           [  1.,   0., 179.,   1.,   0.,   2.,   0.,   9.,   1.,   2.],
           [  3.,   0.,   2.,  18.,   1.,   1.,   1.,   1., 126.,   1.],
           [ 14.,   0.,   1.,   1.,   2.,  10.,   3.,   3.,   6.,   6.],
           [  3.,   2.,   0.,   0.,   1.,   2.,   0.,   2.,   2.,   3.],
           [  0.,   1.,   0.,   0.,   0.,   1.,   1.,   0.,   5.,   1.],
           [ 62.,   0.,   0.,   0.,   2.,   0.,   2.,   0.,   1.,   0.],
           [326.,   0.,   0.,   0., 151.,   0.,   0.,   1.,   0.,   2.]],
          dtype=float32)

    >>> data.get_matrix("metacell,gene#UMIs|Fraction|Log,base=2,factor=1e-1|Abs")
    array([[3.0056686 , 2.9499593 , 1.239466  , 3.2528863 , 3.2528863 ,
            3.2528863 , 3.321928  , 0.64562523, 2.610649  , 3.2528863 ],
           [1.0848889 , 2.6698513 , 2.959358  , 2.6698513 , 2.959358  ,
            2.4288433 , 2.6698513 , 2.4288433 , 1.7369655 , 2.959358  ],
           [0.36534712, 3.2764134 , 3.2322907 , 3.321928  , 1.3523018 ,
            3.321928  , 3.321928  , 3.2764134 , 3.2322907 , 3.1478987 ],
           [3.2497783 , 3.321928  , 0.02566493, 3.2497783 , 3.321928  ,
            3.1810656 , 3.321928  , 2.7744403 , 3.2497783 , 3.1810656 ],
           [3.0651526 , 3.321928  , 3.1457713 , 2.2050104 , 3.2311625 ,
            3.2311625 , 3.2311625 , 3.2311625 , 0.1231482 , 3.2311625 ],
           [1.3063312 , 3.321928  , 3.038135  , 3.038135  , 2.801096  ,
            1.6556655 , 2.5975626 , 2.5975626 , 2.1175697 , 2.1175697 ],
           [1.7369655 , 2.0995355 , 3.321928  , 3.321928  , 2.5849624 ,
            2.0995355 , 3.321928  , 2.0995355 , 2.0995355 , 1.7369655 ],
           [3.321928  , 2.2439256 , 3.321928  , 3.321928  , 3.321928  ,
            2.2439256 , 2.2439256 , 3.321928  , 0.6092099 , 2.2439256 ],
           [0.03614896, 3.321928  , 3.321928  , 3.321928  , 2.9450738 ,
            3.321928  , 2.9450738 , 3.321928  , 3.1212306 , 3.321928  ],
           [0.35999608, 3.321928  , 3.321928  , 3.321928  , 1.2702659 ,
            3.321928  , 3.321928  , 3.2921808 , 3.321928  , 3.2630343 ]],
          dtype=float32)

.. _1d_names:

**1D Names**

All 1D names start with ``axis#``.

* | *axis* ``#``

  The name of the entries of the axis. That is, ``get_vector("axis#")`` is the same as ``axis_entries("axis")``. For
  example:

.. doctest::

    >>> data.get_vector("cell_type#")
    array(['Amnion', 'Forebrain/Midbrain/Hindbrain', 'Neural tube Posterior',
           'Presomitic mesoderm', 'Surface ectoderm', 'caudal mesoderm',
           'epiblast'], dtype=object)

    >>> data.axis_entries("cell_type")
    array(['Amnion', 'Forebrain/Midbrain/Hindbrain', 'Neural tube Posterior',
           'Presomitic mesoderm', 'Surface ectoderm', 'caudal mesoderm',
           'epiblast'], dtype=object)

* | *axis* ``#`` *property*
    [ ``|`` ``!``? `.ElementWise` [ ``,`` *param* ``=`` *value* ]* ]*

  The name of a property with a value per entry along some axis, optionally processed by a series of `.ElementWise`
  `.operations`. For example:

.. doctest::

    >>> data.get_vector("batch#age")
    array([51, 38, 21, 31, 26, 43, 36, 27, 33, 45, 49, 41, 45])

    >>> data.get_vector("batch#age|Clip,min=30,max=45")
    array([45, 38, 30, 31, 30, 43, 36, 30, 33, 45, 45, 41, 45])

* | *axis* [ ``#`` *axis_property* ]+ ``#`` *property*?
    [ ``|`` ``!``? `.ElementWise` [ ``,`` *param* ``=`` *value* ]* ]*

  The name of properties which are indices or entry names of some axes, followed by the name of a property of the final
  axis, optionally processed by a series of `.ElementWise` `.operations`. For example:

.. doctest::

    >>> data.get_vector("metacell#cell_type#color")
    array(['#f7f79e', '#CDE089', '#1a3f52', '#f7f79e', '#cc7818', '#647A4F',
           '#635547', '#635547', '#A8DBF7', '#1a3f52'], dtype=object)

  A property can refer to an axis either by using its exact name as above or adding some qualifier using ``.``. For
  example, if we had a ``metacell#cell_type.projected`` property containing the cell type obtained by projecting the
  data on an atlas, we could write ``metacell#cell_type.projected#color`` to access the color of the projected cell type
  of each metacell, using the ``cell_type#color`` property.

* | *axis* ``#`` *second_axis* ``=`` *entry* ``,`` *property*
    [ ``|`` ``!``? `.ElementWise` [ ``,`` *param* ``=`` *value* ]* ]*

  The slice for a specific entry of the data of a 2D property, optionally processed by a series of `.ElementWise`
  `.operations`. For example:

.. doctest::

    >>> data.get_vector("metacell#gene=FOXA1,UMIs")
    array([6., 2., 1., 0., 0., 0., 2., 1., 0., 0.], dtype=float32)

    >>> data.get_vector("metacell#gene=FOXA1,UMIs|Clip,min=1,max=4")
    array([4., 2., 1., 1., 1., 1., 2., 1., 1., 1.], dtype=float32)

* | *axis* ``#`` *second_axis* ``,`` *property*
    [ ``|`` ``!``? `.ElementWise` [ ``,`` *param* ``=`` *value* ]* ]*
  | ``|`` ``!``? `.Reduction` [ ``,`` *param* ``=`` *value* ]*
    [ ``|`` ``!``? `.ElementWise` [ ``,`` *param* ``=`` *value* ]* ]*

  A reduction of 2D data into a single value per row, optionally processed by a series of `.ElementWise` `.operations`.
  For example:

.. doctest::

    >>> data.get_vector("metacell#gene,UMIs|Sum")
    array([204.,  35., 312., 195., 154.,  46.,  15.,   9.,  67., 480.],
          dtype=float32)

    >>> data.get_vector("metacell#gene,UMIs|Fraction|Log,base=2,factor=1e-5|Max|Clip,min=-1.5,max=-0.5")
    array([-0.89103884, -1.4288044 , -0.5642817 , -0.5       , -0.5       ,
           -1.5       , -1.5       , -0.84797084, -0.5       , -0.5581411 ],
          dtype=float32)

.. _0d_names:

**0D Names**

  No 0D names contain ``#`` (at least not before the first ``|``).

* | *property*

  The name of a 0D data item property. For example:

.. doctest::

    >>> data.get_item("created")
    datetime.datetime(2022, 7, 6, 16, 49, 44)

* | *axis* ``=`` *entry* ``,`` *property*

  The value for a specific entry of the data of a 1D property. For example:

.. doctest::

    >>> data.get_item("batch=Batch_1,age")
    38

* | *axis* ``=`` *entry* ``,`` *second_axis* ``=`` *second_entry* ``,`` *property*

  The value for a specific entry of the data of a 2D property. For example:

.. doctest::

    >>> data.get_item("metacell=Metacell_1,gene=FOXA1,UMIs")
    2.0

* | *axis* ``,`` *property*
  | [ ``|`` ``!``? `.ElementWise` [ ``,`` *param* ``=`` *value* ]* ]*
    ``|`` ``!``? `.Reduction` [ ``,`` *param* ``=`` *value* ]*

  A reduction into a single value of 1D property with a value per entry along some axis, optionally processed by a
  series of `.ElementWise` `.operations`. For example:

.. doctest::

    >>> data.get_item("batch,age|Mean")
    37.38461538461539

    >>> data.get_item("batch,age|Clip,min=30,max=40|Mean")
    36.0

* | *axis* ``,`` *second_axis* ``=`` *entry* ``,`` *property*
  | [ ``|`` ``!``? `.ElementWise` [ ``,`` *param* ``=`` *value* ]* ]*
    ``|`` ``!``? `.Reduction` [ ``,`` *param* ``=`` *value* ]*

  A reduction into a single value of a slice for a specific entry of the data of a 2D property, optionally processed by
  a series of `.ElementWise` `.operations`. For example:

.. doctest::

    >>> data.get_item("metacell,gene=FOXA1,UMIs|Max")
    6.0

    >>> data.get_item("metacell,gene=FOXA1,UMIs|Clip,min=1,max=3|Mean")
    1.4

* | *axis* ``,`` *second_axis* ``,`` *property*
  | [ ``|`` ``!``? `.ElementWise` [ ``,`` *param* ``=`` *value* ]* ]*
    ``|`` ``!``? `.Reduction` [ ``,`` *param* ``=`` *value* ]*
  | [ ``|`` ``!``? `.ElementWise` [ ``,`` *param* ``=`` *value* ]* ]*
    ``|`` ``!``? `.Reduction` [ ``,`` *param* ``=`` *value* ]*

  A reduction of 2D data into a single value per row and then to a single value, optionally processed by a series of
   `.ElementWise` `.operations`. For example:

.. doctest::

    >>> data.get_item("metacell,gene,UMIs|Sum|Max")
    480.0

    >>> data.get_item("metacell,gene,UMIs|Fraction|Log,base=2,factor=1e-5|Max|Clip,min=-1.5,max=-0.5|Mean")
    -0.8790237

.. note::

    See `.operations` for the list of built-in `.ElementWise` and `.Reduction` operations. Additional operations can be
    offered by other Python packages. In all the above, prefixing the operation name with ``!`` will prevent their
    results from being cached. For example, ``cell#gene,UMIs|!Sum`` will not cache the total number of UMIs per cell.
    The current implementation doesn't cache any 0D data regardless of whether a ``!`` was specified.

**Motivation**

The above scheme makes sense if you consider that each name starts with a description of the axes/shape of the result,
followed by how to extract the result from the data set. This means that to get the sum of the UMIs of all the genes for
each cell, we first consider this is per-cell 1D data and therefore must start with ``cell#``. We therefore write
``cell#gene,UMIs|Sum`` instead of ``cell,gene#UMIs|Sum``.

This may seem unintuitive at first, but it has some advantages, such as clearly identify the axes/shape of the result of
a pipeline. An important feature of the scheme is that the name of **any** 1D data along some ``axis`` has the common
prefix ``axis#``. This makes it easy to express data for `.get_columns`, or describe the X and Y coordinates of a
scatter plot, or anything along these lines, by providing the common axis and a list suffixes to append to it.
"""


# pylint: disable=too-many-lines
# pylint: disable=duplicate-code

from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd  # type: ignore
import yaml  # type: ignore

from ..storage import AxisView
from ..storage import MemoryStorage
from ..storage import StorageChain
from ..storage import StorageReader
from ..storage import StorageView
from ..storage import StorageWriter
from ..storage import prefix
from ..storage import suffix
from ..typing import ROW_MAJOR
from ..typing import STR_DTYPE
from ..typing import AnyData
from ..typing import FrameInColumns
from ..typing import FrameInRows
from ..typing import Matrix
from ..typing import MatrixInRows
from ..typing import Series
from ..typing import Vector
from ..typing import as_layout
from ..typing import as_matrix
from ..typing import as_vector
from ..typing import be_dense_in_rows
from ..typing import be_matrix
from ..typing import be_matrix_in_rows
from ..typing import be_sparse_in_rows
from ..typing import be_vector
from ..typing import freeze
from ..typing import has_dtype
from ..typing import is_dense_in_rows
from ..typing import is_matrix_in_rows
from ..typing import is_sparse_in_rows
from ..typing import is_vector
from ..typing import optimize
from . import operations as _operations

# pylint: enable=duplicate-code

__all__ = [
    "DafReader",
    "transpose_name",
]


# pylint: disable=protected-access


class BaseName:  # pylint: disable=too-few-public-methods
    """
    Describe the name of base data of a pipeline.
    """

    def __init__(self, *, base_name: str, full_name: str, dataset_name: str) -> None:
        #: The name of the base data (start of pipeline).
        self.name = base_name

        #: The properties to get the data of.
        #:
        #: If there is more than one, this means looking up properties in other axes.
        #: An empty name means fetching the axis entries.
        self.properties: List[str]

        #: The axes to access for the base data (start of pipeline).
        self.axes: List[str] = []

        #: The entries to pick out of the base axes to get the base data (start of pipeline).
        self.entries: List[Optional[str]] = []

        #: The number of dimensions of the base data (start of pipeline).
        self.ndim: int = 0

        #: The number of dimensions of the final result (end of pipeline).
        self.final_ndim: int = 0

        #: The canonical format of the name.
        self.canonical: str

        parts = base_name.split("#", 1)

        if len(parts) == 2:
            for axis in parts[0].split(","):
                self.axes.append(axis)
                self.entries.append(None)
            self.ndim = self.final_ndim = len(self.axes)
            parts = parts[1:]

        parts = parts[0].split(",")

        self.properties = parts[-1].split("#")
        assert len(self.properties) == 1 or self.ndim == 1, (
            f"specifying an axis properties chain for {self.ndim}D data "
            f"in the data name: {full_name} "
            f"for the data set: {dataset_name}"
        )
        assert (
            "##" not in parts[-1]
        ), f"invalid axis properties chain in the data name: {full_name} for the data set: {dataset_name}"

        for part in parts[:-1]:
            axis_parts = part.split("=", 1)
            axis = axis_parts[0]
            self.axes.append(axis)
            if len(axis_parts) == 1:
                assert len(self.entries) == 0 or self.entries[-1] is None, (
                    f"specifying entry for the first instead of the second axis in the base data name: {base_name} "
                    f"in the data name: {full_name} "
                    f"for the data set: {dataset_name}"
                )
                self.entries.append(None)
                self.ndim += 1
            else:
                self.entries.append(axis_parts[1])

        assert len(self.axes) <= 2, (
            f"{self.ndim}D base data name: {base_name} "
            f"in the data name: {full_name} "
            f"for the data set: {dataset_name}"
        )

        self._finalize()

    def _finalize(self) -> None:
        if len(self.entries) == 2 and self.entries[0] is None and self.entries[1] is not None:
            self.axes.reverse()
            self.entries.reverse()

        self.canonical = "#".join(self.properties)
        if self.ndim == 0:
            if len(self.axes) == 1:
                assert self.entries[0] is not None
                self.canonical = f"{self.axes[0]}={self.entries[0]},{self.canonical}"
            elif len(self.axes) == 2:
                assert self.entries[0] is not None
                assert self.entries[1] is not None
                sorted_axes = self.axes
                sorted_entries = self.entries
                if sorted_axes[0] > sorted_axes[1]:
                    sorted_axes.reverse()
                    sorted_entries.reverse()
                self.canonical = (
                    f"{sorted_axes[0]}={sorted_entries[0]},{sorted_axes[1]}={sorted_entries[1]},{self.canonical}"
                )

        elif self.ndim == 1:
            if len(self.axes) == 1:
                assert self.entries[0] is None
                self.canonical = f"{self.axes[0]}#{self.canonical}"
            else:
                assert len(self.axes) == 2
                assert self.entries[0] is not None
                assert self.entries[1] is None
                self.canonical = f"{self.axes[0]}#{self.axes[1]}={self.entries[1]},{self.canonical}"

        else:
            assert len(self.axes) == 2
            assert self.entries[0] is None
            assert self.entries[1] is None
            self.canonical = f"{self.axes[0]},{self.axes[1]}#{self.canonical}"


class OperationName:  # pylint: disable=too-few-public-methods
    """
    Describe the name of an operation in a pipeline.
    """

    def __init__(self, *, operation_name: str, input_ndim: int, full_name: str, dataset_name: str) -> None:
        #: The name of the operation in the pipeline.
        self.name = operation_name

        #: Whether to cache the results.
        self.cache = not operation_name.startswith("!")
        if not self.cache:
            operation_name = operation_name[1:]

        assert input_ndim > 0, (
            f"0D input for the operation: {self.name}"
            f"in the data name: {full_name} "
            f"for the data set: {dataset_name}"
        )

        parts = operation_name.split(",")

        #: The name of the operation.
        self.operation = parts[0]

        #: The parameters of the operation.
        self.kwargs: Dict[str, str] = {}

        for part in parts[1:]:
            key, value = part.split("=", 1)
            assert key not in self.kwargs, (
                f"repeated parameter: {key} "
                f"for the operation: {self.name} "
                f"in the data name: {full_name} "
                f"for the data set: {dataset_name}"
            )
            self.kwargs[key] = value

        klass = _operations.Operation._registry.get(self.operation)
        assert (
            klass is not None
        ), f"unknown operation: {self.name} in the data name: {full_name} for the data set: {dataset_name}"

        #: The class implementing the operation.
        self.klass = klass

        #: Whether this operation is a `.Reduction`.
        self.is_reduction = _operations.Reduction in klass.mro()

        if self.is_reduction:
            #: The number of dimensions of the result.
            self.ndim = input_ndim - 1
        else:
            self.ndim = input_ndim


class ParsedName:  # pylint: disable=too-few-public-methods
    """
    Parse a full data name.
    """

    def __init__(self, *, full_name: str, dataset_name: str) -> None:
        parts = full_name.split("|")

        #: The name of the base data.
        self.base = BaseName(base_name=parts[0], full_name=full_name, dataset_name=dataset_name)

        #: The number of dimensions of the result.
        self.ndim = self.base.ndim

        #: The names of the pipeline operations.
        self.operations: List[OperationName] = []

        for part in parts[1:]:
            operation = OperationName(
                operation_name=part, input_ndim=self.ndim, full_name=full_name, dataset_name=dataset_name
            )
            self.operations.append(operation)
            self.ndim = operation.ndim

        assert self.ndim == self.base.final_ndim, (
            f"expected {self.base.final_ndim}D result "
            f"but got a {self.ndim}D result "
            f"from the data name: {full_name} "
            f"for the data set: {dataset_name} "
        )


class PipelineState:  # pylint: disable=too-few-public-methods
    """
    State while evaluating an operations pipeline.
    """

    def __init__(self, base_name: BaseName, base_data: Any) -> None:
        assert base_name.ndim > 0
        assert base_name.ndim == base_data.ndim

        #: The data we have computed so far.
        self.data = base_data

        #: The number of dimensions in the data.
        self.ndim = base_data.ndim

        if base_name.ndim == 2:
            #: The original 2D axes. For 1D data, only the 1st axis is meaningful.
            self.axes = (base_name.axes[0], base_name.axes[1])
        else:
            self.axes = (base_name.axes[0], base_name.axes[0])

        #: The pipeline so far (for error messages).
        self.pipeline = base_name.name

        #: The canonical name of the data for caching it in the ``derived`` storage.
        self.canonical = base_name.canonical

        #: Whether to allow caching at all.
        self.allow_cache = base_name.ndim == len(base_name.axes)


class DafReader:  # pylint: disable=too-many-public-methods
    """
    Read-only access to a ``daf`` data set.

    .. note:::

        It is safe to add new data to the ``base`` after wrapping it with a ``DafReader``, but overwriting existing data
        is **not** safe, since any cached ``derived`` data will **not** be updated, causing subtle problems.

    The following data is used in all the examples below:

    .. doctest::

        >>> import daf
        >>> import yaml
        >>> data = daf.DafReader(daf.FilesReader(daf.DAF_EXAMPLE_PATH), name="example")
    """

    def __init__(self, base: StorageReader, *, derived: Optional[StorageWriter] = None, name: str = ".daf#") -> None:
        """
        If the ``name`` starts with ``.``, it is appended to the ``base`` name. If the name ends with ``#``, we append
        the object id to it to make it unique.
        """
        if name.startswith("."):
            name = base.name + name
        if name.endswith("#"):
            name += str(id(self))

        #: The name of the data set for messages.
        self.name = name

        #: The storage the ``daf`` data set is based on.
        self.base = base.as_reader()

        #: How to store derived data computed from the storage data, for example, an alternate layout of 2D data, of the
        #: result of a pipeline (e.g. ``cell,gene#UMIs|Sum``). By default this is stored in a `.MemoryStorage` so
        #: expensive operations (such as `.as_layout`) will only be computed once in the application's lifetime. You can
        #: explicitly set this to `.NO_STORAGE` to disable the caching, or specify some persistent storage such as
        #: `.FilesWriter` to allow the caching to be reused across multiple application invocations. You can even set
        #: this to be the same as the base storage to have everything (base and derived data) be stored in the same
        #: place.
        self.derived = derived or MemoryStorage(name=self.name + ".derived")

        #: A `.StorageChain` to use to actually access the data. This looks first in ``derived`` and then in the
        #: ``base``.
        self.chain = StorageChain([self.derived, self.base], name=self.name + ".chain")

        # Cache mapping from axis entries to indices.
        self._axis_indices: Dict[str, Dict[str, int]] = {}

        for axis in self.base.axis_names():
            if not self.derived.has_axis(axis):
                self.derived.create_axis(axis, freeze(optimize(as_vector(self.chain.axis_entries(axis)))))

    def __str__(self) -> str:
        return f"<{self.__class__.__module__}.{self.__class__.__qualname__} at {id(self)} called {self.name}>"

    def __repr__(self) -> str:
        return yaml.dump(self.description(detail=True), width=99999, sort_keys=False).strip()

    def as_reader(self) -> "DafReader":
        """
        Return the data set as a `.DafReader`.

        This is a no-op (returns self) for "real" read-only data sets, but for writable data sets, it returns a "real"
        read-only wrapper object (that does not implement the writing methods). This ensures that the result can't be
        used to modify the data if passed by mistake to a function that takes a `.DafWriter`.
        """
        return self

    # pylint: disable=duplicate-code

    def description(  # pylint: disable=too-many-branches
        self, *, detail: bool = False, deep: bool = False, description: Optional[Dict] = None
    ) -> Dict:
        """
        Return a dictionary describing the  ``daf`` data set, useful for debugging.

        The result uses the ``name`` field as a key, with a nested dictionary value with the keys ``class``, ``axes``,
        and ``data``.

        If not ``detail``, the ``axes`` will contain a dictionary mapping each axis to a description of its size, and
        the ``data`` will contain just a list of the data names, data, except for `.StorageView` where it will be a
        dictionary mapping each exposed name to the base name.

        If ``detail``, both the ``axes`` and the ``data`` will contain a mapping providing additional
        `.data_description` of the relevant data.

        If ``deep``, there may be additional keys describing the internal storage.

        If ``description`` is provided, collect the result into it. This allows collecting multiple data set
        descriptions into a single overall system state description.

        For example:

        .. doctest::

            >>> print(yaml.dump(data.description()).strip())
            example:
              axes:
                batch: 13 entries
                cell: 524 entries
                cell_type: 7 entries
                gene: 10 entries
                metacell: 10 entries
                sex: 2 entries
              class: daf.access.readers.DafReader
              data:
              - created
              - batch#age
              - batch#sex
              - cell#batch
              - cell#metacell
              - cell_type#color
              - gene#feature_gene
              - gene#forbidden_gene
              - metacell#cell_type
              - metacell#umap_x
              - metacell#umap_y
              - cell,gene#UMIs
              - metacell,gene#UMIs
        """
        description = description or {}
        if self.name in description:
            return description

        self_description: Dict
        description[self.name] = self_description = {}

        self_description["class"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        self_description["axes"] = self.chain._axes_description(detail=detail)
        self_description["data"] = self.chain._data_description(detail=detail)

        if deep:
            self_description["chain"] = self.chain.name
            self_description["derived"] = self.derived.name
            if hasattr(self, "storage"):
                self_description["storage"] = getattr(self, "storage").name
            self_description["base"] = self.base.name
            self.chain.description(deep=True, detail=detail, description=description)

        return description

    # pylint: enable=duplicate-code

    def verify_has(self, names: Union[str, Collection[str]], *, reason: str = "required") -> None:
        """
        Assert that all the listed data ``names`` exist in the data set, regardless if each is a 0D, 1D or 2D data name.

        To verify an axis exists, list it as ``axis#``.

        For example:

        .. doctest::

            >>> data.verify_has("cell#")
            >>> data.verify_has(["metacell,gene#UMIs", "batch#age"])
            >>> data.verify_has(["cell#color"])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
             ...
            AssertionError: missing the data: cell#color which is required in the data set: example


        """
        if isinstance(names, str):
            names = [names]
        for name in names:
            kind = f"axis: {name[:-1]}" if name.endswith("#") else f"data: {name}"
            assert self.has_data(name), f"missing the {kind} which is {reason} in the data set: {self.name}"

    def has_data(self, name: str) -> bool:
        """
        Return whether the data set contains the ``name`` data, regardless of whether it is a 0D, 1D or 2D data.

        To test whether an axis exists, you can use the ``axis#`` name.

        For example:

        .. doctest::

            >>> data.has_data("cell#")
            True

            >>> data.has_data("cell,gene#fraction")
            False
        """
        if name.endswith("#"):
            return self.has_axis(name[:-1])
        if "#" not in name:
            return self.has_item(name)
        axes = prefix(name, "#").split(",")
        assert 1 <= len(axes) <= 2, f"{len(axes)}D name: {name} for: has_data for the data set: {self.name}"
        if len(axes) == 1:
            return self.has_data1d(name)
        return self.has_data2d(name)

    def item_names(self) -> List[str]:
        """
        Return the list of names of the 0D data items that exists in the data set, in alphabetical order.

        For example:

        .. doctest::

            >>> data.item_names()
            ['created']
        """
        return sorted(self.chain.item_names())

    def has_item(self, name: str) -> bool:
        """
        Check whether the ``name`` 0D data item exists in the data set.

        For example:

        .. doctest::

            >>> data.has_item("created")
            True

            >>> data.has_item("modified")
            False
        """
        return self.chain.has_item(name)

    def get_item(self, name: str) -> Any:
        """
        Access a 0D data item from the data set (which must exist) by its ``name``.

        The name is the name of some 0D data as described :ref:`above <0d_names>`.

        For example:

        .. doctest::

            >>> data.get_item("created")
            datetime.datetime(2022, 7, 6, 16, 49, 44)
        """
        parsed_name = ParsedName(full_name=name, dataset_name=self.name)
        assert (
            parsed_name.ndim == 0
        ), f"{parsed_name.ndim}D data name: {name} given to get_item for the data set: {self.name}"
        return self._get_parsed(parsed_name)[1]

    def axis_names(self) -> List[str]:
        """
        Return the list of names of the axes that exist in the data set, in alphabetical order.

        For example:

        .. doctest::

            >>> data.axis_names()
            ['batch', 'cell', 'cell_type', 'gene', 'metacell', 'sex']
        """
        return sorted(self.chain._axis_names())

    def has_axis(self, axis: str) -> bool:
        """
        Check whether the ``axis`` exists in the data set.

        For example:

        .. doctest::

            >>> data.has_axis("cell")
            True

            >>> data.has_axis("height")
            False
        """
        return self.chain._has_axis(axis)

    def axis_size(self, axis: str) -> int:
        """
        Get the number of entries along some ``axis`` (which must exist).

        For example:

        .. doctest::

            >>> data.axis_size("metacell")
            10
        """
        assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"
        return self.chain._axis_size(axis)

    def axis_entries(self, axis: str) -> Vector:
        """
        Get the unique name of each entry in the data set along some ``axis`` (which must exist).

        .. note::

            You can also get the axis entries using ``.get_vector`` by passing it the 1D data name ``axis#``.

        For example:

        .. doctest::

            >>> data.axis_entries("gene")
            array(['RSPO3', 'FOXA1', 'WNT6', 'TNNI1', 'MSGN1', 'LMO2', 'SFRP5',
                   'DLX5', 'ITGA4', 'FOXA2'], dtype=object)

            >>> data.get_vector("gene#")
            array(['RSPO3', 'FOXA1', 'WNT6', 'TNNI1', 'MSGN1', 'LMO2', 'SFRP5',
                   'DLX5', 'ITGA4', 'FOXA2'], dtype=object)
        """
        assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"
        return freeze(optimize(as_vector(self.chain._axis_entries(axis))))

    def axis_indices(self, axis: str) -> Mapping[str, int]:
        """
        Return a mapping from the axis string entries to the integer indices.

        For example:

        .. doctest::

            >>> print(yaml.dump(data.axis_indices("gene")).strip())
            DLX5: 7
            FOXA1: 1
            FOXA2: 9
            ITGA4: 8
            LMO2: 5
            MSGN1: 4
            RSPO3: 0
            SFRP5: 6
            TNNI1: 3
            WNT6: 2
        """
        indices = self._axis_indices.get(axis)
        if indices is None:
            self._axis_indices[axis] = indices = {name: index for index, name in enumerate(self.axis_entries(axis))}
        return indices

    def axis_index(self, axis: str, entry: str) -> int:
        """
        Return the index of the ``entry`` (which must exist) in the entries of the ``axis`` (which must exist).

        For example:

        .. doctest::

            >>> data.axis_index("gene", "FOXA2")
            9
        """
        indices = self.axis_indices(axis)
        index = indices.get(entry)
        assert (
            index is not None
        ), f"missing entry: {entry} in the entries of the axis: {axis} for the data set: {self.name}"
        return index

    def data1d_names(self, axis: str, *, full: bool = True) -> List[str]:
        """
        Return the names of the 1D data that exists in the data set for a specific ``axis`` (which must exist), in
        alphabetical order.

        The returned names are in the format ``axis#name`` which uniquely identifies the 1D data. If not ``full``, the
        returned names include only the simple ``name`` without the ``axis#`` prefix.

        For example:

        .. doctest::

            >>> data.data1d_names("batch")
            ['batch#age', 'batch#sex']

            >>> data.data1d_names("batch", full=False)
            ['age', 'sex']
        """
        assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"
        names = sorted(self.chain._data1d_names(axis))
        if not full:
            names = [suffix(name, "#") for name in names]
        return names

    def has_data1d(self, name: str) -> bool:
        """
        Check whether the ``name`` 1D data exists.

        The name must be in the format ``axis#name`` which uniquely identifies the 1D data.

        For example:

        .. doctest::

            >>> data.has_data1d("batch#age")
            True

            >>> data.has_data1d("batch#height")
            False
        """
        return self.chain.has_data1d(name)

    def get_vector(self, name: str) -> Vector:
        """
        Get the ``name`` 1D data (which must exist) as a `.Vector`.

        The name is the name of some 1D data as described :ref:`above <1d_names>`.

        For example:

        .. doctest::

            >>> data.get_vector("batch#age")
            array([51, 38, 21, 31, 26, 43, 36, 27, 33, 45, 49, 41, 45])
        """
        parsed_name = ParsedName(full_name=name, dataset_name=self.name)
        assert (
            parsed_name.ndim == 1
        ), f"{parsed_name.ndim}D data name: {name} given to get_vector for the data set: {self.name}"
        return be_vector(self._get_parsed(parsed_name)[1])

    def get_series(self, name: str) -> Series:
        """
        Get the ``name`` 1D data (which must exist) as a ``pandas.Series``.

        The name is the name of some 1D data as described :ref:`above <1d_names>`.

        .. doctest::

            >>> data.get_series("batch#age")
            Batch_0     51
            Batch_1     38
            Batch_2     21
            Batch_3     31
            Batch_4     26
            Batch_5     43
            Batch_6     36
            Batch_7     27
            Batch_8     33
            Batch_9     45
            Batch_10    49
            Batch_11    41
            Batch_12    45
            dtype: int64
        """
        vector = self.get_vector(name)
        axis = prefix(name, "#")
        index = self.axis_entries(axis)
        return freeze(optimize(pd.Series(vector, index=index)))

    def data2d_names(self, axes: Union[str, Tuple[str, str]], *, full: bool = True) -> List[str]:
        """
        Return the names of the 2D data that exists in the data set for a specific pair of ``axes`` (which must exist).

        The returned names are in the format ``rows_axis,columns_axis#name`` which uniquely identifies the 2D data. If
        not ``full``, the returned names include only the simple ``name`` without the ``row_axis,columns_axis#`` prefix.

        .. note::

            Data will be listed in the results even if it is only stored in the other layout (that is, as
            ``columns_axis,rows_axis#name``). Such data can still be fetched (e.g. using `.get_matrix`), in which case
            it will be re-layout internally (and the result will be cached in `.derived`).

        .. doctest::

            >>> data.data2d_names("metacell,gene")
            ['metacell,gene#UMIs']

            >>> data.data2d_names("metacell,gene", full=False)
            ['UMIs']
        """
        if isinstance(axes, str):
            parts = axes.split(",")
            assert len(parts) == 2, f"{len(axes)}D axes: {axes} for: data2d_names for the data set: {self.name}"
            axes = (parts[0], parts[1])
        assert self.has_axis(axes[0]), f"missing axis: {axes[0]} in the data set: {self.name}"
        assert self.has_axis(axes[1]), f"missing axis: {axes[1]} in the data set: {self.name}"

        names_set = set(self.chain._data2d_names(axes))
        names_set.update([transpose_name(name) for name in self.chain._data2d_names((axes[1], axes[0]))])
        names = sorted(names_set)
        if not full:
            names = [suffix(name, "#") for name in names]
        return names

    def has_data2d(self, name: str) -> bool:
        """
        Check whether the ``name`` 2D data exists.

        The name must be in the format ``rows_axis,columns_axis#name`` which uniquely identifies the 2D data.

        This will also succeed if only the transposed ``columns_axis,rows_axis#name`` data exists in the data set.
        However, fetching the data in the specified order is likely to be less efficient.

        For example:

        .. doctest::

            >>> data.has_data2d("cell,gene#UMIs")
            True

            >>> data.has_data2d("cell,gene#fraction")
            False
        """
        return self.chain.has_data2d(name) or self.chain.has_data2d(transpose_name(name))

    def get_matrix(self, name: str) -> MatrixInRows:
        """
        Get the ``name`` 2D data (which must exist) as a `.MatrixInRows`.

        The name is the name of some 2D data as described :ref:`above <2d_names>`.

        For example:

        .. doctest::

            >>> data.get_matrix("metacell,gene#UMIs")
            array([[  5.,   6.,  66.,   1.,   1.,   1.,   0., 110.,  13.,   1.],
                   [ 13.,   2.,   1.,   2.,   1.,   3.,   2.,   3.,   7.,   1.],
                   [211.,   1.,   2.,   0.,  91.,   0.,   0.,   1.,   2.,   4.],
                   [  1.,   0., 179.,   1.,   0.,   2.,   0.,   9.,   1.,   2.],
                   [  3.,   0.,   2.,  18.,   1.,   1.,   1.,   1., 126.,   1.],
                   [ 14.,   0.,   1.,   1.,   2.,  10.,   3.,   3.,   6.,   6.],
                   [  3.,   2.,   0.,   0.,   1.,   2.,   0.,   2.,   2.,   3.],
                   [  0.,   1.,   0.,   0.,   0.,   1.,   1.,   0.,   5.,   1.],
                   [ 62.,   0.,   0.,   0.,   2.,   0.,   2.,   0.,   1.,   0.],
                   [326.,   0.,   0.,   0., 151.,   0.,   0.,   1.,   0.,   2.]],
                  dtype=float32)
        """
        parsed_name = ParsedName(full_name=name, dataset_name=self.name)
        assert (
            parsed_name.ndim == 2
        ), f"{parsed_name.ndim}D data name: {name} given to get_matrix for the data set: {self.name}"
        return be_matrix_in_rows(self._get_parsed(parsed_name)[1])

    def get_frame(self, name: str) -> FrameInRows:
        """
        Get the ``name`` 2D data (which must exist) as a ``pandas.DataFrame``.

        The name is the name of some 2D data as described :ref:`above <2d_names>`.

        .. note::

            Storing `.Sparse` data in a ``pandas.DataFrame`` fails in various unpleasant ways. Therefore, data for
            ``get_frame`` is always returned in a `.Dense` format. Do **not** call ``get_frame`` unless you are certain
            that the data size is "within reason", or that the data is memory-mapped from a `.Dense` format on disk. In
            one of our data sets, calling ``get_frame("cell,gene#UMIs")`` would result in creating a ``numpy.ndarray``
            of ~240GB(!), compared to the "mere" ~6GB needed to hold the data in a ``scipy.csr_matrix``.

        For example:

        .. doctest::

            >>> data.get_frame("metacell,gene#UMIs")  # doctest: +ELLIPSIS
            gene        RSPO3  FOXA1   WNT6  TNNI1  MSGN1  LMO2  SFRP5   DLX5  ITGA4  FOXA2
            metacell...
            Metacell_0    5.0    6.0   66.0    1.0    1.0   1.0    0.0  110.0   13.0    1.0
            Metacell_1   13.0    2.0    1.0    2.0    1.0   3.0    2.0    3.0    7.0    1.0
            Metacell_2  211.0    1.0    2.0    0.0   91.0   0.0    0.0    1.0    2.0    4.0
            Metacell_3    1.0    0.0  179.0    1.0    0.0   2.0    0.0    9.0    1.0    2.0
            Metacell_4    3.0    0.0    2.0   18.0    1.0   1.0    1.0    1.0  126.0    1.0
            Metacell_5   14.0    0.0    1.0    1.0    2.0  10.0    3.0    3.0    6.0    6.0
            Metacell_6    3.0    2.0    0.0    0.0    1.0   2.0    0.0    2.0    2.0    3.0
            Metacell_7    0.0    1.0    0.0    0.0    0.0   1.0    1.0    0.0    5.0    1.0
            Metacell_8   62.0    0.0    0.0    0.0    2.0   0.0    2.0    0.0    1.0    0.0
            Metacell_9  326.0    0.0    0.0    0.0  151.0   0.0    0.0    1.0    0.0    2.0
        """
        name += "|Densify"
        dense = be_dense_in_rows(self.get_matrix(name))
        axes = prefix(name, "#").split(",")
        index = self.axis_entries(axes[0])
        columns = self.axis_entries(axes[1])
        frame = pd.DataFrame(dense, index=index, columns=columns)
        frame.index.name = axes[0]
        frame.columns.name = axes[1]
        return freeze(optimize(frame))

    def get_columns(self, axis: str, columns: Optional[Sequence[str]] = None) -> FrameInColumns:
        """
        Get an arbitrary collection of 1D data for the same ``axis`` as ``columns`` of a ``pandas.DataFrame``.

        The returned data will always be in `.COLUMN_MAJOR` order.

        If no ``columns`` are specified, returns all the 1D properties for the ``axis``, in alphabetical order (that is,
        as if ``columns`` was set to `.data1d_names` with ``full=False`` for the ``axis``).

        The specified ``columns`` names should only be the suffix following the ``axis#`` prefix in the 1D name
        of the data, as described :ref:`above <1d_names>`.

        For example:

        .. doctest::

            >>> data.get_columns("batch")  # doctest: +ELLIPSIS
                     age     sex
            batch...
            Batch_0   51  female
            Batch_1   38  female
            Batch_2   21    male
            Batch_3   31  female
            Batch_4   26    male
            Batch_5   43  female
            Batch_6   36  female
            Batch_7   27    male
            Batch_8   33    male
            Batch_9   45  female
            Batch_10  49    male
            Batch_11  41    male
            Batch_12  45    male
        """
        index = self.axis_entries(axis)
        columns = columns or self.data1d_names(axis, full=False)
        data = {column: self.get_vector(f"{axis}#{column}") for column in columns}
        frame = pd.DataFrame(data, index=index)
        frame.index.name = axis
        return freeze(optimize(frame))

    def view(
        self,
        *,
        axes: Optional[Mapping[str, Union[None, str, AnyData, AxisView]]] = None,
        data: Optional[Mapping[str, Optional[str]]] = None,
        name: str = ".view#",
        cache: Optional[StorageWriter] = None,
        hide_implicit: bool = False,
    ) -> "DafReader":
        """
        Create a read-only view of the data set.

        This can be used to create slices of some axes, rename axes and/or data, and/or hide some data. It is a wrapper
        around the constructor of `.StorageView`; see there for the semantics of the parameters, with the exception that
        here keys of the ``data`` dictionary may be *any* data name, including derived data.

        If the ``name`` starts with ``.``, it is appended to both the `.StorageView` and the `.DafReader` names. If the
        name ends with ``#``, we append the object id to it to make it unique.

        .. note::

            If any of the axes is sliced, the view will ignore any derived data based on the sliced axes. While some
            derived data is safe to slice, some isn't, and it isn't easy to tell the difference; for example, when
            slicing the ``gene`` axis, then ``cell,gene#Log,...`` is safe to slice, but
            ``cell,gene#Folds|Significant,...`` is not. The code therefore plays it safe by ignoring any derived data
            using any of the sliced axes.

        For example:

        .. doctest::

            >>> view = data.view(axes=dict(gene=['FOXA1', 'FOXA2']))
            >>> view.axis_entries("gene")
            array(['FOXA1', 'FOXA2'], dtype=object)

        .. doctest::

            >>> view = data.view(data={"metacell,gene#UMIs|Fraction": "fraction"})
            >>> view.get_series("gene#metacell=Metacell_0,fraction")
            RSPO3    0.024510
            FOXA1    0.029412
            WNT6     0.323529
            TNNI1    0.004902
            MSGN1    0.004902
            LMO2     0.004902
            SFRP5    0.000000
            DLX5     0.539216
            ITGA4    0.063725
            FOXA2    0.004902
            dtype: float32
        """
        # pylint: disable=duplicate-code

        if name.startswith("."):
            name = self.name + name

        unique: Optional[List[None]] = None
        if name.endswith("#"):
            unique = []
            name = name + str(id(unique))

        # pylint: enable=duplicate-code

        for axis in axes or {}:
            assert self.has_axis(axis), f"missing axis: {axis} in the data set: {self.name}"

        # pylint: disable=duplicate-code

        canonical_data: Dict[str, Optional[str]] = {}
        for data_name, data_alias in (data or {}).items():
            parsed_name = ParsedName(full_name=data_name, dataset_name=self.name)
            canonical_name = self._get_parsed(parsed_name)[0]
            canonical_data[canonical_name] = data_alias

        # pylint: enable=duplicate-code

        view = DafReader(
            StorageView(
                self._view_base(axes, hide_implicit, name),
                axes=axes,
                data=canonical_data,
                cache=cache,
                hide_implicit=hide_implicit,
                name=name + ".base",
            ),
            name=name,
        )

        if unique is not None:
            setattr(view, "__daf_unique__", unique)  # Prevent it from being garbage collected.

        return view

    def _view_base(
        self, axes: Optional[Mapping[str, Union[None, str, AnyData, AxisView]]], hide_implicit: bool, name: str
    ) -> StorageReader:
        return StorageChain([self._derived_filtered(axes, hide_implicit, name), self.base], name=name + ".chain")

    def _derived_filtered(
        self, axes: Optional[Mapping[str, Union[None, str, AnyData, AxisView]]], hide_implicit: bool, name: str
    ) -> StorageReader:
        axes = axes or {}
        hide_axes: Dict[str, None] = {}
        for axis in self.derived._axis_names():
            hide_axis = False
            if axis in axes:
                axis_view = axes[axis]
                hide_axis = not isinstance(axis_view, str) and (
                    not isinstance(axis_view, AxisView) or axis_view.entries is not None
                )
            else:
                hide_axis = hide_implicit
            if hide_axis:
                hide_axes[axis] = None

        if len(hide_axes) == 0:
            return self.derived

        return StorageView(self.derived, axes=hide_axes, name=name + ".derived.filtered")

    def _get_parsed(self, parsed_name: ParsedName) -> Tuple[str, Any]:
        base_data = self._get_base_data(parsed_name.base)

        if len(parsed_name.operations) == 0:
            return parsed_name.base.name, base_data

        pipeline_state = PipelineState(parsed_name.base, base_data)
        for operation_name in parsed_name.operations:
            self._pipeline_step(pipeline_state, operation_name)

        return pipeline_state.canonical, pipeline_state.data

    def _get_base_data(self, base_name: BaseName) -> Any:
        if len(base_name.axes) == 0:
            return self._get_base_item(base_name)

        if len(base_name.axes) == 1:
            return self._get_base_vector(base_name)

        assert len(base_name.axes) == 2
        return self._get_base_matrix(base_name)

    def _get_base_item(self, base_name: BaseName) -> Any:
        assert len(base_name.properties) == 1
        assert self.has_item(
            base_name.properties[0]
        ), f"missing item: {base_name.properties[0]} in the data set: {self.name}"
        return self.chain._get_item(base_name.properties[0])

    def _get_base_vector(self, base_name: BaseName) -> Any:
        value: Optional[Vector] = None
        axis = base_name.axes[0]
        for next_property in base_name.properties:
            next_name = f"{axis}#{next_property}"
            if next_property == "":
                next_value = self.axis_entries(axis)
            else:
                size = self.axis_size(axis)
                assert self.chain.has_data1d(next_name), f"missing 1D data: {next_name} in the data set: {self.name}"
                next_value = freeze(optimize(be_vector(as_vector(self.chain._get_data1d(axis, next_name)), size=size)))

            if value is None:
                value = next_value
            elif has_dtype(value, STR_DTYPE):
                next_series = pd.Series(next_value, index=self.chain._axis_entries(axis))
                value = freeze(optimize(as_vector(next_series[value])))
            else:
                value = freeze(optimize(next_value[value]))
            axis = next_property
            if not self.chain.has_axis(axis):
                axis = prefix(axis, ".")

        assert value is not None
        entry = base_name.entries[0]
        if entry is not None:
            index = self.axis_index(base_name.axes[0], entry)
            value = value[index]
            assert value is not None

        if len(base_name.properties) > 1:
            self.derived.set_vector(base_name.name, value)

        return value

    def _get_base_matrix(self, base_name: BaseName) -> Any:
        assert len(base_name.properties) == 1

        base_data2d: Any
        base_matrix: Optional[Matrix] = None
        base_matrix_in_rows: Optional[MatrixInRows] = None

        rows_axis, columns_axis = base_name.axes
        rows_size = self.axis_size(rows_axis)
        columns_size = self.axis_size(columns_axis)
        row_entry, column_entry = base_name.entries
        row_index = None if row_entry is None else self.axis_index(rows_axis, row_entry)
        column_index = None if column_entry is None else self.axis_index(columns_axis, column_entry)
        name = f"{rows_axis},{columns_axis}#{base_name.properties[0]}"

        if self.chain.has_data2d(name):
            base_data2d = self.chain.get_data2d(name)
            base_matrix = be_matrix(as_matrix(base_data2d), shape=(rows_size, columns_size))
            if is_matrix_in_rows(base_matrix):
                base_matrix_in_rows = base_matrix

        if base_matrix_in_rows is None:
            transposed_name = f"{columns_axis},{rows_axis}#{base_name.properties[0]}"
            if self.chain.has_data2d(transposed_name):
                base_data2d = self.chain.get_data2d(transposed_name)
                base_matrix = be_matrix(as_matrix(base_data2d).transpose(), shape=(rows_size, columns_size))
                if is_matrix_in_rows(base_matrix):
                    base_matrix_in_rows = base_matrix

        if base_matrix_in_rows is None:
            assert base_matrix is not None, f"missing 2D data: {name} in the data set: {self.name}"
            base_matrix_in_rows = as_layout(base_matrix, ROW_MAJOR)

        base_matrix_in_rows = freeze(optimize(base_matrix_in_rows))
        if id(base_matrix_in_rows) != id(base_data2d):
            self.derived.set_matrix(name, base_matrix_in_rows)

        if row_index is None:
            assert column_index is None
            return base_matrix_in_rows

        if column_index is None:
            return as_vector(base_matrix_in_rows[row_index, :])

        return base_matrix_in_rows[row_index, column_index]

    def _pipeline_step(self, pipeline_state: PipelineState, operation_name: OperationName) -> None:
        pipeline_state.pipeline += "|" + operation_name.name

        operation_object = operation_name.klass(_input_dtype=str(pipeline_state.data.dtype), **operation_name.kwargs)
        assert isinstance(operation_object, (_operations.ElementWise, _operations.Reduction))

        canonical_before = pipeline_state.canonical
        pipeline_state.canonical += "|" + operation_object.canonical

        if isinstance(operation_object, _operations.Reduction):
            pipeline_state.canonical = pipeline_state.canonical.replace("#", ",", 1)
            if pipeline_state.ndim == 2:
                pipeline_state.canonical = pipeline_state.canonical.replace(",", "#", 1)
            pipeline_state.ndim -= 1

        cache = operation_name.cache and pipeline_state.allow_cache
        if self._fetch_derived(pipeline_state):
            cache = False
        elif isinstance(operation_object, _operations.Reduction):
            DafReader._compute_reduction(pipeline_state, operation_object)
        else:
            nop, cache = self._compute_element_wise(pipeline_state, operation_object, cache)
            if nop:
                assert not cache
                pipeline_state.canonical = canonical_before

        if pipeline_state.ndim > 0:
            pipeline_state.data = freeze(optimize(pipeline_state.data))

        if cache:
            if pipeline_state.ndim == 2:
                self.derived.set_matrix(pipeline_state.canonical, be_matrix_in_rows(pipeline_state.data))
            elif pipeline_state.ndim == 1:
                self.derived.set_vector(pipeline_state.canonical, be_vector(pipeline_state.data))
            else:
                self.derived.set_item(pipeline_state.canonical, pipeline_state.data)

    def _fetch_derived(self, pipeline_state: PipelineState) -> bool:
        if pipeline_state.ndim == 2 and self.derived.has_data2d(pipeline_state.canonical):
            pipeline_state.data = as_matrix(self.derived.get_data2d(pipeline_state.canonical))
            return True

        if pipeline_state.ndim == 1 and self.derived.has_data1d(pipeline_state.canonical):
            pipeline_state.data = as_vector(self.derived.get_data1d(pipeline_state.canonical))
            return True

        if pipeline_state.ndim == 0 and self.derived.has_item(pipeline_state.canonical):
            pipeline_state.data = self.derived.get_item(pipeline_state.canonical)
            return True

        return False

    @staticmethod
    def _compute_reduction(pipeline_state: PipelineState, operation_object: _operations.Reduction) -> None:
        dtype = operation_object.dtype
        shape = pipeline_state.data.shape

        if is_dense_in_rows(pipeline_state.data):
            pipeline_state.data = be_vector(
                operation_object.dense_to_vector(pipeline_state.data), dtype=dtype, size=shape[0]
            )

        elif is_sparse_in_rows(pipeline_state.data):
            pipeline_state.data = be_vector(
                operation_object.sparse_to_vector(pipeline_state.data),
                dtype=dtype,
                size=shape[0],
            )

        else:
            pipeline_state.data = operation_object.vector_to_scalar(be_vector(pipeline_state.data))

    def _compute_element_wise(  # pylint: disable=too-many-return-statements
        self, pipeline_state: PipelineState, operation_object: _operations.ElementWise, cache: bool
    ) -> Tuple[bool, bool]:
        dtype = operation_object.dtype
        shape = pipeline_state.data.shape

        if is_dense_in_rows(pipeline_state.data):

            if operation_object.sparsifies:
                pipeline_state.data = be_sparse_in_rows(
                    operation_object.dense_to_sparse(pipeline_state.data), dtype=dtype, shape=shape
                )
                return False, cache

            if operation_object.nop:
                return True, False

            if cache:
                with self.derived._create_dense_in_rows(
                    pipeline_state.canonical, axes=pipeline_state.axes, dtype=dtype, shape=shape
                ) as output_data:
                    operation_object.dense_to_dense(pipeline_state.data, output_data)
                    pipeline_state.data = output_data
                return False, False

            output_data = be_dense_in_rows(np.empty(pipeline_state.data.shape, dtype=dtype))
            operation_object.dense_to_dense(pipeline_state.data, output_data)
            pipeline_state.data = output_data
            return False, False

        if is_sparse_in_rows(pipeline_state.data):

            if operation_object.densifies:
                if cache:
                    with self.derived._create_dense_in_rows(
                        pipeline_state.canonical, axes=pipeline_state.axes, dtype=dtype, shape=shape
                    ) as output_data:
                        operation_object.sparse_to_dense(pipeline_state.data, output_data)
                        pipeline_state.data = output_data
                    return False, False

                output_data = be_dense_in_rows(np.empty(pipeline_state.data.shape, dtype=dtype))
                operation_object.sparse_to_dense(pipeline_state.data, output_data)
                pipeline_state.data = output_data
                return False, False

            if operation_object.nop:
                return True, False

            pipeline_state.data = be_sparse_in_rows(
                operation_object.sparse_to_sparse(pipeline_state.data), dtype=dtype, shape=shape
            )
            return False, cache

        assert is_vector(pipeline_state.data)

        if operation_object.nop:
            return True, False

        pipeline_state.data = be_vector(
            operation_object.vector_to_vector(be_vector(pipeline_state.data)), dtype=dtype, size=shape[0]
        )
        return False, cache


def transpose_name(name: str) -> str:
    """
    Given a 2D data name ``rows_axis,columns_axis#name`` return the transposed data name
    ``columns_axis,rows_axis#name``.

    .. note::

        This will refuse to transpose pipelined names ``rows_axis,columns_axis#name|operation|...`` as doing so would
        change the meaning of the name. For example, ``cell,gene#UMIs|Sum`` gives the sum of the UMIs of all the genes
        in each cell, while ``gene,cell#UMIs|Sum`` gives the sum of the UMIs for all the cells each gene.

    For example:

    .. doctest::

        >>> daf.transpose_name("metacell,gene#UMIs")
        'gene,metacell#UMIs'
    """
    assert "|" not in name, f"transposing the pipelined name: {name}"

    name_parts = name.split("#", 1)

    axes = name_parts[0].split(",")
    assert len(axes) == 2, f"invalid 2D data name: {name}"

    name_parts[0] = ",".join(reversed(axes))
    return "#".join(name_parts)
