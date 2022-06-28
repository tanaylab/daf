"""
Storage objects provide low-level API for storing data in specific formats. To extend ``daf`` to support an additional
format, implement a `.StorageReader` or a `.StorageWriter` for that format.

A storage object contains some 0D data, a set of axes (each with a unique name for each entry), and 1D and 2D data based
on these axes.

Data is identified by its unique ``property`` name for 0D data, by the ``axis;property`` name for 1D data, and by the
``rows_axis,columns_axis;property`` name for 2D data. Such data need only be reported under one name, that is, there is
no requirement to also report the transposed data by the name ``columns_axis,rows_axis;property``. When picking which of
the two to report, ideally the data should be reported such that the result would be in `.ROW_MAJOR` order, if at all
possible. If, however, the storage contains two copies of the same data, in different internal layouts, it should expose
both ``foo,bar;baz`` and ``bar,foo;baz`` such that each is in `.ROW_MAJOR` order, as this will allow the higher code
layers to be more efficient.

.. todo::

    Provide a ``ConcatStorage`` that allows concatenating two data sets along a single axis, reusing all the other axes
    (e.g., concatenating two data sets for distinct cells using identical genes into a single data set containing both
    sets of cells).
"""

from .chains import *
from .files import *
from .h5fs import *
from .interface import *
from .memory import *
from .none import *
from .views import *

from .anndata import *  # isort: skip
