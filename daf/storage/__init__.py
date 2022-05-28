"""
Storage objects provide low-level API for storing data in specific formats. To extend ``daf`` to support an additional
format, implement a `.StorageReader` or a `.StorageWriter` for that format.

A storage object contains some 0D ("blob") data, a set of axes (each with a unique name for each entry), and 1D and 2D
data based on these axes.

Data is identified by its unique ``name`` for 0D data, by the ``axis:name`` unique combination for 1D data, and by the
``rows_axis,columns_axis:name`` unique combination for 2D data. 2D data is always stored in row-major order. Thus for a
storage object to report containing both ``foo,bar:baz`` and ``bar,foo:baz``, it must be that it contains both the data
and its transpose, each in row-major order. Often a storage object would only contain a single copy of the data, so it
would only report containing one of ``foo,bar:baz`` and ``bar,foo:baz``.
"""

from .chains import *
from .interface import *
from .memory import *
from .none import *
