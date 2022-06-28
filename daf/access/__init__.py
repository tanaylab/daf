"""
High-level API for accessing data in a ``daf`` data set.

These interfaces are intended for ``daf`` users (that is, applications built on top of ``daf``). They allows placing any
"reasonable" type of data into a `.DafWriter`, while ensuring that accessing data in a  `.DafReader` will always return
"clean" data. For example, 2D data returned by `.DafReader` is always `.is_optimal` `.is_frozen` `.MatrixInRows`
regardless of whatever was put into it.

This is in contrast to the low-level `.StorageReader` and `.StorageWriter` interface which, to simplify writing storage
adapters, requires storing "clean" data as above, but does not guarantee anything when accessing the stored data. That
is, `.DafReader` and `.DafWriter` satisfy the `robustness principle
<https://en.wikipedia.org/wiki/Robustness_principle>`_ both "up" towards the application and "down" towards the storage
format adapters.

Accessing data in ``daf`` is based on string names in the following format(s):

* 0D data is identified by a simple ``name``, e.g. ``doi_url`` might be a string describing the overall data set.
  There is no restriction on the data type of 0D data except that it should be reasonably de/serializable to allow
  storing it in a disk file.

* 1D/2D data is specified along some axes, where each ``axis`` has a simple name and a string name for each entry along
  the axis.

* 1D data along some axis is identified by ``axis;name``, e.g. ``cell;age`` might assign an age to every cell in the
  data set. Such data is returned as a ``numpy`` 1D array (that is, `.Vector`) or as a ``pandas.Series``.

* 2D data along two axes is identifies by ``rows_axis,columns_axis;name``, e.g. ``cell,gene;UMIs`` would give the number
  of unique molecular identifiers (that is, the count of mRNA molecules) for each gene in each cell.

  All such data is provided in `.ROW_MAJOR` order; that is, in the above example, each row will describe a cell, and
  will contain (consecutively in memory) the UMIs of each gene. Requesting ``gene,cell;UMIs`` will return data where
  each row describes a cell, and will contain (consecutively in memory) its UMIs in each cell.

  .. note::

    Calling ``.transpose()`` on 2D data does **not** modify the memory layout; this is why it is an extremely fast
    operation. That is, the transpose of ``cell,gene;UMIs`` data contains the same rows, columns, and values as
    ``gene,cell;UMIs`` data, but the former will be in `.COLUMN_MAJOR` layout and the latter will be in `.ROW_MAJOR`
    layout. The two may be "equal" but will **not** be identical when it comes to performance (for non-trivial data
    sizes). For example, summing the UMIs of each cell would be **much** slower for the ``gene,cell;UMIs`` data. It is
    therefore **important** to keep track of the memory order of any non-trivial 2D data, and ensure operations are
    applied to the right layout. Otherwise the code will experience **extreme** slowdowns.

  2D data can be stored in either dense (``numpy`` 2D array) or sparse (``scipy.sparse.csr_matrix`` and
  ``scipy.sparse.csc_matrix``) formats. Which one you'll get when accessing the data will depend on what was stored.
  This allows for efficient storage and processing of large sparse matrices, at the cost of requiring the users to
  examine the fetched data (e.g. using `.is_sparse` or `.is_dense`) to pick the right code path to process it (since
  ``numpy`` arrays and ``scipy.sparse`` matrices don't really support the same operations).

  You can also request the data as a ``pandas.DataFrame`` (that is, `.Frame`), in which case, due to ``pandas``
  limitations, the data will always be returned in the dense (``numpy``) format. The index and columns of the frame
  will be the relevant axis entries.
"""

from .operations import *
from .readers import *
from .writers import *
