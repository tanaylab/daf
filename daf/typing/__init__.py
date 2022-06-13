"""
Provide type annotations and support functions for code processing 1D and 2D data.

The code has to deal with many different alternative data types for what is essentially two basic data types: 2D data
and 1D data. These alternative data types expose different APIs and require different code paths for realistic
algorithms. Even simple things like copying data do not work in a uniform, robust way.

.. note::

    Python supports "duck typing" in theory but applying this to 1D/2D data is like trying to build a bird feeder that
    works for both hummningbirds and eagles and penguins by pretending they are all ducks. This is mostly due to
    historical reasons and has little to do with inherent differences between the implementations.

In particular, 2D data can be represented in a wide range of formats and variants within each format. Efficient code for
each representation requires different code paths which places a burden on the consumers of 2D data from ``daf``
containers.

To minimize this burden, ``daf`` restricts the data types it stores to very few variants, specifically either using
row-major or column-major layout for dense matrices and either CSR or CSC format for sparse matrices, as these are the
most commonly used layouts. This requires only a small number of code paths to ensure efficient computation. See the
`.layouts` for details on how ``daf`` deals with the layouts of 2D data.

We also have to deal with the data type of the elements of the data (aka "dtype"), which again is only
*almost* compatible between ``numpy`` and ``pandas``. We don't do it directly using ``mypy`` annotations as it is not
capable enough to express this without a combinatorical explosion of data types. See `.dtypes` for details on how
``daf`` deals with element data types.

We therefore provide here only the minimal ``mypy`` annotations allowing to express the code's intent and select correct
and efficient code paths, and provide some utilities to at least assert the element data type is as expected. In
particular, the type annotations only support the restricted subset we allow to store out of the full set of data types
available in ``numpy``, ``pandas`` and ``scipy.sparse``.

In general we provide ``is_...`` functions that test whether some data is of some format (and also works as a ``mypy``
``TypeGuard``), ``be_...`` functions that assert that some data is of some format (and return it as such, to make
``mypy`` effective and happy), and ``as_...`` functions that convert data in specific formats (optionally forcing the
creation of a copy). Additional support functions are provided as needed in the separate sub-modules; the most useful
are `.as_layout` and `.optimize` which are needed to ensure code is efficient, and `.freeze` which helps protect data
against accidental in-place modification.

Since ``pandas`` and ``scipy.sparse`` provide no ``mypy`` type annotations (at least as of Python 3.10), we are forced
to introduce "fake" type annotation for their types in `.fake_pandas` and `.fake_sparse`. Even though ``numpy`` does
provide ``mypy`` type annotations, its use of a catch-all ``numpy.ndarray`` type is not sufficient for capturing the
code intent. Therefore, in most cases, the results of any computation on the data is effectively ``Any``, which negates
the usefulness of using ``mypy``, unless ``is...`` and ``be_...`` (and the occasional ``as_...``) are used.

The bottom line is that, as always, type annotations in Python are optional. You can ignore them (which makes sense for
quick-and-dirty scripts, where correctness and performance are trivial). If you do want to benefit from them (for
serious code), you need to put in the extra work (adding type annotations and a liberal amount of ``be_...`` calls). The
goal of this module is merely to make it *possible* to do so with the least amount of pain. But even if you do not use
type annotations at all, the support functions provided here are important for writing safe, efficient code.

.. note::

    The type annotations here require advanced ``mypy`` features. The code itself will run find in older Python versions
    (3.7 and above). However type-checking the code will only work on later versions (3.10 and above).
"""

from .optimization import *  # isort: skip

from .dense import *
from .descriptions import *
from .dtypes import *
from .fake_pandas import *
from .fake_sparse import *
from .frames import *
from .freezing import *
from .layouts import *
from .matrices import *
from .series import *
from .sparse import *
from .unions import *
from .vectors import *
