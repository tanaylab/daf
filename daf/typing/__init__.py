"""
Provide type annotations and support functions for code processing 1D and 2D data.

The code has to deal with many different alternative data types for what is essentially two basic data types: 2D
matrices and 1D vectors. Even though Python supports "duck typing" in theory, in practice these alternative data types
expose different APIs and require different code paths for realistic algorithms. Even simple things like copying data do
not work in a uniform, robust way.

.. note::

    Python supports "duck typing" in theory but applying this to 1D/2D data is like trying to build a bird feeder that
    works for both hummningbirds and eagles by pretending they are all ducks. This is mostly due to historical reasons
    and has little to do with inherent differences between the implementations.

Matrices in particular can be represented in a wide range of formats and variants within each format. Efficient code for
each representation requires different code paths which places a high burden on the consumers of matrix data from
``daf`` containers.

To minimize this burden, ``daf`` restricts the data types it stores to very few variants, specifically either using
row-major or column-major layout for dense matrices and either CSR or CSC format for sparse matrices, as these are the
most commonly used layouts, and require only a small number of code paths to ensure efficient computation. See the
:py:obj:`~daf.typing.layouts` for details on how ``daf`` deals with the layouts of 2D data.

Ideally we'd also like to track the data type of the elements of the data, which again is only almost compatible between
``numpy`` and ``pandas``. We don't do it directly using ``mypy`` annotations as it is not capable enough to express this
without a combinatorical explosion of data types. See :py:obj:`~daf.typing.dtypes` for details on how ``daf`` deals with
element data types.

We therefore provide here only the minimal ``mypy`` annotations allowing expressing the code's intent when it comes to
code paths, and provide some utilities to at least assert the element data type is as expected. In particular, these
type annotations only support the restricted subset we allow to store out of the full set of data types available in
``numpy``, ``pandas`` and ``scipy.sparse``.

In general we provide ``is_...`` functions that test whether some data is of some format (and also works as a ``mypy``
``TypeGuard``), ``be_...`` functions that assert that some data is of some format (and return it as such, to make
``mypy`` effective and happy), and ``as_...`` functions that convert data in specific formats (optionally forcing the
creation of a copy). Additional support functions are provided as needed in the separate sub-modules; the most useful
are :py:obj:`~daf.typing.layouts.as_layout` and :py:obj:`~daf.typing.optimization.optimize` which are needed to ensure
code is efficient and :py:obj:`~daf.typing.freezing.freeze` which helps protect data against accidental in-place
modification.

Since ``pandas`` and ``scipy.sparse`` provide no ``mypy`` type annotations (at least as of Python 3.10), we are forced
to introduce "fake" type annotation for their types in :py:obj:`~daf.typing.fake_pandas` and
:py:obj:`~daf.typing.fake_sparse`. Even though ``numpy`` does provide ``mypy`` type annotations these days, its use of a
catch-all ``numpy.ndarray`` type is not sufficient for capturing the code intent. Therefore, in all cases, the results
of any computation on the data types provided here is effectively ``Any``, which negates the usefulness of using
``mypy``.

The bottom line is that, as always, type annotations in Python are optional. You can ignore them (which makes sense for
quick-and-dirty scripts, where correctness and performance are trivial), and if you want to benefit from them (for
serious code), you need to put in the extra work (adding type annotations and a liberal amount of ``be_...`` calls). The
goal of this module is merely to make it *possible* to do so with the least amount of pain. But even if you do not use
type annotations at all, the support functions provided here are important for writing safe, efficient code.

.. note::

    The type annotations here require advanced ``mypy`` features. The code itself will run find in older Python versions
    (3.7 and above), type-checking the code will only work on later versions (3.10 and above).
"""

from .array1d import *
from .array2d import *
from .dense import *
from .descriptions import *
from .dtypes import *
from .fake_pandas import *
from .fake_sparse import *
from .freezing import *
from .grids import *
from .layouts import *
from .matrices import *
from .optimization import *
from .series import *
from .sparse import *
from .tables import *
from .vectors import *
