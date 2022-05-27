"""
Python 1D/2D data has a notion of ``dtype`` to describe the data type of a single elements. In general it is possible to
treat this as a string, e.g. ``bool``, ``float32`` or ``int8`` and be done. This works well for numeric types, at least
as long as you stick to explicitly sized types.

Both ``numpy`` and ``pandas`` also allow to store arbitrary data as "object". In contrast, ``daf`` does not allow
storing such arbitrary data as vector or matrix elements, but does allow storing strings, which greatly complicates the
issue. The ``numpy`` data types for representing strings try to deal with the maximal string size (presumably for
efficiency reasons) with ``dtype`` looking like ``U5`` or ``<U12``. In contrast, ``pandas`` has no concept of a string
``dtype`` and just represents it as an ``object``. But to make things interesting, ``pandas`` also has a ``category``
dtype which it uses to represent strings-out-of-some-limited-set.

Since ``pandas`` uses ``numpy`` internally this results with inconsistent ``dtype`` value for data containing strings.
Depending on whether you access the internal ``numpy`` data or the wrapper ``pandas`` data, and whether this data was
created using a ``numpy`` or a ``pandas`` operation, you can get either ``object``, ``category``, or ``U5``, or
``<U12``, which makes testing for "string-ness" of data basically impossible.

.. note::

    Do **not** try to store arbitrary objects inside 1D/2D data in ``daf``. There is no practical way to protect against
    this, and things will fail in spectacular and unexpected ways.

The way ``daf`` deals with this mess is to restrict itself to storing just plain string data and optimistically assume
that ``object`` means ``str``. We also never store categorical data, only allowing to store plain string data.

.. note::

    In ``daf``, it makes more sense to define a "category" as an "axis", and simply store integer elements whose value
    is the index along that axis. The :py:obj:`~daf.typing.optimizations` module helps with converting categorical data
    into plain string data.

Some ``daf`` functions take a ``dtype`` (or a collection), e.g. when testing whether some data elements have an
acceptable type. This forces us to introduce a single ``dtype`` to stand for "string", which we have chosen to be ``U``.
This value has the advantage you can pass it to either ``numpy`` or ``pandas`` when **creating** new data. You can't
directly **test** for ``dtype == "U"``, of course, but if you pass ``U`` to any ``daf`` function that tests the element
data type (e.g., :py:obj:`~daf.typing.vectors.is_array1d`), then the code will do its best to test that the data
actually contains strings.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Collection
from typing import Union

import numpy as np

# pylint: disable=duplicate-code,cyclic-import

__all__ = [
    "STR_DTYPE",
    "INT_DTYPES",
    "FLOAT_DTYPES",
    "NUM_DTYPES",
    "is_dtype",
]

#: Value of ``dtype`` for strings (``U``).
#:
#: .. note::
#:
#:    This is safe to use when creating new data and when testing the data type using ``daf`` functions. However testing
#:    for ``foo.dtype == "U"`` will **always** fail because "reasons".
STR_DTYPE = "U"

#: Values of ``dtype`` for integers of any size.
INT_DTYPES = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")

#: Values of ``dtype`` for floats of any size.
FLOAT_DTYPES = ("float16", "float32", "float64")

#: Values of ``dtype`` for simple numbers (integers or floats) of any size.
NUM_DTYPES = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64")

#: All the "acceptable" data types.
#:
#: This is used as the default set of data types when testing for whether a ``dtype`` is as expected via
#: :py:obj:`~is_dtype`.
#:
#: .. note::
#:
#:    We are forced to allow ``object`` as it is used for strings, but **not** try to store arbitrary objects inside
#:    ``daf`` 1D/2D data. We also allow for ``category``, but only allow actually storing plain strings.
ALL_DTYPES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "bool",
    "category",
    "object",
)


def is_dtype(dtype: Union[str, np.dtype], dtypes: Union[None, str, Collection[str]] = None) -> bool:
    """
    Check whether a ``numpy`` ``dtype`` is one of the expected ``dtypes``.

    If no ``dtypes`` are provided, tests for :py:obj:`~ALL_DTYPES`.

    When testing for strings, use :py:obj:`~STR_DTYPE` (that is, ``U``), since ``numpy`` and ``pandas`` use many
    different actual ``dtype`` values to represent strings because "reasons".
    """
    dtype = str(dtype)

    if isinstance(dtypes, str):
        dtypes = (dtypes,)

    if dtypes is not None:
        return dtype in dtypes or (  # Simple for non-string dtypes.
            STR_DTYPE in dtypes  # But for strings...
            and (
                STR_DTYPE in dtype  # Numpy U5, <U12.
                or dtype
                in (
                    "category",  # Pandas (non-optimal) data.
                    "object",  # Optimistically assume no data of actual (non-string) objects.
                )
            )
        )

    return STR_DTYPE in dtype or dtype in ALL_DTYPES
