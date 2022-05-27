"""
In general ``daf`` assumes that stored data is not modified in-place, as this would break the caching mechanisms which
are needed for efficiency. Modifying stored data is a bad idea in general regardless of caching as it would cause subtle
bugs when analysis code is reordered.

At the same time, Python doesn't really have a notion of immutable data when it comes to complex data structures.
However, ``numpy`` does have a concept of read-only data, so we make use of it here, and extend it to deal with
``pandas`` and ``scipy.sparse`` data as well (as they use ``numpy`` data under the hood).

In general, ``daf`` always freezes data when it is stored, and accesses return frozen data, to protect against
accidental in-place modification of the stored data.

The code in this module allows to manually :py:obj:`~freeze`, :py:obj:`~unfreeze`, or test whether data
:py:obj:`~is_frozen`, using the ``numpy`` capabilities. In addition, in cases you *really* know what you are doing, it
allows you to temporary modify :py:obj:`~unfrozen` data.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import descriptions as _descriptions
from . import frames as _frames
from . import matrices as _matrices
from . import vectors as _vectors

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "freeze",
    "unfreeze",
    "is_frozen",
    "unfrozen",
]


T = TypeVar("T", bound=Union[_vectors.Vector, _matrices.Matrix, _frames.Frame])


def freeze(data: T) -> T:
    """
    Ensure that some 1/2D data is protected against modification.

    This **tries** to freeze the data in place, but for ``pandas`` we are forced to return a new frozen object, because
    "reasons". Hence the safe idiom is ``data = freeze(data)``. Sigh.
    """
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(freeze(data.values), index=data.index, columns=data.columns)
    if isinstance(data, pd.Series):
        return pd.Series(freeze(data.values), index=data.index)
    if isinstance(data, np.ndarray):
        data.flags.writeable = False
        return data  # type: ignore
    if isinstance(data, (sp.csr_matrix, sp.csc_matrix)):
        assert data.indices.flags.writeable == data.indptr.flags.writeable == data.data.flags.writeable
        data.indices.flags.writeable = data.indptr.flags.writeable = data.data.flags.writeable = False
        return data
    assert False, f"expected a matrix or a vector, got {_descriptions.data_description(data)}"


def unfreeze(data: T) -> T:
    """
    Ensure that some 1/2D data is not protected against modification.

    This **tries** to unfreeze the data in place, but because ``pandas`` has strange behavior, we are forced to return a
    new frozen object, (this is only a wrapper, the data itself is not copied). Hence the safe idiom is
    ``data = freeze(data)``. Sigh.
    """
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(unfreeze(data.values), index=data.index, columns=data.columns)
    if isinstance(data, pd.Series):
        return pd.Series(unfreeze(data.values), index=data.index)
    if isinstance(data, np.ndarray):
        data.flags.writeable = True
        return data  # type: ignore
    if isinstance(data, (sp.csr_matrix, sp.csc_matrix)):
        assert data.indices.flags.writeable == data.indptr.flags.writeable == data.data.flags.writeable
        data.indices.flags.writeable = data.indptr.flags.writeable = data.data.flags.writeable = True
        return data
    assert False, f"expected a matrix or a vector, got {_descriptions.data_description(data)}"


def is_frozen(data: Union[_vectors.Vector, _matrices.Matrix, _frames.Frame]) -> bool:
    """
    Test whether some 1/2D data is protected against modification.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
    if isinstance(data, np.ndarray):
        return not data.flags.writeable
    if isinstance(data, (sp.csr_matrix, sp.csc_matrix)):
        assert data.indices.flags.writeable == data.indptr.flags.writeable == data.data.flags.writeable
        return not data.data.flags.writeable
    assert False, f"expected a matrix or a vector, got {_descriptions.data_description(data)}"


@contextmanager
def unfrozen(data: T) -> Generator[T, None, None]:
    """
    Execute some in-place modification, temporarily unfreezing the 1/2D data.

    Expected usage is:

    .. code::

        data = freeze(data)
        # The ``data`` is immutable here.

        with unfrozen(data) as melted:
            # It is crucial you do **not** write ``as data`` - this will overwrite ``data`` with a mutable reference!
            # ``melted`` data is writable here.

        # The ``data`` stays immutable here, as long as you didn't write ``as data`` above.
    """
    was_frozen = is_frozen(data)
    if was_frozen:
        data = unfreeze(data)

    try:
        yield data

    finally:
        if was_frozen:
            freeze(data)
