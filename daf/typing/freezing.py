"""
In general ``daf`` assumes that stored data is not modified in-place, as this would break the caching mechanisms which
are needed for efficiency. Modifying stored data is a bad idea in general regardless of caching as it would cause subtle
bugs when analysis code is reordered.

At the same time, Python doesn't really have a notion of immutable data when it comes to complex data structures.
However, ``numpy`` does have a concept of read-only data, so we make use of it here, and extend it to deal with
``pandas`` and ``scipy.sparse`` data as well (as they use ``numpy`` data under the hood).

In general, ``daf`` always freezes data when it is stored, and fetching data will return frozen data, to protect against
accidental in-place modification of the stored data.

The code in this module allows to manually `.freeze`, `.unfreeze`, or test whether data `.is_frozen`, using the
``numpy`` capabilities. In addition, in cases you *really* know what you are doing, it allows you to temporary modify
`.unfrozen` data.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator
from typing import TypeVar

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import descriptions as _descriptions
from . import fake_pandas as _fake_pandas  # pylint: disable=unused-import
from . import unions as _unions

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "ProperT",
    "freeze",
    "unfreeze",
    "is_frozen",
    "unfrozen",
]


#: A ``TypeVar`` bound to `.Proper`.
ProperT = TypeVar("ProperT", bound=_unions.Proper)


def freeze(data: ProperT) -> ProperT:
    """
    Ensure that some 1D/2D data is protected against modification.

    This **tries** to freeze the data in place, but because ``pandas`` has strange behavior, we are forced to return a
    new frozen object (this is only a wrapper, the data itself is not copied). Hence the safe idiom is ``data =
    freeze(data)``. Sigh.
    """
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(freeze(data.values), index=data.index, columns=data.columns)
    if isinstance(data, pd.Series):
        return pd.Series(freeze(data.values), index=data.index)
    if isinstance(data, np.ndarray):
        data.flags.writeable = False
        return data  # type: ignore
    if isinstance(data, (sp.csr_matrix, sp.csc_matrix)):
        data.indices.flags.writeable = data.indptr.flags.writeable = data.data.flags.writeable = False
        return data
    assert False, f"expected: proper 1D/2D data, got: {_descriptions.data_description(data)}"


def unfreeze(data: ProperT) -> ProperT:
    """
    Ensure that some 1D/2D data is not protected against modification.

    This **tries** to unfreeze the data in place, but because ``pandas`` has strange behavior, we are forced to return a
    new frozen object (this is only a wrapper, the data itself is not copied). Hence the safe idiom is ``data =
    unfreeze(data)``. Sigh.
    """
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(unfreeze(data.values), index=data.index, columns=data.columns)
    if isinstance(data, pd.Series):
        return pd.Series(unfreeze(data.values), index=data.index)
    if isinstance(data, np.ndarray):
        data.flags.writeable = True
        return data  # type: ignore
    if isinstance(data, (sp.csr_matrix, sp.csc_matrix)):
        data.indices.flags.writeable = data.indptr.flags.writeable = data.data.flags.writeable = True
        return data
    assert False, f"expected: proper 1D/2D data, got: {_descriptions.data_description(data)}"


def is_frozen(data: _unions.Known) -> bool:
    """
    Test whether some 1D/2D data is (known to be) protected against modification.

    .. note::

        This will fail for any ``scipy.sparse.spmatrix`` other than for ``scipy.sparse.csr_matrix`` or
        ``scipy.sparse.csc_matrix``.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
    if isinstance(data, np.ndarray):
        return not data.flags.writeable
    if isinstance(data, (sp.csr_matrix, sp.csc_matrix)):
        return not data.indices.flags.writeable or not data.indptr.flags.writeable or not data.data.flags.writeable
    assert False, f"expected: known 1D/2D data, got: {_descriptions.data_description(data)}"


@contextmanager
def unfrozen(data: ProperT) -> Generator[ProperT, None, None]:
    """
    Execute some in-place modification, temporarily unfreezing the 1D/2D data.

    Expected usage is:

    .. code::

        assert is_frozen(data)  # The ``data`` is immutable here.

        with unfrozen(data) as melted:
            # ``melted`` data is writable here.
            # Do **not** leak the reference to the ``melted`` data to outside the block.
            # In particular, do **not** use the anti-pattern ``with unfrozen(data) as data: ...``.
            assert not is_frozen(melted)

        assert is_frozen(data)  # The ``data`` is immutable here.
    """
    was_frozen = is_frozen(data)
    if was_frozen:
        data = unfreeze(data)

    try:
        yield data

    finally:
        if was_frozen:
            freeze(data)
