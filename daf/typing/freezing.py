"""
In general ``daf`` assumes that stored data is not modified in-place, as this would break the caching mechanisms which
are needed for efficiency. Modifying stored data is a bad idea in general regardless of caching as it would cause subtle
bugs when analysis code is reordered.

At the same time, Python doesn't really have a notion of immutable data when it comes to complex data structures.
However, ``numpy`` does have a concept of read-only data, so we make use of it here, and extend it to deal with
``pandas`` and ``scipy.sparse`` data as well (as they use ``numpy`` data under the hood).

In general, ``daf`` always freezes data when it is stored, and accesses return frozen data, to protect against
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
from typing import Union

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import descriptions as _descriptions
from . import fake_pandas as _fake_pandas  # pylint: disable=unused-import

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "freeze",
    "unfreeze",
    "is_frozen",
    "unfrozen",
]


#: Any data type that ``daf`` basically understands.
ProperData = Union[np.ndarray, sp.csr_matrix, sp.csc_matrix, _fake_pandas.PandasSeries, _fake_pandas.PandasFrame]

T = TypeVar("T", bound=ProperData)


def freeze(data: T) -> T:
    """
    Ensure that some 1/2D data is protected against modification.

    This **tries** to unfreeze the data in place, but because ``pandas`` has strange behavior, we are forced to return a
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
        assert data.indices.flags.writeable == data.indptr.flags.writeable == data.data.flags.writeable
        data.indices.flags.writeable = data.indptr.flags.writeable = data.data.flags.writeable = False
        return data
    _descriptions.assert_data(False, "matrix of vector", data, None)
    assert False, "never happens"


def unfreeze(data: T) -> T:
    """
    Ensure that some 1/2D data is not protected against modification.

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
        assert data.indices.flags.writeable == data.indptr.flags.writeable == data.data.flags.writeable
        data.indices.flags.writeable = data.indptr.flags.writeable = data.data.flags.writeable = True
        return data
    _descriptions.assert_data(False, "matrix of vector", data, None)
    assert False, "never happens"


def is_frozen(data: ProperData) -> bool:
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
    _descriptions.assert_data(False, "matrix of vector", data, None)
    assert False, "never happens"


@contextmanager
def unfrozen(data: T) -> Generator[T, None, None]:
    """
    Execute some in-place modification, temporarily unfreezing the 1/2D data.

    Expected usage is:

    .. code::

        data = freeze(data)
        # The ``data`` is immutable here.

        with unfrozen(data) as melted:
            # ``melted`` data is writable here.
            # Do **not** leak the reference to the ``melted`` data to outside the block.
            # In particular, do **not** write ``with unfrozen(data) as data:``.

        # The ``data`` stays immutable here, as long as you didn't leak ``melted`` above.
    """
    was_frozen = is_frozen(data)
    if was_frozen:
        data = unfreeze(data)

    try:
        yield data

    finally:
        if was_frozen:
            freeze(data)
