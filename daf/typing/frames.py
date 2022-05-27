""""
The types here describe 2D data where different columns may have different data types, inside a ``pandas.DataFrame``,
which can be obtained from ``daf`` storage.

Logically and operationally this is a distinct data type from a homogeneous data frame where all the data elements have
the same type (that is, :py:obj:`~daf.typing.tables.Table`). We call the heterogeneous type :py:obj:`~Frame` as this
is the terminology established by ``pandas``, which doesn't recognize the existence of the homogeneous case, so even
if/when it provides ``mypy`` annotations, we'd still need to set up the types here (similar to the problem with
``numpy.ndarray``).

In ``daf``, all heterogeneous frames are always in :py:obj:`~daf.typing.layout.COLUMN_MAJOR` layout. In general it
makes little sense to shoehorn heterogeneous column data into a single 2D array (which ``pandas`` does, at the cost of
forcing the data type of the array to become ``object`` and therefore a massive loss of efficiency), but that's just the
way it is.

In ``daf`` we never directly store heterogeneous frames, but they can be returned as a result of accessing multiple
vector data which share the same axis.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import NewType

try:
    from typing import Annotated  # pylint: disable=unused-import
    from typing import TypeGuard  # pylint: disable=unused-import
except ImportError:
    pass  # Older python versions.

import pandas as pd  # type: ignore

from . import descriptions as _descriptions
from . import fake_pandas as _fake_pandas  # pylint: disable=unused-import

# pylint: enable=duplicate-code,cyclic-import

__all__ = [
    "Frame",
    "is_frame",
    "be_frame",
]

#: 2-dimensional ``pandas`` frame with heterogeneous data in column-major layout.
Frame = NewType("Frame", "Annotated[_fake_pandas.PandasFrame, 'mixed']")


def is_frame(data: Any) -> TypeGuard[Frame]:
    """
    Check whether some ``data`` is a :py:obj:`~Frame`.

    There's no point specifying a ``dtype`` here as each column may have a different one,
    or a ``layout`` as frames are (or should) always be in column-major layout.
    """
    return isinstance(data, pd.DataFrame)


def be_frame(data: Any) -> Frame:
    """
    Assert that some ``data`` is a :py:obj:`~Frame`, and return it as such for ``mypy``.

    There's no point specifying a ``dtype`` here as each column may have a different one,
    or a ``layout`` as frames are (or should) always be in column-major layout.
    """
    _descriptions.assert_data(is_frame(data), "pandas.DataFrame", data, None)
    return data
