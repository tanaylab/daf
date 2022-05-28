"""
The ``numpy`` package provides the convenient ``numpy.allclose`` function which can compare any dense data, not only
``numpy.ndarray`` but also ``pandas.DataFrame`` and ``pandas.Series`` data.

There are two issues with it, however. First, it will cheerfully compare two large matrices with incompatible layout,
which will take a very long time; Second and more importantly here, it will not compare sparse matrices, and
``scipy.sparse`` as no equivalent.

The code here provides an alternative which *only* compares data when it is efficient to do so, and also supports sparse
matrix formats (that are relevant to ``daf``).
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from . import descriptions as _descriptions
from . import layouts as _layouts
from . import optimization as _optimization

# pylint: enable=duplicate-code,cyclic-import

__all__ = ["fast_all_close"]


def fast_all_close(
    left: Any,
    right: Any,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """
    Generalize ``numpy.allclose`` to handle more types, and restrict it to only support efficient comparisons.

    For this to be efficient:

    * Both values must be vectors (``numpy`` or ``pandas``), or

    * Both values must be must be `.Dense` matrices (``numpy`` or ``pandas``), or

    * Both values must be must be `.Sparse` matrices.

    And if the values are matrices:

    * Both matrices must be in `.ROW_MAJOR` layout, or

    * Both matrices must be in `.COLUMN_MAJOR` layout.

    Otherwise the code will ``assert``.

    .. note::

        When comparing sparse matrices, the ``rtol``, ``atol`` and ``equal_nan`` values are only used to compare the
        non-zero values, after ensuring their structure is identical in both matrices. This requires both matrices to be
        `.is_optimal`.
    """

    if left.shape != right.shape:
        return False

    if (  # pylint: disable=too-many-boolean-expressions
        (
            (isinstance(left, sp.csr_matrix) and isinstance(right, sp.csr_matrix))
            or (isinstance(left, sp.csc_matrix) and isinstance(right, sp.csc_matrix))
        )
        and _optimization.is_optimal(left)
        and _optimization.is_optimal(right)
    ):
        return (
            np.allclose(left.indptr, right.indptr)
            and np.allclose(left.indices, right.indices)
            and np.allclose(left.data, right.data, rtol=rtol, atol=atol, equal_nan=equal_nan)
        )

    if (  # pylint: disable=too-many-boolean-expressions
        isinstance(left, (np.ndarray, pd.DataFrame, pd.Series))
        and isinstance(right, (np.ndarray, pd.DataFrame, pd.Series))
        and (
            left.ndim == right.ndim == 1
            or (
                left.ndim == right.ndim == 2
                and (
                    (_layouts.ROW_MAJOR.is_layout_of(left) and _layouts.ROW_MAJOR.is_layout_of(right))
                    or (_layouts.COLUMN_MAJOR.is_layout_of(left) and _layouts.COLUMN_MAJOR.is_layout_of(right))
                )
            )
        )
    ):
        return np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=equal_nan)

    assert False, f"comparing a {_descriptions.data_description(left)} with a {_descriptions.data_description(right)}"
