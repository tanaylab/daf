"""
Data in Axes in Formats

.. todo::

    Extend ``daf`` to support "masked arrays" for storing nullable integers and nullable Booleans. This requires
    accompanying each nullable array with a Boolean mask of valid elements, and using the appropriate ``numpy`` and
    ``pandas`` APIs to deal with this.
"""

import sys

__author__ = "Oren Ben-Kiki"
__email__ = "oren@ben-kiki.org"
__version__ = "0.1.0-dev.1"


# pylint: disable=wrong-import-position


from .storage import *
from .typing import *
