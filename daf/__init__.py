"""
This re-exports everything in the sub-modules directly from the ``daf`` module, so, for example, you can write ``from
daf import MemoryStorage`` or ``daf.MemoryStorage`` instead of the full name ``daf.storage.memory.MemoryStorage``.

.. todo::

    Extend ``daf`` to support "masked arrays" for storing nullable integers and nullable Booleans. This requires
    accompanying each nullable array with a Boolean mask of valid elements, and using the appropriate ``numpy`` and
    ``pandas`` APIs to deal with this.
"""

# See https://github.com/jwilk/python-syntax-errors
# pylint: disable=using-constant-test,missing-function-docstring,pointless-statement
if 0:

    async def function(value):
        f"{await value}"  # Python >= 3.7 is required


__author__ = "Oren Ben-Kiki"
__email__ = "oren@ben-kiki.org"
__version__ = "0.1.0-dev.1"

# pylint: disable=wrong-import-position

import sys

from .access import *
from .storage import *
from .typing import *
