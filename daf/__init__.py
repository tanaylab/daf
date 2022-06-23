"""
The top-level ``daf`` module re-exports everything all the sub-modules, so, for example, you can write ``from daf import
MemoryStorage`` or ``daf.MemoryStorage`` instead of the full name ``daf.storage.memory.MemoryStorage``.
"""

# See https://github.com/jwilk/python-syntax-errors
# pylint: disable=using-constant-test,missing-function-docstring,pointless-statement
if 0:

    async def function(value):
        f"{await value}"  # Python >= 3.7 is required


__author__ = "Oren Ben-Kiki"
__email__ = "oren@ben-kiki.org"
__version__ = "0.1.0"

# pylint: disable=wrong-import-position

import sys

from .access import *
from .groups import *
from .storage import *
from .typing import *
