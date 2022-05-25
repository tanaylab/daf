"""
Utilities for testing.
"""

import warnings
from contextlib import contextmanager
from typing import Generator


def allow_np_matrix() -> None:
    """
    Disable warnings about deprecated np.matrix.
    """
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning, message=r".*the matrix subclass.*")


@contextmanager
def expect_raise(expected: str) -> Generator[None, None, None]:
    """
    Execute some code in ``with`` which should fail an assertion with the ``expected`` message.
    """
    try:
        yield
    except Exception as error:  # pylint: disable=broad-except
        actual = str(error)
        if actual == expected:
            return

        index = 0
        while index < len(actual) and index < len(expected):
            if actual[index] != expected[index]:
                break
            index += 1

        assert False, (
            "unexpected assertion;\n"
            f"expected: {expected}\n"
            f"actual:   {actual}\n"
            f"         {''.join([' '] * index)}_^_"
        )

    assert False, f"missing assertion: {expected}"
