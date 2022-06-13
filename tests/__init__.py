"""
Utilities for testing.
"""

import re
import warnings
from contextlib import contextmanager
from textwrap import dedent
from typing import Generator
from typing import List
from typing import Union

import yaml  # type: ignore

from daf import DafReader
from daf import StorageReader


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
        assert False, "unexpected assertion;\n" + _diff(expected, actual)

    assert False, f"missing expected assertion: {expected}"


ID_PATTERN = re.compile("#\\d+")


def expect_description(
    data: Union[StorageReader, DafReader], *, detail: bool = False, deep: bool = False, expected: str
) -> None:
    """
    Verify the expected shallow description of some data.
    """
    expected = dedent(expected).strip()
    actual = ID_PATTERN.sub(
        "#<id>", yaml.dump(data.description(detail=detail, deep=deep), width=1000, sort_keys=False).strip()
    )
    if actual == expected:
        return
    assert False, "unexpected description;\n" + _diff(expected, actual)


def _diff(expected: str, actual: str) -> str:
    actual_lines = actual.split("\n")
    expected_lines = expected.split("\n")

    result: List[str] = []

    seen = 0
    found = False
    for (expected_line, actual_line) in zip(expected_lines, actual_lines):
        seen += 1

        if expected_line == actual_line:
            result.append(f"identical: {expected_line}\n")
            continue

        index = 0
        while index < len(actual_line) and index < len(expected_line):
            if actual_line[index] != expected_line[index]:
                break
            index += 1

        result.append(f"EXPECTED : {expected_line}\n")
        result.append(f"ACTUAL   : {actual_line}\n")
        result.append(f"DIFF     :{''.join([' '] * index)}_^_\n")
        found = True
        break

    if found:
        for actual_line in actual_lines[seen:]:
            result.append(f"actual   : {actual_line}\n")
    else:
        if len(actual_lines) < len(expected_lines):
            result.append("DIFF: actual is missing:")
            for expected_line in expected_lines[len(actual_lines) :]:
                result.append(f"missing  : {expected_line}\n")
        else:
            result.append("DIFF: actual has trailing:")
            for actual_line in actual_lines[len(expected_lines) :]:
                result.append(f"trailing : {actual_line}\n")

    return "".join(result)
