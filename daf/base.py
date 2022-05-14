"""
Base API
--------
"""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Collection
from typing import Optional

from .typing import AnyMajor
from .typing import Array1D
from .typing import Frame
from .typing import Grid
from .typing import Matrix
from .typing import Series
from .typing import Vector


class Daf(ABC):  # pylint: disable=too-few-public-methods
    """
    The common API (base class) for accessing data in axes in files.
    """

    @abstractmethod
    def has_datum(self, name: str) -> bool:
        """
        Check whether the ``name`` 0D ("blob") datum exists.
        """

    @abstractmethod
    def get_datum(self, name: str) -> Any:
        """
        Access a 0D ("blob") datum.
        """

    @abstractmethod
    def set_datum(self, name: str, datum: Any) -> None:
        """
        Set (override) a ``name`` 0D ("blob") datum.
        """

    @abstractmethod
    def has_axis(self, name: str) -> bool:
        """
        Check whether the ``name`` axis exists.
        """

    @abstractmethod
    def create_axis(self, name: str, index: Collection[str]) -> None:
        """
        Create a new axis with a unique ``name`` and an ``index`` assigning a name for each entry along the index.

        It is an error to try an override an existing axis.
        """

    @abstractmethod
    def get_axis(self, name: str) -> Array1D:
        """
        Get the name of each entry for some ``axis`` (which must exist).
        """

    @abstractmethod
    def has_vector(self, name: str) -> bool:
        """
        Check whether the ``name`` 1D vector data exists.
        """

    @abstractmethod
    def set_vector(self, name: str, vector: Vector) -> None:
        """
        Set (override) a ``name`` 1D data.

        The name should be in the format ``axis_name:name_in_axis``, where the ``axis_name`` identifies the axis (the
        number and semantics of the entries) and the ``name_in_axis`` identifies the specific data for that specific
        axis.
        """

    @abstractmethod
    def get_array1d(self, name: str) -> Array1D:
        """
        Get 1D data (which must exist) as an :py:const:`daf.typing.Array1D` by its ``name``.

        The name should be in the format ``axis_name:name_in_axis``, where the ``axis_name`` identifies the axis (the
        number and semantics of the entries) and the ``name_in_axis`` identifies the specific data for that specific
        axis.
        """

    @abstractmethod
    def get_series(self, name: str) -> Series:
        """
        Get 1D data (which must exist) as an ``pandas.Series`` by its ``name``

        The name should be in the format ``axis_name:name_in_axis``, where the ``axis_name`` identifies the axis (the
        number and semantics of the entries) and the ``name_in_axis`` identifies the specific data for that specific
        axis.

        The ``index`` of the series is the names given to entries by the chosen axis.
        """

    @abstractmethod
    def has_matrix(self, name: str) -> bool:
        """
        Check whether the ``name`` 2D matrix data exists.
        """

    @abstractmethod
    def set_matrix(self, name: str, matrix: Matrix) -> None:
        """
        Set (override) a ``name`` 1D data.

        The name should be in the format ``rows_axis_name,columns_axis_name:name_in_axes``, where the ``rows_axis_name``
        and ``columns_axis_name`` identify the axis (the number and semantics of the entries) and the ``name_in_axes``
        identifies the specific data for this specific pair of axes. It is always possible to switch the order of the
        axis to specify the transposed matrix, that is, each ``name_in_axes`` is unique for each unordered pair of axis
        names.
        """

    @abstractmethod
    def get_grid(self, name: str, *, layout: Optional[AnyMajor] = None) -> Grid:
        """
        Get a :py:const:`daf.typing.Grid` data (which must exist) by its ``name``.

        The name should be in the format ``rows_axis_name,columns_axis_name:name_in_axes``, where the ``rows_axis_name``
        and ``columns_axis_name`` identify the axis (the number and semantics of the entries) and the ``name_in_axes``
        identifies the specific data for this specific pair of axes. It is always possible to switch the order of the
        axis to obtain the transposed matrix, that is, each ``name_in_axes`` is unique for each unordered pair of axis
        names.

        If ``layout`` is specified, the returned grid will be in that layout.
        """

    @abstractmethod
    def get_frame(self, name: str, *, layout: Optional[AnyMajor] = None) -> Frame:
        """
        Get a :py:const:`daf.typing.Frame` data (which must exist) by its ``name``.

        The name should be in the format ``rows_axis_name,columns_axis_name:name_in_axes``, where the ``rows_axis_name``
        and ``columns_axis_name`` identify the axis (the number and semantics of the entries) and the ``name_in_axes``
        identifies the specific data for this specific pair of axes. It is always possible to switch the order of the
        axis to obtain the transposed matrix, that is, each ``name_in_axes`` is unique for each unordered pair of axis
        names.

        The ``index`` and ``columns`` of the frame are the names given to the row and column entries by the chosen axes.

        If ``layout`` is specified, the returned grid will be in that layout.
        """

    @abstractmethod
    def get_table(self, rows_axis: str, column_names: Collection[str]) -> Frame:
        """
        Get a :py:const:`daf.typing.Frame` data (which must exist) by the ``name`` of each column.

        The ``index`` of the frame is the names given to entries by the chosen ``rows_axis``, and the ``columns`` are
        the specified ``column_names``.

        The returned frame is always in column-major layout.
        """
