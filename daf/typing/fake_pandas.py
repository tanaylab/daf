"""
Pandas doesn't provide type annotations. We don't try to overcome this here, but we do want to allow for saying "this is
a pandas frame in column-major layout", at least as an option. To do this we need to use ``typing.NewType`` which
requires the annotated class to be known. As a workaround we define fake pandas series and frame classes and use them
for the annotation instead. To shut ``mypy`` up we need to populate them with all the public interface of the real
classes.

.. todo::

    If/when ``pandas`` will provide some form of type annotations, get rid of the `.fake_pandas` module.
"""

# pylint: disable=duplicate-code,cyclic-import

from __future__ import annotations

from typing import Any
from typing import Tuple

# pylint: enable=duplicate-code,cyclic-import

# pylint: disable=unused-argument,missing-function-docstring,no-self-use,invalid-name
# pylint: disable=too-many-public-methods,too-many-lines


__all__ = ["PandasSeries", "PandasFrame"]


class PandasSeries:
    """
    Fake class for ``mypy``.
    """

    array: Any
    at: Any
    attrs: Any
    axes: Any
    cat: Any
    dt: Any
    dtype: Any
    dtypes: Any
    empty: Any
    flags: Any
    hasnans: Any
    iat: Any
    iloc: Any
    index: Any
    is_monotonic: Any
    is_monotonic_decreasing: Any
    is_monotonic_increasing: Any
    is_unique: Any
    loc: Any
    name: Any
    nbytes: Any
    ndim: int
    shape: Tuple[int]
    size: Any
    sparse: Any
    str: Any
    T: Any
    values: Any

    def abs(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def add_prefix(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def add(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def add_suffix(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def aggregate(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def agg(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def align(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def all(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def any(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def append(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def argmax(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def argmin(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def argsort(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def asfreq(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def asof(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def astype(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def at_time(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def autocorr(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def backfill(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def between(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def between_time(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def bfill(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def bool(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def clip(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def combine_first(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def combine(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def compare(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def convert_dtypes(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def copy(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def corr(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def count(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def cov(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def cummax(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def cummin(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def cumprod(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def cumsum(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def describe(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def diff(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def divide(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def divmod(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def div(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def dot(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def drop_duplicates(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def droplevel(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def dropna(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def drop(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def duplicated(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def eq(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def equals(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def ewm(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def expanding(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def explode(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def factorize(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def ffill(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def fillna(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def filter(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def first(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def first_valid_index(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def floordiv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def ge(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def get(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def groupby(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def gt(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def head(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def hist(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def idxmax(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def idxmin(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def infer_objects(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def info(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def interpolate(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def isin(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def isna(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def isnull(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def item(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def items(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def iteritems(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def keys(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def kurtosis(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def kurt(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def last(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def last_valid_index(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def le(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def lt(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mad(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def map(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mask(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def max(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def median(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def memory_usage(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def min(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mode(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mod(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mul(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def multiply(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def ne(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def nlargest(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def notna(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def notnull(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def nsmallest(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def nunique(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pad(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pct_change(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pipe(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pow(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def prod(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def product(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def quantile(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def radd(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rank(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def ravel(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rdivmod(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rdiv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def reindex_like(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def reindex(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rename_axis(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rename(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def reorder_levels(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def repeat(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def replace(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def resample(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def reset_index(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rfloordiv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rmod(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rmul(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rolling(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def round(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rpow(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rsub(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rtruediv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def searchsorted(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sem(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def set_axis(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def set_flags(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def shift(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def skew(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def slice_shift(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sort_index(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sort_values(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def squeeze(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def std(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sub(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def subtract(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def swapaxes(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def swaplevel(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tail(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def take(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_clipboard(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_csv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_dict(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_excel(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_frame(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_hdf(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_json(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_latex(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_list(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tolist(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_markdown(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_numpy(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_period(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_pickle(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_sql(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_string(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_timestamp(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_xarray(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def transform(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def transpose(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def truediv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def truncate(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tshift(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tz_convert(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tz_localize(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def unique(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def unstack(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def update(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def value_counts(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def var(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def view(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def where(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def xs(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __abs__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __add__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __and__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __array_ufunc__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __array_wrap__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __bool__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __contains__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __copy__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __deepcopy__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __delitem__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __divmod__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __finalize__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __float__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __floordiv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __getattr__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __getitem__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __getstate__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __iadd__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __iand__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __ifloordiv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __imod__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __imul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __int__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __invert__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __ior__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __ipow__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __isub__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __iter__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __itruediv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __ixor__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __len__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __long__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __matmul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __mod__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __mul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __neg__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __nonzero__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __or__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __pos__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __pow__(self, *args: Any, **kwargs: Any) -> Any:  # pylint: disable=unexpected-special-method-signature
        ...

    def __radd__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rand__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rdivmod__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rfloordiv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rmatmul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rmod__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rmul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __ror__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __round__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rpow__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rsub__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rtruediv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rxor__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __setitem__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __setstate__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __sub__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __truediv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __xor__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    # pylint: enable=unused-argument,missing-function-docstring,no-self-use,invalid-name


class PandasFrame:  # pylint: disable=too-many-public-methods
    """
    Fake class for ``mypy``.
    """

    ndim: int
    shape: Tuple[int, int]

    a: Any
    at: Any
    attrs: Any
    axes: Any
    b: Any
    columns: Any
    dtypes: Any
    empty: Any
    flags: Any
    iat: Any
    iloc: Any
    index: Any
    loc: Any
    size: Any
    sparse: Any
    style: Any
    T: Any
    values: Any

    # pylint: disable=unused-argument,missing-function-docstring,no-self-use,invalid-name

    def abs(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def add_prefix(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def add(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def add_suffix(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def aggregate(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def agg(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def align(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def all(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def any(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def append(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def applymap(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def asfreq(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def asof(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def assign(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def astype(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def at_time(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def backfill(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def between_time(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def bfill(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def bool(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def boxplot(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def clip(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def combine_first(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def combine(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def compare(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def convert_dtypes(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def copy(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def corr(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def corrwith(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def count(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def cov(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def cummax(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def cummin(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def cumprod(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def cumsum(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def isnull(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def subtract(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def describe(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def diff(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def divide(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def div(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def dot(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def drop_duplicates(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def droplevel(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def dropna(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def drop(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def duplicated(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def eq(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def equals(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def ewm(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def expanding(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def explode(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def ffill(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def fillna(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def filter(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def first(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def first_valid_index(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def floordiv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def from_dict(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def from_records(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def ge(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def get(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def groupby(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def gt(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def head(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def hist(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def idxmax(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def idxmin(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def infer_objects(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def info(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def insert(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def interpolate(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def isin(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def isna(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def items(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def iteritems(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def iterrows(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def itertuples(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def join(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def keys(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def kurtosis(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def kurt(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def last(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def last_valid_index(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def le(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def lookup(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def lt(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mad(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mask(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def max(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def median(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def melt(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def memory_usage(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def merge(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def min(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mode(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mod(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def mul(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def multiply(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def ne(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def nlargest(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def notna(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def notnull(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def nsmallest(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def nunique(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pad(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pct_change(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pipe(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pivot(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pivot_table(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def pow(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def prod(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def product(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def quantile(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def query(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def radd(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rank(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rdiv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def reindex_like(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def reindex(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rename_axis(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rename(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def reorder_levels(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def replace(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def resample(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def reset_index(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rfloordiv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rmod(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rmul(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rolling(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def round(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rpow(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rsub(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def rtruediv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def select_dtypes(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sem(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def set_axis(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def set_flags(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def set_index(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def shift(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def skew(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def slice_shift(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sort_index(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sort_values(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def squeeze(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def stack(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def std(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sub(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def swapaxes(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def swaplevel(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tail(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def take(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_clipboard(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_csv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_dict(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_excel(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_feather(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_gbq(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_hdf(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_html(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_json(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_latex(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_markdown(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_numpy(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_parquet(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_period(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_pickle(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_records(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_sql(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_stata(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_string(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_timestamp(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_xarray(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_xml(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def transform(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def transpose(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def truediv(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def truncate(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tshift(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tz_convert(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tz_localize(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def unstack(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def update(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def value_counts(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def var(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def where(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def xs(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __abs__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __add__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __and__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __array_ufunc__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __array_wrap__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __bool__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __contains__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __copy__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __deepcopy__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __delitem__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __divmod__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __finalize__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __floordiv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __getattr__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __getitem__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __getstate__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __iadd__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __iand__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __ifloordiv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __imod__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __imul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __invert__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __ior__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __ipow__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __isub__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __iter__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __itruediv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __ixor__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __len__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __matmul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __mod__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __mul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __neg__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __nonzero__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __or__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __pos__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __pow__(self, *args: Any, **kwargs: Any) -> Any:  # pylint: disable=unexpected-special-method-signature
        ...

    def __radd__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rand__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rdivmod__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rfloordiv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rmatmul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rmod__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rmul__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __ror__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __round__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rpow__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rsub__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rtruediv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __rxor__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __setitem__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __setstate__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __sub__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __truediv__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __weakref__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __xor__(self, *args: Any, **kwargs: Any) -> Any:
        ...
