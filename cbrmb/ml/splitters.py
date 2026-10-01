"""Cross-validation splitters and the ``outer=`` / ``inner=`` shorthand."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.utils import check_random_state

__all__ = ["RepeatedStratifiedGroupKFold", "make_cv"]


class RepeatedStratifiedGroupKFold:
    """:class:`~sklearn.model_selection.StratifiedGroupKFold`, repeated with
    different shuffles. All samples of a group (e.g. a patient with several
    sample types or visits) always land in the same fold."""

    def __init__(self, n_splits=4, n_repeats=25, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y, groups):
        if groups is None:
            raise ValueError("RepeatedStratifiedGroupKFold needs groups")
        rng = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            cv = StratifiedGroupKFold(self.n_splits, shuffle=True,
                                      random_state=rng.randint(np.iinfo(np.int32).max))
            yield from cv.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats

    def __repr__(self):
        return (f"RepeatedStratifiedGroupKFold(n_splits={self.n_splits}, "
                f"n_repeats={self.n_repeats}, random_state={self.random_state})")


def make_cv(spec, *, grouped=False, random_state=None):
    """Turn an ``outer=`` / ``inner=`` spec into a splitter.

    ``spec`` is an int (folds, one repeat), a tuple ``(n_splits, n_repeats)``
    or any object with a ``split`` method, which is returned unchanged. Int and
    tuple specs give repeated *stratified* k-fold, grouped when ``grouped``.
    """
    if hasattr(spec, "split"):
        return spec
    if isinstance(spec, (int, np.integer)):
        spec = (int(spec), 1)
    n_splits, n_repeats = spec
    if grouped:
        return RepeatedStratifiedGroupKFold(n_splits, n_repeats, random_state=random_state)
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


def folds_per_repeat(cv):
    """Folds per repeat of a splitter, or ``None`` if not a repeated k-fold."""
    if isinstance(cv, RepeatedStratifiedGroupKFold):
        return cv.n_splits
    cvargs = getattr(cv, "cvargs", None)
    if cvargs and "n_splits" in cvargs:
        return cvargs["n_splits"]
    return None
