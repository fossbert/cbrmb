"""Filtering helpers for binary clinical covariates."""

from __future__ import annotations

from functools import singledispatch
from typing import Union

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

__all__ = ["bernoulli_var", "filter_bernoulli"]


@singledispatch
def bernoulli_var(thresh, n: int):
    """Variance of a Bernoulli variable at a probability implied by ``thresh``.

    ``thresh`` as ``int``: interpreted as a count out of ``n`` (p = thresh / n).
    ``thresh`` as ``float``: interpreted directly as the probability p.
    """
    raise TypeError("Wrong argument type, please use float or integer.")


@bernoulli_var.register
def _(thresh: int, n: int):
    p = thresh / n
    bvar = p * (1 - p)
    print(
        f"Using threshold p = {max(p, 1 - p):.1%} with variance {bvar:.2f} "
        f"as determined from n = {thresh}"
    )
    return bvar


@bernoulli_var.register
def _(thresh: float, n: int):
    bvar = thresh * (1 - thresh)
    print(f"Using threshold for p at {max(thresh, 1 - thresh):.1%} with variance {bvar:.2f}")
    return bvar


def filter_bernoulli(df: pd.DataFrame, thresh: Union[float, int] = 0.85):
    """Report near-constant binary features by variance threshold.

    Returns the list of dropped feature names (most-constant first), or an empty
    list if every feature clears the threshold implied by ``thresh`` (see
    :func:`bernoulli_var`).
    """
    sel = VarianceThreshold(threshold=bernoulli_var(thresh, len(df)))
    sel.fit(df)

    features_kept = sel.get_feature_names_out(sel.feature_names_in_)
    if len(features_kept) == sel.n_features_in_:
        print("All features made the threshold!")
        return []

    features_dropped = [
        (f, v)
        for f, v in zip(sel.feature_names_in_, sel.variances_)
        if f not in set(features_kept)
    ]
    features_dropped.sort(key=lambda x: x[1])
    print(
        "Dropping the following features: "
        + ", ".join(f"{f}:{v:.2f}" for f, v in features_dropped)
    )
    return [f for f, _ in features_dropped]
