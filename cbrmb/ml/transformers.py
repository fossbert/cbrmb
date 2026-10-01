"""scikit-learn transformers for compositional / count feature tables.

All of them learn nothing from ``y`` and are safe inside nested CV pipelines:
whatever they learn (e.g. which features are prevalent) is fit on the training
split only.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["PrevalenceThreshold", "RelativeAbundance", "CLR"]


class PrevalenceThreshold(SelectorMixin, BaseEstimator):
    """Keep features present (value > 0) in enough training samples.

    The pipeline counterpart of :func:`cbrmb.filter_prevalence`, with the same
    ``>=`` rule.

    Parameters
    ----------
    threshold : float or int
        A float is the minimum fraction of samples, an int the minimum number
        of samples with a non-zero value.
    """

    def __init__(self, threshold: Union[int, float] = 0.1):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = validate_data(self, X, accept_sparse=("csr", "csc"), dtype=float,
                          ensure_all_finite="allow-nan")
        present = np.asarray(X > 0)
        if isinstance(self.threshold, (int, np.integer)) and not isinstance(self.threshold, bool):
            self.prevalences_ = present.sum(axis=0).astype(float)
        elif isinstance(self.threshold, (float, np.floating)):
            self.prevalences_ = present.mean(axis=0)
        else:
            raise ValueError(f"threshold must be an int or a float, got {self.threshold!r}")
        if not self._get_support_mask().any():
            raise ValueError(f"No feature reaches the prevalence threshold {self.threshold}")
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "prevalences_")
        return self.prevalences_ >= self.threshold

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = True
        return tags


class RelativeAbundance(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Scale every sample to sum to ``scale`` (closure). Stateless.

    Note that closing *after* a feature filter re-normalises over the kept
    features only.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def fit(self, X, y=None):
        validate_data(self, X, dtype=float)
        return self

    def transform(self, X):
        X = validate_data(self, X, dtype=float, reset=False)
        tot = X.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(tot > 0, X / tot, 0.0)
        return out * self.scale


class CLR(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Centred log-ratio transform with a pseudocount. Stateless.

    ``clr(x)_i = log(x_i + pseudocount) - mean_j log(x_j + pseudocount)``
    """

    def __init__(self, pseudocount: float = 1.0):
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        validate_data(self, X, dtype=float)
        return self

    def transform(self, X):
        X = validate_data(self, X, dtype=float, reset=False)
        lx = np.log(X + self.pseudocount)
        return lx - lx.mean(axis=1, keepdims=True)
