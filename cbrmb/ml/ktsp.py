"""k top scoring pairs (k-TSP), a pure-NumPy port of Bioconductor ``switchBox``.

A *top scoring pair* is a feature pair ``(a, b)`` whose ordering ``a > b`` flips
between the two classes. Because only the within-sample ordering of features
matters, k-TSP rules are invariant to any monotone per-sample transformation
(library size, relative abundance, log, CLR, ...). That makes them attractive for
transferring a signature between cohorts with different sequencing depths or
batch effects.

The scoring and pair selection follow ``switchBox::SWAP.Train.KTSP`` (version
1.44) exactly, including the secondary rank-difference tie-breaker, the
optional ``handleTies`` scoring and the ``SWAP.Filter.Wilcoxon`` pre-filter;
selected pairs, scores and tie votes are identical to R (see
``tests/test_ktsp.py``). Two deliberate deviations, both affecting only data
with tied values (zeros in microbiome tables):

* Choosing ``k`` from a range (``KbyTtest``) uses the same comparison rule as
  prediction. switchBox compares its numeric ``tieVote`` against factor labels
  there, so it silently uses ``a >= b`` for every pair during ``k`` selection.
* Feature names describe the comparison that is actually evaluated (switchBox
  labels a ``>`` comparison ``"a>=b"`` and a ``>=`` comparison ``"a>>b"``).

Class orientation: as in switchBox, with classes sorted (``classes_``), a pair
``a>b`` that is true votes for ``classes_[1]``.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["KTSP", "KTSPClassifier", "tsp_scores", "wilcoxon_filter"]

# tie vote codes (switchBox ``tieVote``): 0 = "both", 1 = ties side with
# classes_[0], 2 = ties side with classes_[1]
_TIE_BOTH, _TIE_FIRST, _TIE_SECOND = 0, 1, 2

# target size of the boolean (n_samples, block, n_features) comparison cube
_BLOCK_CELLS = 2e7


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _class_indicator(y, classes):
    """switchBox ``situation``: 1 for classes_[0] (first factor level), else 0."""
    first = np.asarray(y) == classes[0]
    return first, int(first.sum()), int((~first).sum())


def _within_sample_ranks(X):
    """Within-sample feature ranks as switchBox computes them.

    switchBox ranks ``rbind(inputMat, inputMat)``, i.e. every value twice, which
    maps a plain average rank ``r`` to ``2 r - 0.5``. The transform is monotone
    (comparisons unchanged) but doubles the secondary score, so it is kept.
    All values stay exactly representable, which keeps sums exact.
    """
    return 2.0 * rankdata(X, axis=1) - 0.5


def _less_counts(R, first):
    """``LT[c][i, j]`` = #samples of class ``c`` with ``R[:, i] < R[:, j]``.

    ``c = 1`` is classes_[0] (switchBox's class 1), ``c = 0`` the other class.
    ``count(R_i > R_j) = LT[c].T`` so one pass gives all pairwise orderings.
    Samples are sorted by class and laid out contiguously so each class count
    is a reduction over a contiguous slice; ranks are compared as small ints.
    """
    n, m = R.shape
    n1 = int(first.sum())
    order = np.argsort(~first, kind="stable")
    ranks = np.rint(2 * R + 1)                       # 4 * plain rank: exact integers
    dtype = np.int16 if ranks.max() < np.iinfo(np.int16).max else np.int32
    Rt = np.ascontiguousarray(ranks.astype(dtype)[order].T)
    lt1 = np.empty((m, m), dtype=np.int64)
    lt0 = np.empty((m, m), dtype=np.int64)
    block = max(1, int(_BLOCK_CELLS // max(1, n * m)))
    for start in range(0, m, block):
        sl = slice(start, min(start + block, m))
        cmp = (Rt[sl, None, :] < Rt[None, :, :]).view(np.uint8)
        lt1[sl] = cmp[..., :n1].sum(axis=-1, dtype=np.int32)
        lt0[sl] = cmp[..., n1:].sum(axis=-1, dtype=np.int32)
    return lt0, lt1


def tsp_scores(X, y, *, handle_ties=False, classes=None):
    """Signed pairwise TSP scores, as ``switchBox:::calculateSignedScore``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        Binary labels.
    handle_ties : bool
        switchBox ``handleTies``: score each pair under three tie conventions
        and keep the best one (recorded in the returned tie vote).
    classes : sequence of two labels, optional
        Class order; defaults to the sorted unique labels.

    Returns
    -------
    score : ndarray of shape (n_features, n_features)
        ``score[i, j] > 0`` when ``X_i > X_j`` is more frequent in
        ``classes[0]`` than in ``classes[1]``. Includes the secondary score
        (difference in mean rank gap, scaled by 1e-6) that breaks ties.
    tie_vote : ndarray of int, same shape
        switchBox ``tieVote`` codes (all 0 unless ``handle_ties``).
    """
    X = np.asarray(X, dtype=float)
    classes = np.unique(y) if classes is None else np.asarray(classes)
    if len(classes) != 2:
        raise ValueError(f"k-TSP needs exactly two classes, got {len(classes)}")
    first, n1, n0 = _class_indicator(y, classes)
    R = _within_sample_ranks(X)
    lt0, lt1 = _less_counts(R, first)
    gt0, gt1 = lt0.T, lt1.T

    # secondary score: sum over class of (x_j - x_i) = S[j] - S[i], exact
    S1 = R[first].sum(axis=0)
    S0 = R[~first].sum(axis=0)
    p1 = S1[None, :] - S1[:, None]
    p0 = S0[None, :] - S0[:, None]
    secondary = (p0 - p1) / n0 / n1 / 1e6

    if not handle_ties:
        # C: score = -#(x_i>=x_j|0)/n0 + #(x_i>=x_j|1)/n1 + #(x_i<=x_j|0)/n0 - #(x_i<=x_j|1)/n1
        nl0, nl1 = n0 - lt0, n1 - lt1          # x_i >= x_j
        nb0, nb1 = n0 - gt0, n1 - gt1          # x_i <= x_j
        score = (-nl0) / n0 + nl1 / n1 + nb0 / n0 - nb1 / n1
        score = (score + secondary) / 2
        tie_vote = np.zeros(score.shape, dtype=np.int64)
    else:
        eq1 = n1 - lt1 - gt1
        eq0 = n0 - lt0 - gt0
        base = gt1 / n1 - gt0 / n0
        tie = eq1 / n1 - eq0 / n0
        alt = base + tie
        tie_none = np.abs(tie) < 1e-9
        use_alt = ~tie_none & (np.abs(base) < np.abs(alt))
        score = np.where(tie_none, base + 0.5 * tie, np.where(use_alt, alt, base))
        tie_vote = np.where(tie_none, 0, np.where(use_alt, 1, 2)).astype(np.int64)
        score = score + secondary
    np.fill_diagonal(score, 0.0)
    return score, tie_vote


def _top_pairs(score, tie_vote, maxk, disjoint=True):
    """switchBox ``makeTSPTable``: greedy walk down ``|score|``.

    Returns ``(pairs, scores, tie_votes)`` with ``pairs[r] = (a, b)`` oriented so
    that ``X_a > X_b`` votes for classes_[1].
    """
    m = score.shape[0]
    flat = score.ravel(order="F")             # R's column-major score vector
    tv_flat = tie_vote.ravel(order="F")
    absflat = np.abs(flat)
    order = np.argsort(-absflat, kind="stable")
    swap_tie = np.array([0, 2, 1])
    used, set1, set2 = set(), set(), set()
    pairs, scores, ties = [], [], []
    for idx in order:
        i, j = idx % m, idx // m              # row (data1 feature), column
        if i == j:
            continue
        if (i in set1 and j in set2) or (i in set2 and j in set1):
            continue
        if disjoint and (i in used or j in used):
            continue
        s = flat[idx]
        if s > 0:
            a, b, t = j, i, swap_tie[tv_flat[idx]]
        else:
            a, b, t = i, j, tv_flat[idx]
        set1.add(a)
        set2.add(b)
        used.update((a, b))
        pairs.append((a, b))
        scores.append(absflat[idx])
        ties.append(t)
        if len(pairs) == maxk:
            break
    return (
        np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        np.asarray(scores, dtype=float),
        np.asarray(ties, dtype=np.int64),
    )


def _comparisons(X, pairs, tie_votes):
    """Pair indicators as ``SWAP.KTSP.Statistics``: ``>`` unless ties side with
    classes_[0] (code 1), then ``>=``. Ties under code 0 count as 0, as in R."""
    xa = X[:, pairs[:, 0]]
    xb = X[:, pairs[:, 1]]
    ge = tie_votes == _TIE_FIRST
    return np.where(ge[None, :], xa >= xb, xa > xb)


def _choose_k(X, y, classes, pairs, tie_votes, krange):
    """switchBox ``KbyTtest``: k maximising the class separation of the votes."""
    first = np.asarray(y) == classes[0]
    comps = _comparisons(X, pairs, tie_votes).astype(float)
    best_t, best_k = -np.inf, None
    for k in krange:
        kk = min(k, len(pairs))
        stat = comps[:, :kk].sum(axis=1)
        s0, s1 = stat[~first], stat[first]
        t = abs(s0.mean() - s1.mean()) / np.sqrt(s1.var(ddof=1) + s0.var(ddof=1) + 1e-9)
        if abs(best_t - t) > 1e-7 and best_t < t:
            best_t, best_k = t, kk
    return best_k


def wilcoxon_filter(X, y, n_features=100, *, updown=True, classes=None):
    """``switchBox::SWAP.Filter.Wilcoxon``: column indices of retained features.

    Features are ranked within samples, then a rank-sum statistic compares the
    classes per feature. With ``updown`` the ``n_features/2`` most up- and
    down-shifted features are kept (switchBox keeps one extra at the bottom),
    in switchBox's order, which matters for tie-breaking downstream.
    """
    X = np.asarray(X, dtype=float)
    classes = np.unique(y) if classes is None else np.asarray(classes)
    first = np.asarray(y) == classes[0]
    n, m = int(first.sum()), int((~first).sum())
    within = rankdata(X, axis=1)
    across = rankdata(within, axis=0)
    w = (across[first].sum(axis=0) - n * (n + m + 1) / 2) / np.sqrt(n * m * (n + m + 1) / 12)
    p = X.shape[1]
    if updown:
        s = np.argsort(-w, kind="stable")
        half = int(round(n_features / 2))
        up = s[: min(half, p)]
        down = s[max(p - half, 1) - 1:]
        keep = list(dict.fromkeys(np.concatenate([up, down]).tolist()))
    else:
        keep = np.argsort(-np.abs(w), kind="stable")[: min(n_features, p)].tolist()
    return np.asarray(keep, dtype=np.int64)


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
class _KTSPBase(BaseEstimator):
    def __init__(
        self,
        k: Union[int, Sequence[int]] = 10,
        *,
        prefilter: Union[int, None] = None,
        updown: bool = True,
        handle_ties: bool = False,
        disjoint: bool = True,
    ):
        self.k = k
        self.prefilter = prefilter
        self.updown = updown
        self.handle_ties = handle_ties
        self.disjoint = disjoint

    def _fit_pairs(self, X, y):
        X = validate_data(self, X, dtype=float)
        check_classification_targets(y)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"k-TSP needs exactly two classes, got {self.classes_}")
        krange = np.atleast_1d(self.k).astype(int)
        if (krange < 1).any():
            raise ValueError("k must be >= 1")

        cols = np.arange(X.shape[1])
        if self.prefilter is not None:
            cols = wilcoxon_filter(X, y, self.prefilter, updown=self.updown, classes=self.classes_)
        if len(cols) < 4:
            raise ValueError("k-TSP needs at least 4 features after filtering")
        score, tie_vote = tsp_scores(X[:, cols], y, handle_ties=self.handle_ties, classes=self.classes_)
        pairs, scores, ties = _top_pairs(score, tie_vote, int(krange.max()), self.disjoint)
        pairs = cols[pairs]
        k = int(krange[0]) if len(krange) == 1 else _choose_k(X, y, self.classes_, pairs, ties, krange)
        k = min(k, len(pairs))
        self.k_ = k
        self.pairs_ = pairs[:k]
        self.scores_ = scores[:k]
        self.tie_votes_ = ties[:k]
        return X

    def _input_names(self, input_features=None):
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
            if len(names) != self.n_features_in_:
                raise ValueError(f"input_features has {len(names)} names, expected {self.n_features_in_}")
            return names
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            names = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return names

    def _pair_names(self, input_features=None):
        names = self._input_names(input_features)
        ops = np.where(self.tie_votes_ == _TIE_FIRST, ">=", ">")
        return np.array(
            [f"{names[a]}{op}{names[b]}" for (a, b), op in zip(self.pairs_, ops)], dtype=object
        )

    def get_pairs(self, input_features=None):
        """Selected pairs as a DataFrame ``a, b, comparison, score, tie_vote``.

        ``a <comparison> b`` being true votes for ``classes_[1]``. Inside a
        pipeline pass the names of the features entering this step, e.g.
        ``pipe[-1].get_pairs(pipe[:-1].get_feature_names_out())``.
        """
        import pandas as pd

        check_is_fitted(self, "pairs_")
        names = self._input_names(input_features)
        return pd.DataFrame({
            "a": names[self.pairs_[:, 0]],
            "b": names[self.pairs_[:, 1]],
            "comparison": np.where(self.tie_votes_ == _TIE_FIRST, ">=", ">"),
            "score": self.scores_,
            "tie_vote": self.tie_votes_,
        })

    def _votes(self, X):
        check_is_fitted(self, "pairs_")
        X = validate_data(self, X, dtype=float, reset=False)
        return _comparisons(X, self.pairs_, self.tie_votes_)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags


class KTSP(TransformerMixin, _KTSPBase):
    """Replace the feature matrix by k binary top-scoring-pair indicators.

    Use it as a feature generator in front of any classifier, with ``k`` tuned
    like any other hyperparameter, e.g.
    ``Pipeline([("prev", PrevalenceThreshold(0.1)), ("ktsp", KTSP(k=10)),
    ("clf", LogisticRegression())])``.

    Parameters
    ----------
    k : int or sequence of int
        Number of pairs. A sequence is switchBox's ``krange``: pairs are chosen
        for ``max(k)`` and the best ``k`` is picked by a t-like separation
        criterion on the training votes.
    prefilter : int, optional
        Keep only this many features before pairing, via the switchBox Wilcoxon
        filter (``SWAP.Filter.Wilcoxon``, ``featureNo``). ``None`` = no filter
        (switchBox's default is 100). Pair scoring is O(n_features**2).
    updown : bool
        Wilcoxon filter keeps half up-, half down-shifted features.
    handle_ties : bool
        switchBox ``handleTies``. Recommended for zero-inflated count data.
    disjoint : bool
        Each feature appears in at most one pair.

    Attributes
    ----------
    classes_, k_, pairs_ (column indices), scores_, tie_votes_; see
    :meth:`get_pairs` for a readable table.
    """

    def fit(self, X, y):
        self._fit_pairs(X, y)
        return self

    def transform(self, X):
        return self._votes(X).astype(float)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "pairs_")
        return self._pair_names(input_features)


class KTSPClassifier(ClassifierMixin, _KTSPBase):
    """The classic k-TSP classifier: unweighted majority vote of k pairs.

    Parameters are those of :class:`KTSP`. ``predict`` follows
    ``switchBox::SWAP.KTSP.Classify`` (more than half the pairs vote for
    ``classes_[1]``); ``predict_proba`` returns the vote fraction, which is
    what AUROC is computed on.

    ``feature_importances_`` holds the pair scores, and
    ``get_feature_names_out`` the pair names, so nested-CV weight tables work.
    """

    def fit(self, X, y):
        self._fit_pairs(X, y)
        self.feature_importances_ = self.scores_
        return self

    def decision_function(self, X):
        """switchBox k-TSP statistic: #votes for classes_[1] minus k/2."""
        return self._votes(X).sum(axis=1) - self.k_ / 2

    def predict_proba(self, X):
        frac = self._votes(X).mean(axis=1)
        return np.column_stack([1 - frac, frac])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "pairs_")
        return self._pair_names(input_features)
