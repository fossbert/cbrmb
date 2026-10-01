"""Cross-cohort validation: train in one cohort, validate in the others.

Two schemes, which can be combined:

* ``"pairwise"``: for every cohort, a full nested CV within that cohort
  (internal performance) whose outer-split models, plus a final model tuned
  and refit on the whole cohort, predict every *other* cohort.
* ``"loco"`` (leave one cohort out): nested CV on all but one cohort,
  stratified by label x cohort, validated on the held-out cohort.

All preprocessing inside the pipeline (prevalence filter, transformation,
feature selection) is learned on the training cohort(s) only and applied
unchanged to the validation cohorts, so the external scores are honest
transfer estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .ncv import NestedCVResult, _take, nested_cv

__all__ = ["cross_cohort", "CrossCohortResult"]


@dataclass
class CrossCohortResult:
    """Output of :func:`cross_cohort`.

    Attributes
    ----------
    results : dict
        ``{train_label: NestedCVResult}``. Train labels are the cohort name
        (pairwise) or ``"LOCO:<held-out cohort>"``.
    train_cohorts : dict
        ``{train_label: [cohorts used for training]}``.
    cohorts : list
        All cohorts, in order.
    """

    results: dict
    train_cohorts: dict
    cohorts: list

    def scores(self, metric: Optional[str] = None) -> pd.DataFrame:
        """Long table ``train, test, kind, split, metric, value``.

        ``kind`` is ``"internal"`` (held-out folds of the training cohort),
        ``"external"`` (outer-split models on another cohort) or ``"final"``
        (the locked model refit on the whole training cohort).
        """
        rows = []
        for label, res in self.results.items():
            internal = res.scores.melt(id_vars=["split"], value_vars=res.metrics,
                                       var_name="metric", value_name="value")
            if not label.startswith("LOCO:"):
                rows.append(internal.assign(train=label, test=label, kind="internal"))
            if len(res.external_scores):
                ext = res.external_scores[["split", "dataset", "metric", "value"]]
                rows.append(ext.rename(columns={"dataset": "test"}).assign(train=label, kind="external"))
            if res.final is not None and len(res.final["external_scores"]):
                fin = res.final["external_scores"].rename(columns={"dataset": "test"})
                rows.append(fin.assign(train=label, kind="final", split="final"))
        out = pd.concat(rows, ignore_index=True)[["train", "test", "kind", "split", "metric", "value"]]
        if metric is not None:
            out = out[out["metric"] == metric].reset_index(drop=True)
        return out

    def summary(self, metric: Optional[str] = None, alpha: float = 0.05) -> pd.DataFrame:
        """Mean, SD and empirical interval per train x test x kind (x metric)."""
        g = self.scores(metric).groupby(["train", "test", "kind", "metric"], sort=False)["value"]
        return pd.DataFrame({
            "n": g.count(),
            "mean": g.mean(),
            "sd": g.std(),
            "ci_low": g.quantile(alpha / 2),
            "ci_high": g.quantile(1 - alpha / 2),
        }).reset_index()

    def matrix(self, metric: Optional[str] = None, kind: str = "external") -> pd.DataFrame:
        """Train x test matrix of mean scores.

        The diagonal is the internal nested-CV score; off-diagonal cells are
        the mean over outer-split models (``kind="external"``) or the score of
        the locked final model (``kind="final"``).
        """
        if kind not in ("external", "final"):
            raise ValueError("kind must be 'external' or 'final'")
        metric = metric or next(iter(self.results.values())).primary
        s = self.scores(metric)
        s = s[s["kind"].isin(["internal", kind])]
        m = s.groupby(["train", "test"], sort=False)["value"].mean().unstack("test")
        order_rows = [r for r in self.results if r in m.index]
        order_cols = [c for c in self.cohorts if c in m.columns]
        m = m.loc[order_rows, order_cols]
        m.index.name, m.columns.name = "train", "test"
        return m


def cross_cohort(
    estimator,
    param_grid,
    X,
    y,
    cohort,
    *,
    scheme: Union[str, Sequence[str]] = "pairwise",
    groups=None,
    metadata=None,
    refit_final: bool = True,
    min_class_size: int = 5,
    **nested_kwargs,
) -> CrossCohortResult:
    """Train within each cohort (or all but one) and validate on the others.

    Parameters
    ----------
    estimator, param_grid
        As in :func:`nested_cv`; build the whole preprocessing into the pipeline.
    X : DataFrame or array
        All cohorts stacked, with a common feature space (same columns).
    y : array-like
        Binary labels (e.g. PDAC vs Control).
    cohort : array-like
        Cohort label per row.
    scheme : {"pairwise", "loco"} or both as a list
        See the module docstring.
    groups : array-like, optional
        Group labels (e.g. patient IDs), kept together in all CV splits.
    metadata : DataFrame, optional
        Row-aligned; carried into all predictions.
    refit_final : bool
        Fit the locked model per training set and score it on the validation
        cohorts (``kind="final"``).
    min_class_size : int
        Cohorts with fewer samples in either class are not used for training
        (they are still used for validation).
    **nested_kwargs
        Passed to :func:`nested_cv` (``outer``, ``inner``, ``scoring``,
        ``search``, ``n_jobs``, ``random_state``, ...).

    Returns
    -------
    CrossCohortResult
    """
    y = np.asarray(y)
    cohort = np.asarray(cohort)
    groups = None if groups is None else np.asarray(groups)
    md = None if metadata is None else pd.DataFrame(metadata).reset_index(drop=True)
    schemes = [scheme] if isinstance(scheme, str) else list(scheme)
    bad = set(schemes) - {"pairwise", "loco"}
    if bad:
        raise ValueError(f"unknown scheme(s) {bad}")
    if groups is not None:
        shared = pd.DataFrame({"g": groups, "c": cohort}).groupby("g")["c"].nunique()
        if (shared > 1).any():
            raise ValueError("some groups span several cohorts")

    cohorts = list(pd.unique(cohort))

    def subset(mask):
        idx = np.flatnonzero(mask)
        return (_take(X, idx), y[idx], None if groups is None else groups[idx],
                None if md is None else md.iloc[idx])

    def trainable(mask):
        _, counts = np.unique(y[mask], return_counts=True)
        return len(counts) == 2 and counts.min() >= min_class_size

    results, train_cohorts = {}, {}
    if "pairwise" in schemes:
        for c in cohorts:
            mask = cohort == c
            if not trainable(mask):
                continue
            Xc, yc, gc, mc = subset(mask)
            external = {}
            for d in cohorts:
                if d != c:
                    Xd, yd, _, md_d = subset(cohort == d)
                    external[d] = (Xd, yd, md_d)
            results[c] = nested_cv(estimator, param_grid, Xc, yc, groups=gc, metadata=mc,
                                   external=external, refit_final=refit_final, **nested_kwargs)
            train_cohorts[c] = [c]
    if "loco" in schemes:
        for c in cohorts:
            mask = cohort != c
            if not trainable(mask):
                continue
            Xr, yr, gr, mr = subset(mask)
            Xc, yc, _, mc = subset(cohort == c)
            strat = pd.Series(yr).astype(str).to_numpy() + "|" + cohort[mask].astype(str)
            label = f"LOCO:{c}"
            results[label] = nested_cv(estimator, param_grid, Xr, yr, groups=gr, metadata=mr,
                                       stratify=strat, external={c: (Xc, yc, mc)},
                                       refit_final=refit_final, **nested_kwargs)
            train_cohorts[label] = [d for d in cohorts if d != c]
    if not results:
        raise ValueError("no cohort had enough samples of both classes to train on")
    return CrossCohortResult(results=results, train_cohorts=train_cohorts, cohorts=cohorts)
