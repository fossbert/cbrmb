"""Nested cross-validation for any scikit-learn estimator.

The design follows Hermida, Gertz & Ruppin (Nat Commun 2022): an outer,
repeated stratified k-fold (default 4 folds x 25 repeats = 100 model
instances) estimates performance on held-out data; inside each outer training
split a repeated inner CV (default 3 x 5) tunes the hyperparameters of the
*whole* pipeline (filtering, transformation, feature selection, classifier),
and the best setting is refit on the full outer training split. Groups
(e.g. patients with several samples) can be kept together in both loops.

Everything comes back as tidy tables in a :class:`NestedCVResult`: per-split
scores, out-of-fold and external predictions, chosen hyperparameters, feature
weights and the inner grid, plus helpers for model comparison (paired Wilcoxon
over the same splits), feature stability and label permutation tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone, is_classifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    get_scorer,
    jaccard_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .splitters import folds_per_repeat, make_cv

__all__ = ["nested_cv", "NestedCVResult", "permutation_test", "PermutationResult"]

DEFAULT_SCORING = ("roc_auc", "average_precision", "balanced_accuracy")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _take(X, idx):
    return X.iloc[idx] if isinstance(X, (pd.DataFrame, pd.Series)) else X[idx]


def _labels(X):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return np.asarray(X.index)
    return np.arange(len(X))


def _as_metadata(metadata, n):
    if metadata is None:
        return None
    md = pd.DataFrame(metadata).reset_index(drop=True)
    if len(md) != n:
        raise ValueError(f"metadata has {len(md)} rows, expected {n}")
    return md


def _normalise_external(external, X):
    """{name: (X, y[, metadata])} with columns aligned to the training table."""
    out = {}
    for name, val in (external or {}).items():
        if len(val) == 2:
            Xe, ye = val
            me = None
        else:
            Xe, ye, me = val
        if isinstance(X, pd.DataFrame):
            if not isinstance(Xe, pd.DataFrame):
                raise TypeError(f"external[{name!r}] must be a DataFrame like X")
            missing = X.columns.difference(Xe.columns)
            if len(missing):
                raise ValueError(f"external[{name!r}] lacks {len(missing)} training columns, "
                                 f"e.g. {list(missing[:3])}")
            Xe = Xe[X.columns]
        out[name] = (Xe, np.asarray(ye), _as_metadata(me, len(Xe)))
    return out


# binary metrics whose sklearn scorer hard-codes pos_label=1
_POS_LABEL_METRICS = {
    "average_precision": (average_precision_score, ("decision_function", "predict_proba")),
    "f1": (f1_score, "predict"),
    "precision": (precision_score, "predict"),
    "recall": (recall_score, "predict"),
    "jaccard": (jaccard_score, "predict"),
}


def _scorers(scoring, classes):
    """Scorer objects; ``pos_label`` is set to ``classes[1]`` for label-based
    binary metrics so string labels (e.g. "Control"/"PDAC") work."""
    names = [scoring] if isinstance(scoring, str) else list(scoring)
    out = {}
    for n in names:
        if n in _POS_LABEL_METRICS and len(classes) == 2:
            fn, response = _POS_LABEL_METRICS[n]
            out[n] = make_scorer(fn, response_method=response, pos_label=classes[1])
        else:
            out[n] = get_scorer(n)
    return names, out


def _safe_score(scorer, model, X, y):
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(scorer(model, X, y))
    except ValueError:
        return np.nan


def _continuous_score(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return np.full(len(proba), np.nan)
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return d if np.ndim(d) == 1 else np.full(len(d), np.nan)
    return np.full(len(X), np.nan)


def _predictions(model, X, y, metadata, **ids):
    df = pd.DataFrame({
        **ids,
        "sample": _labels(X),
        "y_true": np.asarray(y),
        "y_pred": model.predict(X),
        "y_score": _continuous_score(model, X),
    })
    if metadata is not None:
        df = pd.concat([df, metadata.reset_index(drop=True)], axis=1)
    return df


def _feature_weights(model):
    """(names, weights) of the final estimator, or ``None``.

    Uses ``coef_`` (binary) or ``feature_importances_``; names come from the
    final step itself when it renames features (e.g. :class:`KTSPClassifier`
    pairs), otherwise from the preceding pipeline steps.
    """
    last = model[-1] if isinstance(model, Pipeline) else model
    if hasattr(last, "coef_"):
        w = np.asarray(last.coef_, dtype=float)
        if w.ndim == 2 and w.shape[0] == 1:
            w = w[0]
        if w.ndim != 1:
            return None
    elif hasattr(last, "feature_importances_"):
        w = np.asarray(last.feature_importances_, dtype=float)
    else:
        return None
    names = None
    try:
        if is_classifier(last) and hasattr(last, "get_feature_names_out"):
            upstream = (model[:-1].get_feature_names_out()
                        if isinstance(model, Pipeline) and len(model) > 1 else None)
            names = np.asarray(last.get_feature_names_out(upstream))
        elif isinstance(model, Pipeline) and len(model) > 1:
            names = np.asarray(model[:-1].get_feature_names_out())
        elif hasattr(last, "feature_names_in_"):
            names = np.asarray(last.feature_names_in_)
    except (AttributeError, ValueError, TypeError):
        names = None
    if names is None or len(names) != len(w):
        names = np.array([f"x{i}" for i in range(len(w))], dtype=object)
    return names, w


def _weights_frame(model, split):
    fw = _feature_weights(model)
    if fw is None:
        return None
    names, w = fw
    df = pd.DataFrame({"split": split, "feature": names, "weight": w})
    nz = df["weight"] != 0
    df["rank"] = np.nan
    df.loc[nz, "rank"] = df.loc[nz, "weight"].abs().rank(ascending=False, method="min")
    return df


def _cv_results_frame(search, split):
    cvr = search.cv_results_
    df = pd.DataFrame(list(cvr["params"]))
    df.insert(0, "split", split)
    df["mean_test_score"] = cvr["mean_test_score"]
    df["std_test_score"] = cvr["std_test_score"]
    df["rank_test_score"] = cvr["rank_test_score"]
    return df


def _make_search(estimator, param_grid, *, search, n_iter, primary, inner_splits, n_jobs, seed):
    if not param_grid:
        return None
    common = dict(scoring=primary, cv=inner_splits, refit=True, n_jobs=n_jobs, error_score="raise")
    if search == "grid":
        return GridSearchCV(clone(estimator), param_grid, **common)
    if search == "random":
        return RandomizedSearchCV(clone(estimator), param_grid, n_iter=n_iter, random_state=seed, **common)
    raise ValueError(f"search must be 'grid' or 'random', got {search!r}")


def _fit_tuned(estimator, param_grid, X, y, strat, groups, inner_cv, *, search, n_iter, primary,
               n_jobs, seed):
    """Tune on (X, y) with inner CV, refit. Returns (model, search or None)."""
    inner_splits = list(inner_cv.split(np.zeros(len(y)), strat, groups))
    srch = _make_search(estimator, param_grid, search=search, n_iter=n_iter, primary=primary,
                        inner_splits=inner_splits, n_jobs=n_jobs, seed=seed)
    if srch is None:
        return clone(estimator).fit(X, y), None
    srch.fit(X, y)
    return srch.best_estimator_, srch


def _fit_split(split, repeat, fold, train, test, *, estimator, param_grid, X, y, strat, groups,
               inner, metadata, external, scorers, primary, search, n_iter, inner_n_jobs, seed,
               keep_estimator):
    grp_tr = None if groups is None else groups[train]
    inner_cv = make_cv(inner, grouped=groups is not None, random_state=seed)
    Xtr, ytr = _take(X, train), y[train]
    model, srch = _fit_tuned(estimator, param_grid, Xtr, ytr, strat[train], grp_tr, inner_cv,
                             search=search, n_iter=n_iter, primary=primary,
                             n_jobs=inner_n_jobs, seed=seed)
    Xte, yte = _take(X, test), y[test]
    ids = dict(split=split, repeat=repeat, fold=fold)
    row = {**ids, "n_train": len(train), "n_test": len(test),
           "inner_best_score": srch.best_score_ if srch is not None else np.nan}
    for name, sc in scorers.items():
        row[name] = _safe_score(sc, model, Xte, yte)
    md_te = None if metadata is None else metadata.iloc[test]
    preds = [_predictions(model, Xte, yte, md_te, dataset="test", **ids)]
    ext_rows = []
    for dname, (Xe, ye, me) in external.items():
        preds.append(_predictions(model, Xe, ye, me, dataset=dname, **ids))
        for name, sc in scorers.items():
            ext_rows.append({**ids, "dataset": dname, "metric": name,
                             "value": _safe_score(sc, model, Xe, ye)})
    return {
        "row": row,
        "external": ext_rows,
        "pred": pd.concat(preds, ignore_index=True),
        "params": {**ids, **(srch.best_params_ if srch is not None else {})},
        "weights": _weights_frame(model, split),
        "cv_results": _cv_results_frame(srch, split) if srch is not None else None,
        "test_key": tuple(np.sort(test)),
        "estimator": model if keep_estimator else None,
    }


# --------------------------------------------------------------------------- #
# result
# --------------------------------------------------------------------------- #
@dataclass
class NestedCVResult:
    """Tidy output of :func:`nested_cv`.

    Attributes
    ----------
    scores : DataFrame
        One row per outer split: ``split, repeat, fold, n_train, n_test,
        inner_best_score`` and one column per metric on the held-out fold.
    external_scores : DataFrame
        Long table ``split, repeat, fold, dataset, metric, value`` of every
        outer-split model scored on every external dataset.
    predictions : DataFrame
        ``split, repeat, fold, dataset, sample, y_true, y_pred, y_score`` (+
        metadata columns); ``dataset`` is ``"test"`` for out-of-fold rows.
        ``y_score`` is the probability of ``classes_[1]`` (or the decision
        function).
    best_params : DataFrame
        Hyperparameters chosen by the inner CV, per split.
    weights : DataFrame or None
        ``split, feature, weight, rank`` (rank by |weight| among non-zero).
    cv_results : DataFrame or None
        The inner grid per split (mean/std inner score for every setting).
    final : dict or None
        With ``refit_final=True``: the model tuned and refit on all data
        (``estimator``, ``best_params``, ``best_score``, ``external_scores``,
        ``predictions``, ``weights``), i.e. the locked model to report and
        transfer.
    """

    scores: pd.DataFrame
    external_scores: pd.DataFrame
    predictions: pd.DataFrame
    best_params: pd.DataFrame
    weights: Optional[pd.DataFrame]
    cv_results: Optional[pd.DataFrame]
    metrics: list
    classes_: np.ndarray
    estimators: Optional[list] = None
    final: Optional[dict] = None
    _test_keys: list = field(default_factory=list, repr=False)

    @property
    def primary(self):
        """The metric used for hyperparameter selection."""
        return self.metrics[0]

    def split_scores(self, metric=None):
        """Wide table: one row per split, one column per dataset (``test`` first)."""
        metric = metric or self.primary
        wide = self.scores.set_index("split")[[metric]].rename(columns={metric: "test"})
        if len(self.external_scores):
            ext = (self.external_scores[self.external_scores["metric"] == metric]
                   .pivot(index="split", columns="dataset", values="value"))
            wide = wide.join(ext)
        return wide

    def summary(self, alpha=0.05):
        """Mean, SD, median and empirical (1-alpha) interval per dataset x metric.

        Rows with ``kind == "final"`` hold the single score of the locked final
        model on each external dataset.
        """
        long = self.scores.melt(id_vars=["split"], value_vars=self.metrics,
                                var_name="metric", value_name="value")
        long["dataset"] = "test"
        if len(self.external_scores):
            long = pd.concat([long, self.external_scores[["split", "dataset", "metric", "value"]]])
        g = long.groupby(["dataset", "metric"], sort=False)["value"]
        out = pd.DataFrame({
            "n": g.count(),
            "mean": g.mean(),
            "sd": g.std(),
            "median": g.median(),
            "ci_low": g.quantile(alpha / 2),
            "ci_high": g.quantile(1 - alpha / 2),
        })
        out = out.reset_index().assign(kind="cv")
        if self.final is not None and len(self.final["external_scores"]):
            fin = self.final["external_scores"].rename(columns={"value": "mean"})
            out = pd.concat([out, fin.assign(n=1, kind="final")], ignore_index=True)
        return out[["dataset", "metric", "kind", "n", "mean", "sd", "median", "ci_low", "ci_high"]]

    def compare(self, other: "NestedCVResult", metric=None, dataset="test", alternative="two-sided"):
        """Paired Wilcoxon signed-rank test of this model vs ``other``.

        Both results must come from the same outer splits (same data order,
        ``outer`` and ``random_state``), e.g. a microbiome+clinical model vs a
        clinical-only model.
        """
        metric = metric or self.primary
        if self._test_keys and other._test_keys and self._test_keys != other._test_keys:
            raise ValueError("the two results were not computed on the same outer splits")
        a = self.split_scores(metric)[dataset]
        b = other.split_scores(metric)[dataset]
        both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        diff = both["a"] - both["b"]
        res = stats.wilcoxon(both["a"], both["b"], alternative=alternative)
        return pd.Series({
            "metric": metric, "dataset": dataset, "n": len(both),
            "mean_a": both["a"].mean(), "mean_b": both["b"].mean(),
            "mean_diff": diff.mean(), "median_diff": diff.median(),
            "statistic": res.statistic, "pvalue": res.pvalue,
        })

    def feature_stability(self, top=50, min_freq=0.2, alpha=0.01, adjust="holm"):
        """Hermida et al.'s feature consensus across the outer-split models.

        Per feature: how often it was selected (non-zero weight), how often it
        was among the ``top`` features by |weight|, its mean weight and median
        rank where selected, and a two-sided Wilcoxon signed-rank test that its
        non-zero weights are shifted away from 0 (meaningful for signed
        coefficients, not for always-positive importances). ``stable`` marks
        features in the top ``top`` in at least ``min_freq`` of the models with
        adjusted p <= ``alpha``.
        """
        from statsmodels.stats.multitest import multipletests

        if self.weights is None:
            raise ValueError("the final estimator exposes no coef_ / feature_importances_")
        n_models = self.scores["split"].nunique()
        w = self.weights[self.weights["weight"] != 0]
        rows = []
        for feat, d in w.groupby("feature", sort=False):
            vals = d["weight"].to_numpy()
            p = np.nan
            if len(vals) >= 2:
                try:
                    p = stats.wilcoxon(vals).pvalue
                except ValueError:
                    pass
            rows.append({
                "feature": feat,
                "freq_selected": len(vals) / n_models,
                "freq_top": (d["rank"] <= top).sum() / n_models,
                "mean_weight": vals.mean(),
                "median_rank": d["rank"].median(),
                "n_positive": int((vals > 0).sum()),
                "n_negative": int((vals < 0).sum()),
                "pval": p,
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        ok = out["pval"].notna()
        out["padj"] = np.nan
        if ok.any():
            out.loc[ok, "padj"] = multipletests(out.loc[ok, "pval"], method=adjust)[1]
        out["stable"] = (out["freq_top"] >= min_freq) & (out["padj"] <= alpha)
        return out.sort_values(["stable", "freq_top", "median_rank"],
                               ascending=[False, False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def nested_cv(
    estimator,
    param_grid: Optional[Union[Mapping, Sequence[Mapping]]],
    X,
    y,
    *,
    groups=None,
    stratify=None,
    outer=(4, 25),
    inner=(3, 5),
    scoring: Union[str, Sequence[str]] = DEFAULT_SCORING,
    search: str = "grid",
    n_iter: int = 50,
    external: Optional[Mapping[str, tuple]] = None,
    metadata=None,
    refit_final: bool = False,
    n_jobs: Optional[int] = None,
    inner_n_jobs: Optional[int] = None,
    random_state: Optional[int] = 0,
    keep_estimators: bool = False,
    verbose: int = 0,
) -> NestedCVResult:
    """Nested cross-validation of an estimator / pipeline.

    Parameters
    ----------
    estimator : estimator or Pipeline
        Put every data-dependent step (prevalence filter, transformation,
        feature selection, scaling) into the pipeline so it is learned on the
        training split only. Set ``random_state`` inside the estimator for
        reproducible random forests etc.
    param_grid : dict, list of dicts or None
        Pipeline parameter grid (``"step__param"``). ``None`` or ``{}`` fits
        the estimator as is (no inner CV), e.g. for a ``DummyClassifier``
        baseline.
    X : DataFrame or array of shape (n_samples, n_features)
        A DataFrame keeps sample IDs in the predictions and feature names in
        the weights, and allows column-name selection in a ColumnTransformer.
    y : array-like of shape (n_samples,)
        Binary labels; ``classes_[1]`` (the larger label) is the positive class
        for AUROC / average precision.
    groups : array-like, optional
        Group labels (e.g. patient IDs). If given, outer and inner splits keep
        groups together (:class:`RepeatedStratifiedGroupKFold`).
    stratify : array-like, optional
        Labels to stratify on instead of ``y`` (e.g. ``y`` x cohort when
        pooling cohorts). Scoring still uses ``y``.
    outer, inner : int, (n_splits, n_repeats) or CV splitter
        Defaults follow Hermida et al.: outer 4-fold x 25 repeats, inner
        3-fold x 5 repeats. Pass e.g.
        ``StratifiedShuffleSplit(100, test_size=0.25)`` for repeated hold-out.
    scoring : str or list of str
        sklearn scorer names. The first one selects hyperparameters; all are
        reported on test and external data.
    search : {"grid", "random"}
        ``GridSearchCV`` or ``RandomizedSearchCV`` (``n_iter`` settings).
    external : dict, optional
        ``{name: (X, y)}`` or ``{name: (X, y, metadata)}``: independent
        datasets every outer-split model (and the final model) predicts.
        DataFrame columns are aligned to ``X`` by name.
    metadata : DataFrame, optional
        Row-aligned with ``X``; its columns are carried into ``predictions``
        (e.g. diagnostic certainty), not used for fitting.
    refit_final : bool
        Also tune on all of ``X`` (inner CV) and refit: the locked model,
        scored on ``external``.
    n_jobs : int, optional
        Parallel outer splits (joblib).
    inner_n_jobs : int, optional
        Parallelism inside each grid search.
    random_state : int or None
        Seeds the outer splits and, derived from it, inner splits and
        randomized searches.
    keep_estimators : bool
        Keep the fitted model of every outer split (memory!).

    Returns
    -------
    NestedCVResult
    """
    y = np.asarray(y)
    metric_names, scorers = _scorers(scoring, np.unique(y))
    primary = scorers[metric_names[0]]
    n = len(y)
    groups = None if groups is None else np.asarray(groups)
    strat = y if stratify is None else np.asarray(stratify)
    md = _as_metadata(metadata, n)
    ext = _normalise_external(external, X)

    rng = np.random.RandomState(random_state)
    outer_cv = make_cv(outer, grouped=groups is not None,
                       random_state=rng.randint(np.iinfo(np.int32).max))
    splits = list(outer_cv.split(np.zeros(n), strat, groups))
    seeds = rng.randint(np.iinfo(np.int32).max, size=len(splits) + 1)
    per_rep = folds_per_repeat(outer_cv)

    common = dict(estimator=estimator, param_grid=param_grid, X=X, y=y, strat=strat,
                  groups=groups, inner=inner, metadata=md, external=ext, scorers=scorers,
                  primary=primary, search=search, n_iter=n_iter, inner_n_jobs=inner_n_jobs,
                  keep_estimator=keep_estimators)
    jobs = (
        delayed(_fit_split)(
            s,
            s // per_rep if per_rep else 0,
            s % per_rep if per_rep else s,
            tr, te, seed=int(seeds[s]), **common,
        )
        for s, (tr, te) in enumerate(splits)
    )
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)

    weights = [o["weights"] for o in out if o["weights"] is not None]
    cvr = [o["cv_results"] for o in out if o["cv_results"] is not None]
    ext_rows = [r for o in out for r in o["external"]]
    result = NestedCVResult(
        scores=pd.DataFrame([o["row"] for o in out]),
        external_scores=pd.DataFrame(ext_rows, columns=["split", "repeat", "fold", "dataset", "metric", "value"]),
        predictions=pd.concat([o["pred"] for o in out], ignore_index=True),
        best_params=pd.DataFrame([o["params"] for o in out]),
        weights=pd.concat(weights, ignore_index=True) if weights else None,
        cv_results=pd.concat(cvr, ignore_index=True) if cvr else None,
        metrics=metric_names,
        classes_=np.unique(y),
        estimators=[o["estimator"] for o in out] if keep_estimators else None,
        _test_keys=[o["test_key"] for o in out],
    )

    if refit_final:
        inner_cv = make_cv(inner, grouped=groups is not None, random_state=int(seeds[-1]))
        model, srch = _fit_tuned(estimator, param_grid, X, y, strat, groups, inner_cv,
                                 search=search, n_iter=n_iter, primary=primary,
                                 n_jobs=inner_n_jobs if inner_n_jobs is not None else n_jobs,
                                 seed=int(seeds[-1]))
        fin_rows, fin_preds = [], []
        for dname, (Xe, ye, me) in ext.items():
            fin_preds.append(_predictions(model, Xe, ye, me, dataset=dname))
            for name, sc in scorers.items():
                fin_rows.append({"dataset": dname, "metric": name, "value": _safe_score(sc, model, Xe, ye)})
        result.final = {
            "estimator": model,
            "best_params": srch.best_params_ if srch is not None else {},
            "best_score": srch.best_score_ if srch is not None else np.nan,
            "cv_results": _cv_results_frame(srch, "final") if srch is not None else None,
            "external_scores": pd.DataFrame(fin_rows, columns=["dataset", "metric", "value"]),
            "predictions": pd.concat(fin_preds, ignore_index=True) if fin_preds else pd.DataFrame(),
            "weights": _weights_frame(model, "final"),
        }
    return result


# --------------------------------------------------------------------------- #
# permutation test
# --------------------------------------------------------------------------- #
@dataclass
class PermutationResult:
    """Observed mean score, the permutation null and a one-sided p-value."""

    metric: str
    observed: float
    null: np.ndarray
    pvalue: float

    def __repr__(self):
        return (f"PermutationResult(metric={self.metric!r}, observed={self.observed:.3f}, "
                f"null_mean={np.nanmean(self.null):.3f}, n={len(self.null)}, pvalue={self.pvalue:.4g})")


def _permute(y, groups, rng):
    if groups is None:
        return rng.permutation(y)
    df = pd.DataFrame({"g": groups, "y": y})
    per_group = df.groupby("g", sort=False)["y"].agg(lambda s: s.iloc[0])
    if (df.groupby("g")["y"].nunique() > 1).any():
        raise ValueError("labels vary within groups; cannot permute at group level")
    shuffled = pd.Series(rng.permutation(per_group.to_numpy()), index=per_group.index)
    return shuffled.loc[groups].to_numpy()


def permutation_test(
    estimator,
    param_grid,
    X,
    y,
    *,
    n_permutations: int = 100,
    metric: Optional[str] = None,
    observed: Optional[NestedCVResult] = None,
    groups=None,
    random_state: Optional[int] = 0,
    n_jobs: Optional[int] = None,
    **nested_kwargs: Any,
) -> PermutationResult:
    """Label permutation test of the whole nested CV procedure.

    The labels are shuffled (at group level if ``groups`` is given) and the
    complete nested CV is rerun ``n_permutations`` times; the p-value is
    ``(1 + #{null >= observed}) / (1 + n_permutations)`` on the mean test score.
    Expensive: each permutation costs as much as the original run. Pass an
    existing ``observed`` result to avoid recomputing it.
    """
    y = np.asarray(y)
    groups = None if groups is None else np.asarray(groups)
    nested_kwargs = {**nested_kwargs, "groups": groups}
    if observed is None:
        observed = nested_cv(estimator, param_grid, X, y, random_state=random_state,
                             n_jobs=n_jobs, **nested_kwargs)
    metric = metric or observed.primary
    obs = float(observed.scores[metric].mean())

    rng = np.random.RandomState(random_state)
    perms = [_permute(y, groups, rng) for _ in range(n_permutations)]
    seeds = rng.randint(np.iinfo(np.int32).max, size=n_permutations)

    def one(yp, seed):
        res = nested_cv(estimator, param_grid, X, yp, random_state=int(seed), n_jobs=1,
                        **nested_kwargs)
        return float(res.scores[metric].mean())

    null = np.asarray(Parallel(n_jobs=n_jobs)(delayed(one)(yp, s) for yp, s in zip(perms, seeds)))
    p = (1 + np.sum(null >= obs)) / (1 + len(null))
    return PermutationResult(metric=metric, observed=obs, null=null, pvalue=float(p))
