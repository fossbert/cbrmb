"""Survival endpoints (OS, PFS/EFS): what does the microbiome add to a clinical model?

Designed for small cohorts, where an outcome-trained high-dimensional model is
not estimable. Instead a few pre-specified microbiome measures (alpha diversity,
ordination axes, single taxa) are added one at a time to a small, pre-specified
clinical Cox model:

* :func:`cox_added_value` -- base vs base + feature: hazard ratio, likelihood
  ratio test, gain in Harrell's C, and the gain corrected for overfitting by
  Harrell's bootstrap optimism (the algorithm of ``rms::validate``).
* :func:`cox_screen` -- one adjusted Cox model per feature with BH-FDR, the
  survival counterpart of :func:`cbrmb.mwu_test` (exploratory).
* :func:`outcome_free_score` -- compress many clinical covariates into one
  score without looking at the outcome (PCA data reduction), so a richer
  clinical baseline costs a single degree of freedom.

Cox models are fit with ``statsmodels`` ``PHReg`` using Efron ties, which
reproduces R ``survival::coxph`` (coefficients, standard errors, log
partial likelihood). The concordance index follows
``survival::concordance(..., reverse=TRUE)``: pairs with tied times are not
comparable unless only the earlier-listed one is an event, risk ties count
one half, and with ``strata`` only pairs within a stratum are compared.

Input layout follows the rest of cbrmb: ``features`` is a wide table (samples
in rows), ``clinical`` holds time, event, base covariates and strata; both are
aligned on their index. Rows with a missing value in any variable a model uses
are dropped for that model, so every comparison is made on identical rows.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg

from .pvalues import fdr

__all__ = [
    "cox_added_value",
    "cox_screen",
    "outcome_free_score",
    "concordance_index",
    "AddedValueResult",
    "ClinicalScore",
]


# --------------------------------------------------------------------------- #
# core: Cox fit and concordance
# --------------------------------------------------------------------------- #
@dataclass
class _CoxFit:
    params: np.ndarray
    bse: np.ndarray
    llf: float
    lp: np.ndarray          # linear predictor on the fitted rows


def _null_llf(time, event, strata):
    """Log partial likelihood of the model without covariates."""
    model = PHReg(time, np.zeros((len(time), 1)), status=event, strata=strata, ties="efron")
    return float(model.loglike(np.zeros(1)))


def _fit_cox(time, event, X, strata=None):
    """Efron Cox fit; ``X`` may have zero columns (null model)."""
    if X.shape[1] == 0:
        return _CoxFit(np.zeros(0), np.zeros(0), _null_llf(time, event, strata), np.zeros(len(time)))
    model = PHReg(time, X, status=event, strata=strata, ties="efron")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp=False)
    params = np.asarray(res.params, dtype=float)
    return _CoxFit(params, np.asarray(res.bse, dtype=float), float(res.llf), X @ params)


def concordance_index(time, event, risk, strata=None):
    """Harrell's C for a risk score (higher = earlier event).

    Matches ``survival::concordance(Surv(time, event) ~ risk, reverse=TRUE)``:
    a pair is comparable when the shorter time is an event (an event and a
    censoring at the same time count as comparable, two events at the same
    time do not); tied risks count 1/2. With ``strata`` only pairs within the
    same stratum are used.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    groups = np.zeros(len(time)) if strata is None else np.asarray(strata)
    conc = disc = tied = 0.0
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        t, e, r = time[idx], event[idx], risk[idx]
        for i in np.flatnonzero(e):
            later = (t > t[i]) | ((t == t[i]) & ~e)
            rj = r[later]
            conc += np.sum(r[i] > rj)
            disc += np.sum(r[i] < rj)
            tied += np.sum(r[i] == rj)
    n = conc + disc + tied
    return np.nan if n == 0 else (conc + 0.5 * tied) / n


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #
def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _design(clinical, cols):
    """Numeric design matrix for ``cols``; categoricals become dummies."""
    if not cols:
        return pd.DataFrame(index=clinical.index)
    X = clinical[cols]
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    if cat:
        X = pd.get_dummies(X, columns=cat, drop_first=True, dtype=float)
    return X.astype(float)


def _align(features, clinical):
    features = pd.DataFrame(features)
    for name, d in (("features", features), ("clinical", clinical)):
        dup = d.index[d.index.duplicated()]
        if len(dup):
            raise ValueError(f"{name} has duplicate index labels (one row per patient expected), "
                             f"e.g. {list(dup.unique()[:3])}")
    common = features.index.intersection(clinical.index)
    if len(common) == 0:
        raise ValueError("features and clinical share no index labels")
    if len(common) < len(features.index) or len(common) < len(clinical.index):
        warnings.warn(f"using the {len(common)} samples present in both features and clinical",
                      stacklevel=3)
    return features.loc[common], clinical.loc[common]


def _complete(time, event, X, strata, extra=None):
    ok = time.notna() & event.notna() & X.notna().all(axis=1)
    if strata is not None:
        ok &= strata.notna()
    if extra is not None:
        ok &= extra.notna()
    return ok.to_numpy()


def _zscore(x):
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if sd > 0 else x * np.nan


# --------------------------------------------------------------------------- #
# cox_screen
# --------------------------------------------------------------------------- #
def _one_feature(time, event, Xb, strata, f, standardize):
    ok = _complete(time, event, Xb, strata, f)
    t = time.to_numpy(float)[ok]
    e = event.to_numpy(float)[ok].astype(int)
    s = None if strata is None else strata.to_numpy()[ok]
    xb = Xb.to_numpy(float)[ok]
    xf = f.to_numpy(float)[ok]
    if standardize:
        xf = _zscore(pd.Series(xf)).to_numpy()
    out = {"n": int(ok.sum()), "events": int(e.sum())}
    if out["events"] == 0 or not np.isfinite(xf).all() or np.std(xf) == 0:
        return out, None
    base = _fit_cox(t, e, xb, s)
    full = _fit_cox(t, e, np.column_stack([xb, xf]), s)
    beta, se = full.params[-1], full.bse[-1]
    lr = 2 * (full.llf - base.llf)
    out.update({
        "coef": beta, "se": se,
        "hr": np.exp(beta), "hr_low": np.exp(beta - 1.959964 * se), "hr_high": np.exp(beta + 1.959964 * se),
        "pval_wald": 2 * stats.norm.sf(abs(beta / se)),
        "lr_stat": lr, "pval": stats.chi2.sf(max(lr, 0.0), 1),
        "epv": e.sum() / (xb.shape[1] + 1),
    })
    return out, (t, e, s, xb, xf, base, full)


def cox_screen(
    features,
    clinical: pd.DataFrame,
    *,
    time: str,
    event: str,
    adjust: Union[str, Sequence[str], None] = None,
    strata: Optional[str] = None,
    standardize: bool = True,
) -> pd.DataFrame:
    """One Cox model per feature, adjusted for clinical covariates (exploratory).

    Parameters
    ----------
    features : DataFrame
        Samples x features (e.g. CLR abundances, diversity indices).
    clinical : DataFrame
        Must contain ``time``, ``event`` and the ``adjust`` / ``strata``
        columns; aligned to ``features`` by index.
    time, event : str
        Column names; ``event`` is 1 for the event, 0 for censored.
    adjust : str or list of str, optional
        Covariates in every model. Non-numeric columns become dummies.
    strata : str, optional
        Stratification variable (separate baseline hazard per level, no
        coefficient).
    standardize : bool
        Scale each feature to mean 0 / SD 1 so ``hr`` is per standard deviation.

    Returns
    -------
    DataFrame indexed by feature: ``n, events, coef, se, hr, hr_low, hr_high,
    pval_wald, lr_stat, pval`` (likelihood ratio test, 1 df), ``fdr`` (BH on
    ``pval``) and ``epv`` (events per estimated covariate incl. the feature),
    sorted by ``pval``. Features that could not be fit are dropped.
    """
    features, clinical = _align(features, clinical)
    Xb = _design(clinical, _as_list(adjust))
    t, e = clinical[time], clinical[event]
    s = None if strata is None else clinical[strata]
    rows = {}
    for col in features.columns:
        out, _ = _one_feature(t, e, Xb, s, features[col], standardize)
        if "pval" in out:
            rows[col] = out
    res = pd.DataFrame.from_dict(rows, orient="index")
    if res.empty:
        return res
    res["fdr"] = fdr(res["pval"].to_numpy())
    res = res[["n", "events", "coef", "se", "hr", "hr_low", "hr_high", "pval_wald", "lr_stat",
               "pval", "fdr", "epv"]]
    res.index.name = "feature"
    return res.sort_values("pval")


# --------------------------------------------------------------------------- #
# cox_added_value
# --------------------------------------------------------------------------- #
@dataclass
class AddedValueResult:
    """Output of :func:`cox_added_value`.

    Attributes
    ----------
    table : DataFrame
        One row per feature: ``n, events, epv``; the feature's ``hr`` (per SD
        if standardized) with 95 % CI and Wald p; the likelihood ratio test
        base vs base + feature (``lr_stat``, ``pval``, ``fdr``); apparent
        concordance ``c_base``, ``c_full``, ``delta_c``; and with bootstrap
        ``c_base_corr``, ``c_full_corr``, ``delta_c_corr`` (optimism-corrected)
        plus ``delta_c_optimism`` and ``n_boot_ok``.
    base : DataFrame
        The base model on all complete rows (coef, se, hr, CI, p) for reporting.
    base_c : float
        Apparent C of the base model on those rows.
    """

    table: pd.DataFrame
    base: pd.DataFrame
    base_c: float


def _boot_optimism(t, e, s, xb, xf, groups, n_boot, rng):
    """Harrell's optimism for C of base, full and their difference."""
    n = len(t)
    full_X = np.column_stack([xb, xf])
    units = np.arange(n) if groups is None else groups
    uniq = pd.unique(units)
    opt = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        if groups is None:
            idx = pick
        else:
            idx = np.concatenate([np.flatnonzero(units == uniq[k]) for k in pick])
        tb, eb = t[idx], e[idx]
        if eb.sum() < 2:
            continue
        sb = None if s is None else s[idx]
        try:
            fb_base = _fit_cox(tb, eb, xb[idx], sb)
            fb_full = _fit_cox(tb, eb, full_X[idx], sb)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not (np.isfinite(fb_base.params).all() and np.isfinite(fb_full.params).all()):
            continue
        cb_app = concordance_index(tb, eb, fb_base.lp, sb)
        cf_app = concordance_index(tb, eb, fb_full.lp, sb)
        cb_test = concordance_index(t, e, xb @ fb_base.params, s)
        cf_test = concordance_index(t, e, full_X @ fb_full.params, s)
        opt.append((cb_app - cb_test, cf_app - cf_test))
    if not opt:
        return np.nan, np.nan, np.nan, 0
    opt = np.asarray(opt)
    ob, of = np.nanmean(opt[:, 0]), np.nanmean(opt[:, 1])
    return ob, of, np.nanmean(opt[:, 1] - opt[:, 0]), len(opt)


def cox_added_value(
    features,
    clinical: pd.DataFrame,
    *,
    time: str,
    event: str,
    base: Union[str, Sequence[str], None] = None,
    strata: Optional[str] = None,
    standardize: bool = True,
    n_boot: int = 500,
    groups: Optional[str] = None,
    min_epv: float = 10,
    random_state: Optional[int] = 0,
) -> AddedValueResult:
    """Does a feature add prognostic information to a clinical Cox model?

    For each column of ``features`` separately, the base model
    (``base`` covariates, optionally ``strata``) is compared with base +
    feature on the same rows:

    * **Hazard ratio** of the feature in the full model (per SD with
      ``standardize``), 95 % Wald CI.
    * **Likelihood ratio test** (1 df) -- the primary test of added value.
    * **Concordance** (Harrell's C) of both models and the gain ``delta_c``.
      Apparent C is optimistic when it is computed on the data the model was
      fit on; with ``n_boot > 0`` Harrell's bootstrap optimism correction is
      applied (as ``rms::validate``): refit both models on each bootstrap
      sample, take C on the bootstrap sample minus C on the original data,
      average, subtract. ``delta_c_corr`` is the overfitting-corrected gain.

    Parameters
    ----------
    features : DataFrame
        Samples x candidate measures, e.g. Shannon, PCoA axes, a few taxa.
        Keep it to a handful of *pre-specified* measures; for many features
        use :func:`cox_screen`.
    clinical : DataFrame
        ``time``, ``event``, ``base`` and ``strata`` columns, aligned by index.
    base : str or list of str, optional
        Clinical covariates of the base model. ``None`` compares against the
        null model (C = 0.5). An outcome-free composite from
        :func:`outcome_free_score` costs one degree of freedom.
    strata : str, optional
        Stratification variable (e.g. treatment regimen); concordance is then
        computed within strata.
    n_boot : int
        Bootstrap replicates for the optimism correction (0 = skip).
    groups : str, optional
        Column in ``clinical`` with a patient ID, if a patient has several
        rows: the bootstrap then resamples patients.
    min_epv : float
        Warn when events per estimated coefficient of the full model fall
        below this (rule of thumb 10).

    Returns
    -------
    AddedValueResult
    """
    features, clinical = _align(features, clinical)
    base_cols = _as_list(base)
    Xb = _design(clinical, base_cols)
    t_all, e_all = clinical[time], clinical[event]
    s_all = None if strata is None else clinical[strata]
    g_all = None if groups is None else clinical[groups].to_numpy()
    rng = np.random.default_rng(random_state)

    # base model on all rows complete for the base, for reporting
    ok = _complete(t_all, e_all, Xb, s_all)
    s0 = None if s_all is None else s_all.to_numpy()[ok]
    fit0 = _fit_cox(t_all.to_numpy(float)[ok], e_all.to_numpy(float)[ok].astype(int),
                    Xb.to_numpy(float)[ok], s0)
    base_tab = pd.DataFrame({"coef": fit0.params, "se": fit0.bse}, index=Xb.columns)
    base_tab["hr"] = np.exp(base_tab["coef"])
    base_tab["hr_low"] = np.exp(base_tab["coef"] - 1.959964 * base_tab["se"])
    base_tab["hr_high"] = np.exp(base_tab["coef"] + 1.959964 * base_tab["se"])
    base_tab["pval"] = 2 * stats.norm.sf(np.abs(base_tab["coef"] / base_tab["se"]))
    base_c = concordance_index(t_all.to_numpy(float)[ok], e_all.to_numpy(float)[ok], fit0.lp, s0)

    rows = {}
    low_epv = []
    for col in features.columns:
        out, fitted = _one_feature(t_all, e_all, Xb, s_all, features[col], standardize)
        if fitted is None:
            continue
        t, e, s, xb, xf, fb, ff = fitted
        out["c_base"] = concordance_index(t, e, fb.lp, s) if xb.shape[1] else 0.5
        out["c_full"] = concordance_index(t, e, ff.lp, s)
        out["delta_c"] = out["c_full"] - out["c_base"]
        if n_boot:
            okf = _complete(t_all, e_all, Xb, s_all, features[col])
            g = None if g_all is None else g_all[okf]
            ob, of, od, nb = _boot_optimism(t, e, s, xb, xf, g, n_boot, rng)
            out.update({"c_base_corr": out["c_base"] - ob, "c_full_corr": out["c_full"] - of,
                        "delta_c_corr": out["delta_c"] - od, "delta_c_optimism": od, "n_boot_ok": nb})
        if out["epv"] < min_epv:
            low_epv.append(col)
        rows[col] = out
    table = pd.DataFrame.from_dict(rows, orient="index")
    if not table.empty:
        table["fdr"] = fdr(table["pval"].to_numpy())
        order = ["n", "events", "epv", "hr", "hr_low", "hr_high", "pval_wald", "lr_stat", "pval", "fdr",
                 "c_base", "c_full", "delta_c", "c_base_corr", "c_full_corr", "delta_c_corr",
                 "delta_c_optimism", "n_boot_ok", "coef", "se"]
        table = table[[c for c in order if c in table.columns]]
        table.index.name = "feature"
    if low_epv:
        warnings.warn(f"{len(low_epv)} model(s) have fewer than {min_epv} events per coefficient "
                      f"(e.g. {low_epv[0]!r}: {rows[low_epv[0]]['epv']:.1f}); estimates are unstable",
                      stacklevel=2)
    return AddedValueResult(table=table, base=base_tab, base_c=base_c)


# --------------------------------------------------------------------------- #
# outcome_free_score
# --------------------------------------------------------------------------- #
@dataclass
class ClinicalScore:
    """Output of :func:`outcome_free_score`.

    Attributes
    ----------
    score : Series
        First principal component (mean 0, SD 1), one value per row.
    loadings : Series
        Weight of each (log-transformed, standardized) variable.
    explained_variance : float
        Fraction of the total variance the component carries.
    n_imputed : Series
        Number of imputed values per variable.
    """

    score: pd.Series
    loadings: pd.Series
    explained_variance: float
    n_imputed: pd.Series


def outcome_free_score(
    clinical: pd.DataFrame,
    cols: Sequence[str],
    *,
    log: Sequence[str] = (),
    orient: Optional[str] = None,
    impute: str = "median",
    name: str = "clinical_score",
) -> ClinicalScore:
    """Summarise several clinical covariates in one score, blind to the outcome.

    Harrell's data reduction: because the outcome is never used, the score
    can enter a Cox model as a single covariate without inflating the
    degrees of freedom, and it need not be recomputed inside a bootstrap.
    Steps: optional ``log`` of skewed variables (``log1p``), imputation of
    missing values, standardization, first principal component.

    Parameters
    ----------
    clinical : DataFrame
    cols : list of str
        Numeric variables, e.g. the COMM-PACT prognostic set (albumin,
        bilirubin, CA 19-9, CRP, LDH, NLR, ECOG, number of metastatic sites).
        Binary 0/1 variables are fine.
    log : list of str
        Subset of ``cols`` to transform with ``log1p`` first (labs spanning
        orders of magnitude: CA 19-9, CRP, LDH, NLR, bilirubin).
    orient : str, optional
        Variable that should load positively (e.g. ``"ca19_9"``), so that a
        higher score means worse prognosis. The PCA sign is otherwise arbitrary.
    impute : {"median", "knn"}
        Median per variable, or k-nearest-neighbour (k = 5) on the
        standardized variables.
    name : str
        Name of the returned score Series.

    Returns
    -------
    ClinicalScore
    """
    from sklearn.decomposition import PCA

    cols = list(cols)
    d = clinical[cols].astype(float).copy()
    for c in log:
        if c not in cols:
            raise ValueError(f"log column {c!r} not in cols")
        if (d[c] < 0).any():
            raise ValueError(f"log column {c!r} has negative values")
        d[c] = np.log1p(d[c])
    n_imp = d.isna().sum()
    mu, sd = d.mean(), d.std(ddof=1)
    if (sd == 0).any():
        raise ValueError(f"constant column(s): {list(sd.index[sd == 0])}")
    z = (d - mu) / sd
    if impute == "median":
        z = z.fillna(z.median())
    elif impute == "knn":
        from sklearn.impute import KNNImputer

        z = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(z), index=z.index, columns=cols)
    else:
        raise ValueError("impute must be 'median' or 'knn'")
    z = (z - z.mean()) / z.std(ddof=1)
    pca = PCA(n_components=1).fit(z.to_numpy())
    load = pd.Series(pca.components_[0], index=cols)
    if orient is not None:
        if orient not in cols:
            raise ValueError(f"orient column {orient!r} not in cols")
        if load[orient] < 0:
            load = -load
    score = pd.Series(z.to_numpy() @ load.to_numpy(), index=clinical.index, name=name)
    score = (score - score.mean()) / score.std(ddof=1)
    return ClinicalScore(score=score, loadings=load, explained_variance=float(pca.explained_variance_ratio_[0]),
                         n_imputed=n_imp)
