"""Distance-based confounder / effect-modifier screening via PERMANOVA.

Wraps ``GUniFrac::adonis3``. When several samples come from the same subject
(repeated measures) the default free permutation is anti-conservative for
subject-level covariates; pass ``subject=`` and this module switches to the
correct restricted-permutation scheme (via the R ``permute`` package):

* covariate **constant within subject** (``level == "between"``) -> permute whole
  subjects as units (``how(plots = Plots(strata = subject, type = "free"),
  within = Within("none"))``); ``subject`` is *not* added to the model. This
  requires a *balanced* subject structure (equal samples per subject) --
  ``permute::Plots`` can't free-permute unevenly sized blocks. When it isn't
  (the rule, not the exception, for real cohort data), :func:`test_confounder`
  automatically falls back to ``reduce="medoid"`` and warns.
* covariate **varies within subject** (``"within"`` / ``"mixed"``) -> add
  ``subject`` as the first model term and permute only within subject
  (``how(blocks = subject)``); the covariate is tested on the within-subject
  residual. No balance requirement.

Alternatively ``reduce="medoid"`` / ``"first"`` collapses to one sample per
subject and runs an ordinary PERMANOVA -- pass it explicitly to apply it to
*every* covariate regardless of level.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import require_rpy2, r_package
from ._bridge import numpy_to_rpy2, pandas_to_rpy2, rpy2_to_pandas, square_array

__all__ = [
    "subject_variation",
    "screen_confounder",
    "test_confounder",
    "test_confounder_adjusted",
    "mediation_decompose",
    "bootstrap_mediation",
    "screen_effect_modifiers",
    "betadisper",
    "remove_confounder_nan",
]

_LEVELS_TESTABLE = {"between", "within", "mixed"}


# --------------------------------------------------------------------------- #
# subject-structure classification
# --------------------------------------------------------------------------- #
def _subject_array(n, subject):
    if subject is None:
        return None
    s = subject.values if isinstance(subject, pd.Series) else np.asarray(subject)
    if len(s) != n:
        raise ValueError(f"subject has length {len(s)}, expected {n}")
    return s.astype(object)


def _classify(values: pd.Series, subject: np.ndarray) -> tuple[str, float, int]:
    """Return ``(level, frac_subjects_varying, n_non_na)`` for one covariate."""
    v = pd.Series(np.asarray(values), copy=False)
    mask = v.notna().to_numpy()
    n_non_na = int(mask.sum())
    nlev = v[mask].nunique()

    is_numeric = pd.api.types.is_numeric_dtype(v)
    if nlev <= 1:
        return "constant", 0.0, n_non_na
    if not is_numeric and nlev == n_non_na:
        return "id_like", 1.0, n_non_na

    if subject is None:
        return "between", 0.0, n_non_na

    df = pd.DataFrame({"s": subject[mask], "v": v[mask].to_numpy()})
    grp = df.groupby("s")["v"]
    per_subj = grp.nunique()
    sizes = grp.size()
    multi = per_subj.index[(sizes > 1).to_numpy()]
    if len(multi) == 0:
        return "between", 0.0, n_non_na
    frac = float((per_subj.loc[multi] > 1).mean())

    if frac == 0.0:
        level = "between"
    elif frac >= 1.0 - 1e-9:
        level = "within"
    else:
        level = "mixed"
    return level, frac, n_non_na


def subject_variation(meta: pd.DataFrame, subject=None) -> pd.DataFrame:
    """Classify how each column of ``meta`` varies relative to ``subject``.

    ``subject`` is an array aligned to ``meta``'s rows, or a column name in
    ``meta``. Returns a DataFrame indexed by variable with ``n_levels``,
    ``n_non_na``, ``frac_subjects_varying`` and ``level`` in
    ``{"between", "within", "mixed", "constant", "id_like"}``.
    """
    if isinstance(subject, str):
        subject, meta = meta[subject], meta.drop(columns=subject)
    subj = _subject_array(len(meta), subject)

    rows = []
    for name, col in meta.items():
        level, frac, n_non_na = _classify(col, subj)
        rows.append((name, pd.Series(np.asarray(col)).nunique(dropna=True),
                     n_non_na, round(frac, 3), level))
    return pd.DataFrame(
        rows,
        columns=["var", "n_levels", "n_non_na", "frac_subjects_varying", "level"],
    ).set_index("var")


# --------------------------------------------------------------------------- #
# NaN / reduction helpers
# --------------------------------------------------------------------------- #
def remove_confounder_nan(distmat, confounder: pd.Series, subject=None):
    """Drop NaN rows of ``confounder`` and the matching rows/cols of ``distmat``.

    When ``subject`` is given it is subset the same way and returned as a third
    value.
    """
    distmat = square_array(distmat)
    keep = np.asarray(confounder.notnull().values)
    idx = np.arange(distmat.shape[0])[keep]
    d = distmat[np.ix_(idx, idx)]
    c = confounder.iloc[idx]
    if subject is None:
        return d, c
    return d, c, np.asarray(subject)[idx]


def _reduce_picks(distmat, subject: np.ndarray, how: str) -> np.ndarray:
    """Row positions selecting one representative sample per subject.

    ``"medoid"``: the within-subject sample closest (in ``distmat``) to its
    other same-subject samples. ``"first"``: the first occurrence. Shared by
    :func:`_reduce_to_subject` and the ``scheme="between"`` fallbacks (see
    module docstring), where several aligned value series (e.g. a covariate
    and a modifier) need to be subset by the exact same positions.
    """
    order = pd.unique(subject)
    pos = {s: np.where(subject == s)[0] for s in order}

    if how == "first":
        picks = [pos[s][0] for s in order]
    elif how == "medoid":
        picks = []
        for s in order:
            ii = pos[s]
            if len(ii) == 1:
                picks.append(int(ii[0]))
            else:
                sub = distmat[np.ix_(ii, ii)]
                picks.append(int(ii[int(np.argmin(sub.sum(axis=0)))]))
    else:
        raise ValueError("reduce must be 'first' or 'medoid'")

    return np.asarray(picks)


def _reduce_to_subject(distmat, values: pd.Series, subject: np.ndarray, how: str):
    """Collapse to one sample per subject; return ``(D, values, subjects)``."""
    picks = _reduce_picks(distmat, subject, how)
    vals = values.iloc[picks].reset_index(drop=True)
    return distmat[np.ix_(picks, picks)], vals, pd.unique(subject)


def _reduce_to_subject_multi(distmat, values_list, subject: np.ndarray, how: str):
    """Like :func:`_reduce_to_subject` but for several aligned series at once.

    Used where two covariates (a target and an adjustment variable) must be
    collapsed to the same one-sample-per-subject positions.
    """
    picks = _reduce_picks(distmat, subject, how)
    reduced = [v.iloc[picks].reset_index(drop=True) for v in values_list]
    return distmat[np.ix_(picks, picks)], reduced, pd.unique(subject)


# --------------------------------------------------------------------------- #
# permute::how controls
# --------------------------------------------------------------------------- #
def _r_factor(x):
    ro, *_ = require_rpy2()
    return ro.r["factor"](ro.StrVector([str(v) for v in x]))


def _how_between(subject, n_perm):
    permute = r_package("permute")
    sizes = pd.Series(subject).value_counts()
    if sizes.nunique() > 1:
        raise ValueError(
            "scheme='between' permutiert ganze Subjects (permute::Plots strata) und "
            "setzt dafür eine balancierte Subject-Struktur voraus (gleich viele Proben "
            f"je Subject); gefunden: {int(sizes.min())}-{int(sizes.max())} Proben/Subject "
            f"über {len(sizes)} Subjects. Das ist bei echten Longitudinal-/Kohortendaten "
            "die Regel, nicht die Ausnahme -- vor dem Aufruf mit reduce='first' oder "
            "reduce='medoid' auf 1 Probe/Subject reduzieren (test_confounder/"
            "screen_confounder, Parameter reduce)."
        )
    subj_f = _r_factor(subject)
    plots = permute.Plots(strata=subj_f, type="free")
    within = permute.Within(type="none")
    return permute.how(plots=plots, within=within, nperm=int(n_perm))


def _how_blocks(subject, n_perm):
    permute = r_package("permute")
    return permute.how(blocks=_r_factor(subject), nperm=int(n_perm))


def _resolve_scheme(scheme, values, subject, *, verbose=False):
    """Map ``scheme="auto"`` to between/within from the covariate's structure.

    Returns the concrete scheme, or ``None`` when the covariate is untestable
    (constant / one value per sample).
    """
    if scheme != "auto":
        return scheme
    level, _, _ = _classify(pd.Series(np.asarray(values)), subject)
    resolved = {"between": "between", "within": "within", "mixed": "within"}.get(level)
    if resolved is None and verbose:
        print(f"skip: covariate is '{level}'")
    return resolved


def _control(scheme, subject, n_perm):
    """permute::how object (or plain int) for a given scheme."""
    if scheme == "between":
        return _how_between(subject, n_perm)
    if scheme in ("within", "mixed"):
        return _how_blocks(subject, n_perm)
    return int(n_perm)  # "free"


def _r_ready_series(s: pd.Series) -> pd.Series:
    """Coerce one covariate into something ``pandas2ri`` can convert.

    Real numeric dtypes pass through untouched. Everything else -- object
    columns with mixed Python types (the classic ``1, 2, "IV"`` clinical
    field), ``category``, plain strings -- is turned into a homogeneous
    string Series with NaN preserved, which R then reads as a factor. Without
    this a single mixed-type column aborts a whole ``screen_confounder`` run.
    """
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s
    return s.where(s.isna(), s.astype(str)).astype(object)


# --------------------------------------------------------------------------- #
# adonis runners
# --------------------------------------------------------------------------- #
def _aovtab(**adonis_kwargs) -> pd.DataFrame:
    gunifrac = r_package("GUniFrac")
    return rpy2_to_pandas(gunifrac.adonis3(**adonis_kwargs)[0])


def _r2_pval(row: pd.Series) -> pd.Series:
    return pd.Series(
        {"R2": float(row["R2"]), "Pr(>F)": float(row["Pr(>F)"])}
    )


def _adonis_free(distmat, values, seed):
    ro, *_ = require_rpy2()
    from rpy2.robjects import Formula

    r_stats = r_package("stats")
    fmla = Formula("y ~ x")
    env = fmla.environment
    env["y"] = r_stats.as_dist(numpy_to_rpy2(distmat))
    env["x"] = pandas_to_rpy2(values)
    ro.r("set.seed")(seed)
    return _r2_pval(_aovtab(formula=fmla).iloc[0])


def _adonis_between(distmat, values, subject, seed, n_perm):
    ro, *_ = require_rpy2()
    from rpy2.robjects import Formula

    r_stats = r_package("stats")
    fmla = Formula("y ~ x")
    env = fmla.environment
    env["y"] = r_stats.as_dist(numpy_to_rpy2(distmat))
    env["x"] = pandas_to_rpy2(values)
    ctrl = _how_between(subject, n_perm)
    ro.r("set.seed")(seed)
    return _r2_pval(_aovtab(formula=fmla, permutations=ctrl).iloc[0])


def _adonis_within(distmat, values, subject, seed, n_perm):
    ro, *_ = require_rpy2()
    from rpy2.robjects import Formula

    r_stats = r_package("stats")
    fmla = Formula("y ~ subj + x")
    env = fmla.environment
    env["y"] = r_stats.as_dist(numpy_to_rpy2(distmat))
    env["subj"] = _r_factor(subject)
    env["x"] = pandas_to_rpy2(values)
    ctrl = _how_blocks(subject, n_perm)
    ro.r("set.seed")(seed)
    tab = _aovtab(formula=fmla, permutations=ctrl)  # sequential: subj, x, ...
    return _r2_pval(tab.loc["x"])


# --------------------------------------------------------------------------- #
# adonis runners -- adjusted (two-covariate, sequential) variants for
# test_confounder_adjusted / mediation_decompose
# --------------------------------------------------------------------------- #
def _adonis_free_adjusted(distmat, adjust, target, seed, n_perm):
    """``y ~ adjust + target``, free permutation, sequential (Type I) SS."""
    ro, *_ = require_rpy2()
    from rpy2.robjects import Formula

    r_stats = r_package("stats")
    fmla = Formula("y ~ m + x")
    env = fmla.environment
    env["y"] = r_stats.as_dist(numpy_to_rpy2(distmat))
    env["m"] = pandas_to_rpy2(adjust)
    env["x"] = pandas_to_rpy2(target)
    ro.r("set.seed")(seed)
    tab = _aovtab(formula=fmla, permutations=int(n_perm))  # sequential: m, x
    return _r2_pval(tab.loc["x"])


def _adonis_between_adjusted(distmat, adjust, target, subject, seed, n_perm):
    """``y ~ adjust + target``, whole-subject permutation, sequential SS."""
    ro, *_ = require_rpy2()
    from rpy2.robjects import Formula

    r_stats = r_package("stats")
    fmla = Formula("y ~ m + x")
    env = fmla.environment
    env["y"] = r_stats.as_dist(numpy_to_rpy2(distmat))
    env["m"] = pandas_to_rpy2(adjust)
    env["x"] = pandas_to_rpy2(target)
    ctrl = _how_between(subject, n_perm)
    ro.r("set.seed")(seed)
    tab = _aovtab(formula=fmla, permutations=ctrl)
    return _r2_pval(tab.loc["x"])


def _adonis_within_adjusted(distmat, adjust, target, subject, seed, n_perm):
    """``y ~ subj + adjust + target``, within-subject-blocked permutation."""
    ro, *_ = require_rpy2()
    from rpy2.robjects import Formula

    r_stats = r_package("stats")
    fmla = Formula("y ~ subj + m + x")
    env = fmla.environment
    env["y"] = r_stats.as_dist(numpy_to_rpy2(distmat))
    env["subj"] = _r_factor(subject)
    env["m"] = pandas_to_rpy2(adjust)
    env["x"] = pandas_to_rpy2(target)
    ctrl = _how_blocks(subject, n_perm)
    ro.r("set.seed")(seed)
    tab = _aovtab(formula=fmla, permutations=ctrl)  # sequential: subj, m, x
    return _r2_pval(tab.loc["x"])


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def test_confounder(
    distmat,
    confounder: pd.Series,
    seed: int = 42,
    *,
    subject=None,
    scheme: str = "auto",
    n_perm: int = 999,
    reduce: str | None = None,
    verbose: bool = False,
):
    """PERMANOVA of one covariate against a precomputed distance matrix.

    Without ``subject`` this is the plain free-permutation adonis (unchanged
    behaviour). With ``subject`` see the module docstring: ``scheme`` is
    ``"auto"`` (classify via :func:`subject_variation`), or one of
    ``"between"`` / ``"within"`` / ``"free"``. ``reduce`` (``"medoid"`` or
    ``"first"``) collapses to one sample per subject and ignores ``scheme``.

    Returns a 2-element Series ``{"R2", "Pr(>F)"}`` (so ``r2, pval = ...`` still
    works); ``.attrs["scheme"]`` records what was run -- including
    ``"between->reduce:medoid"`` when the automatic fallback below kicked in.

    Non-numeric covariates (including object columns with mixed Python types)
    are stringified before the R hand-off, so a messy metadata column no longer
    aborts the call -- see :func:`_r_ready_series`.

    A resolved/explicit ``scheme="between"`` needs a balanced subject
    structure (equal samples per subject); if it isn't, this automatically
    falls back to ``reduce="medoid"`` and raises a ``UserWarning`` naming the
    covariate and the imbalance, rather than letting the underlying R call
    fail with an opaque error.
    """
    distmat = square_array(distmat)
    confounder = pd.Series(confounder).reset_index(drop=True)
    subj = _subject_array(distmat.shape[0], subject)

    if confounder.isnull().any():
        if subj is None:
            distmat, confounder = remove_confounder_nan(distmat, confounder)
        else:
            distmat, confounder, subj = remove_confounder_nan(distmat, confounder, subj)
        confounder = confounder.reset_index(drop=True)

    confounder = _r_ready_series(confounder)

    if subj is None:
        out = _adonis_free(distmat, confounder, seed)
        out.attrs["scheme"] = "free"
        return out

    if reduce is not None:
        distmat, confounder, _ = _reduce_to_subject(distmat, confounder, subj, reduce)
        out = _adonis_free(distmat, confounder, seed)
        out.attrs["scheme"] = f"reduce:{reduce}"
        return out

    if scheme == "auto":
        level, _, _ = _classify(confounder, subj)
        resolved = _resolve_scheme("auto", confounder, subj, verbose=verbose)
        if resolved is None:
            out = pd.Series({"R2": np.nan, "Pr(>F)": np.nan})
            out.attrs["scheme"] = f"skip:{level}"
            return out
        scheme = resolved

    if scheme == "free":
        out = _adonis_free(distmat, confounder, seed)
    elif scheme == "between":
        try:
            out = _adonis_between(distmat, confounder, subj, seed, n_perm)
        except ValueError as e:
            cov_label = f"[{confounder.name}] " if confounder.name else ""
            warnings.warn(
                f"{cov_label}{e} Falle automatisch auf reduce='medoid' zurück (1 "
                "Probe/Subject, anschließend gewöhnliche freie Permutation)."
            )
            d_r, c_r, _ = _reduce_to_subject(distmat, confounder, subj, "medoid")
            out = _adonis_free(d_r, c_r, seed)
            out.attrs["scheme"] = "between->reduce:medoid"
            return out
    elif scheme in ("within", "mixed"):
        out = _adonis_within(distmat, confounder, subj, seed, n_perm)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    out.attrs["scheme"] = scheme
    return out


def test_confounder_adjusted(
    distmat,
    target: pd.Series,
    adjust_for: pd.Series,
    seed: int = 42,
    *,
    subject=None,
    scheme: str = "auto",
    n_perm: int = 999,
    reduce: str | None = None,
):
    """Sequential (Type I) PERMANOVA of ``target`` *after* ``adjust_for``.

    Fits ``y ~ [subj +] adjust_for + target`` (``adjust_for`` always enters
    first) and returns ``target``'s row of the sequential ANOVA table -- the
    variance ``target`` explains *beyond* ``adjust_for``. Pair this with the
    plain, unadjusted :func:`test_confounder` (``target``'s "total effect")
    to approximate how much of that total effect overlaps with
    ``adjust_for``::

        total  = test_confounder(dist, exposure, subject=subj)
        direct = test_confounder_adjusted(dist, exposure, mediator, subject=subj)
        r2_indirect = total["R2"] - direct["R2"]

    :func:`mediation_decompose` packages exactly this pair; use it directly
    unless you need the two calls separately.

    This is a descriptive PERMANOVA-R2 decomposition, not a formal causal
    mediation estimate -- it does not check the no-exposure-induced-
    mediator-outcome-confounder assumption a product-of-coefficients
    estimate would need, and R2 shares are not strictly additive/causal
    quantities the way a coefficient decomposition would be. Treat it as a
    quick, assumption-light screen for whether a candidate mediator changes
    an exposure's community-level association at all, before investing in
    per-taxon causal mediation (e.g. ``mediation``/``CMAverse`` in R).

    Same ``subject`` handling as :func:`test_confounder`. Without
    ``subject``: ordinary free-permutation adonis. With ``subject`` and
    ``scheme="auto"``: the scheme is decided from ``target``'s *own* level
    (:func:`subject_variation`) exactly as in :func:`test_confounder`,
    *not* ``adjust_for``'s -- a between-subject ``target`` always gets the
    whole-subject-permutation ``y ~ adjust_for + target`` model, a
    within/mixed ``target`` always gets the within-subject-blocked
    ``y ~ subj + adjust_for + target`` model, regardless of what
    ``adjust_for`` does. (``adjust_for`` only ever enters as a control term;
    letting it drive the scheme choice would force a between-subject
    ``target`` into a ``subj``-blocked model, where it has zero within-subject
    residual left and gets aliased out of the ANOVA table entirely --
    ``target``'s own row would simply be missing.) The whole-subject scheme
    has the same automatic ``reduce="medoid"`` fallback on an unbalanced
    subject structure as :func:`test_confounder`. ``reduce`` forces
    collapsing to one sample/subject up front, as in :func:`test_confounder`.

    A between-subject ``adjust_for`` inside a within-subject-blocked model
    (``target`` within/mixed, ``adjust_for`` between) is aliased by the
    ``subj`` term (it has no within-subject variance left to explain once
    ``subj`` is in the model) and drops out of the ANOVA table entirely --
    ``target``'s own row is unaffected, but that particular ``adjust_for``
    contributes no actual adjustment in this case; a between-subject
    covariate has no within-subject information to adjust away regardless.

    If either ``target`` or ``adjust_for`` is untestable (``"constant"`` or
    ``"id_like"``, e.g. after a degenerate resample in
    :func:`bootstrap_mediation`), returns ``{"R2": nan, "Pr(>F)": nan}`` with
    ``.attrs["scheme"]`` naming which one, instead of failing inside the R
    call -- mirroring :func:`test_confounder`'s handling of the same case.
    """
    distmat = square_array(distmat)
    target = pd.Series(target).reset_index(drop=True)
    adjust_for = pd.Series(adjust_for).reset_index(drop=True)
    subj = _subject_array(distmat.shape[0], subject)

    joint_na = target.isna().to_numpy() | adjust_for.isna().to_numpy()
    if joint_na.any():
        idx = np.arange(distmat.shape[0])[~joint_na]
        distmat = distmat[np.ix_(idx, idx)]
        target = target.iloc[idx].reset_index(drop=True)
        adjust_for = adjust_for.iloc[idx].reset_index(drop=True)
        if subj is not None:
            subj = subj[idx]

    target = _r_ready_series(target)
    adjust_for = _r_ready_series(adjust_for)

    t_level, _, _ = _classify(target, subj)
    a_level, _, _ = _classify(adjust_for, subj)
    if t_level in ("constant", "id_like") or a_level in ("constant", "id_like"):
        # mirrors test_confounder's untestable-covariate handling; matters most for
        # bootstrap_mediation, where a resample can by chance degenerate a between-
        # subject factor to a single level
        out = pd.Series({"R2": np.nan, "Pr(>F)": np.nan})
        out.attrs["scheme"] = f"skip:target={t_level},adjust_for={a_level}"
        return out

    if subj is None:
        out = _adonis_free_adjusted(distmat, adjust_for, target, seed, n_perm)
        out.attrs["scheme"] = "free"
        return out

    if reduce is not None:
        distmat, (adjust_for, target), _ = _reduce_to_subject_multi(
            distmat, [adjust_for, target], subj, reduce
        )
        out = _adonis_free_adjusted(distmat, adjust_for, target, seed, n_perm)
        out.attrs["scheme"] = f"reduce:{reduce}"
        return out

    if scheme == "auto":
        # driven by target's own level only, like test_confounder -- adjust_for
        # is a control term, not what decides the permutation scheme (see
        # docstring: keying this off adjust_for too aliases a between-subject
        # target out of the ANOVA table whenever adjust_for is within/mixed)
        scheme = "within" if t_level in ("within", "mixed") else "between"

    if scheme == "between":
        try:
            out = _adonis_between_adjusted(distmat, adjust_for, target, subj, seed, n_perm)
        except ValueError as e:
            warnings.warn(
                f"{e} Falle automatisch auf reduce='medoid' zurück (1 Probe/Subject, "
                "anschließend gewöhnliche freie Permutation)."
            )
            d_r, (m_r, t_r), _ = _reduce_to_subject_multi(
                distmat, [adjust_for, target], subj, "medoid"
            )
            out = _adonis_free_adjusted(d_r, m_r, t_r, seed, n_perm)
            out.attrs["scheme"] = "between->reduce:medoid"
            return out
    elif scheme in ("within", "mixed"):
        out = _adonis_within_adjusted(distmat, adjust_for, target, subj, seed, n_perm)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    out.attrs["scheme"] = scheme
    return out


def screen_confounder(
    distmat,
    confounders: pd.DataFrame,
    seed: int = 42,
    *,
    subject=None,
    scheme: str = "auto",
    n_perm: int = 999,
    reduce: str | None = None,
    verbose: bool = True,
):
    """Run :func:`test_confounder` for every column of ``confounders``.

    Without ``subject``: unchanged: skips single-level columns, returns
    ``["var", "r2", "pval"]``.

    With ``subject``: also classifies each column (:func:`subject_variation`),
    picks the restricted-permutation scheme per column, and returns
    ``["var", "level", "n", "scheme", "r2", "pval", "fdr"]`` sorted by ``pval``.
    """
    if subject is None:
        rows = []
        for name, values in confounders.items():
            if values.nunique() <= 1:
                if verbose:
                    print(f"Dropping {name}, levels not suitable")
                continue
            try:
                rows.append((name, *test_confounder(distmat, values, seed)))
            except Exception as e:
                if verbose:
                    print(f"Skipping {name}: {e}")
        return pd.DataFrame(rows, columns=("var", "r2", "pval"))

    from ..pvalues import fdr

    n = square_array(distmat).shape[0]
    subj = _subject_array(n, subject)
    rows = []
    for name, values in confounders.items():
        level, _, n_non_na = _classify(values, subj)
        if level not in _LEVELS_TESTABLE:
            if verbose:
                print(f"Dropping {name}: level '{level}'")
            continue
        try:
            if reduce is not None:
                res = test_confounder(distmat, values, seed, subject=subj,
                                      n_perm=n_perm, reduce=reduce)
            else:
                res = test_confounder(distmat, values, seed, subject=subj,
                                      scheme=scheme, n_perm=n_perm)
        except Exception as e:
            if verbose:
                print(f"Skipping {name}: {e}")
            continue
        rows.append((name, level, n_non_na, res.attrs.get("scheme", ""),
                     float(res["R2"]), float(res["Pr(>F)"])))

    out = pd.DataFrame(
        rows, columns=["var", "level", "n", "scheme", "r2", "pval"]
    )
    if len(out):
        out["fdr"] = fdr(out["pval"])
        out = out.sort_values("pval", ignore_index=True)
    return out


def screen_effect_modifiers(
    distmat,
    covariate: pd.Series,
    modifiers: pd.DataFrame,
    *,
    subject=None,
    seed: int = 42,
    n_perm: int = 999,
):
    """Test ``covariate : modifier`` interaction for each column of ``modifiers``.

    Fits ``D ~ [subject +] covariate + modifier + covariate:modifier`` (adonis is
    sequential, so the interaction is tested last = adjusted for both main
    effects). Permutation scheme follows the "more within-subject" of covariate
    and modifier. Returns ``["modifier", "cov_level", "mod_level", "scheme",
    "r2_interaction", "pval", "fdr"]`` sorted by ``pval``.

    When ``scheme`` resolves to ``"between"`` (both covariate and modifier are
    subject-constant) this needs a balanced subject structure for that
    modifier's non-missing rows; if it isn't, falls back per-modifier to
    ``reduce="medoid"`` (``scheme`` then reads ``"between->reduce:medoid"``)
    and raises a ``UserWarning`` naming the modifier.
    """
    ro, *_ = require_rpy2()
    from rpy2.robjects import Formula

    from ..pvalues import fdr

    r_stats = r_package("stats")
    distmat = square_array(distmat)
    covariate = pd.Series(covariate).reset_index(drop=True)
    subj = _subject_array(distmat.shape[0], subject)
    cov_level, _, _ = _classify(covariate, subj)
    covariate = _r_ready_series(covariate)

    rank = {"between": 0, "constant": 0, "id_like": 0, "mixed": 1, "within": 2}
    rows = []
    for name, mod in modifiers.items():
        mod = pd.Series(np.asarray(mod)).reset_index(drop=True)
        mod_level, _, _ = _classify(mod, subj)
        mod = _r_ready_series(mod)

        joint_na = covariate.isna().to_numpy() | mod.isna().to_numpy()
        d = distmat[np.ix_(~joint_na, ~joint_na)]
        cov_i = covariate[~joint_na]
        mod_i = mod[~joint_na]
        subj_i = None if subj is None else subj[~joint_na]

        within_ish = subj_i is not None and max(
            rank.get(cov_level, 1), rank.get(mod_level, 1)
        ) >= 1
        use_subj_term = False
        if subj_i is None:
            fmla = Formula("y ~ x * m")
            scheme = "free"
            ctrl = int(n_perm)
        elif within_ish:
            fmla = Formula("y ~ subj + x * m")
            scheme = "within"
            ctrl = _how_blocks(subj_i, n_perm)
            use_subj_term = True
        else:
            fmla = Formula("y ~ x * m")
            scheme = "between"
            try:
                ctrl = _how_between(subj_i, n_perm)
            except ValueError as e:
                warnings.warn(
                    f"[{name}] {e} Falle automatisch auf reduce='medoid' zurück (1 "
                    "Probe/Subject, anschließend gewöhnliche freie Permutation)."
                )
                picks = _reduce_picks(d, subj_i, "medoid")
                d = d[np.ix_(picks, picks)]
                cov_i = cov_i.iloc[picks].reset_index(drop=True)
                mod_i = mod_i.iloc[picks].reset_index(drop=True)
                scheme = "between->reduce:medoid"
                ctrl = int(n_perm)

        env = fmla.environment
        env["y"] = r_stats.as_dist(numpy_to_rpy2(d))
        env["x"] = pandas_to_rpy2(cov_i)
        env["m"] = pandas_to_rpy2(mod_i)
        if use_subj_term:
            env["subj"] = _r_factor(subj_i)
        ro.r("set.seed")(seed)
        tab = _aovtab(formula=fmla, permutations=ctrl)

        inter = [ix for ix in tab.index if ":" in ix]
        row = tab.loc[inter[0]] if inter else tab.iloc[-3]
        rows.append((name, cov_level, mod_level, scheme,
                     float(row["R2"]), float(row["Pr(>F)"])))

    out = pd.DataFrame(
        rows,
        columns=["modifier", "cov_level", "mod_level", "scheme",
                 "r2_interaction", "pval"],
    )
    if len(out):
        out["fdr"] = fdr(out["pval"])
        out = out.sort_values("pval", ignore_index=True)
    return out


def betadisper(distmat, group, *, subject=None, scheme: str = "auto",
               n_perm: int = 999, seed: int = 42):
    """Homogeneity-of-dispersion test (``vegan::betadisper`` + ``permutest``).

    A significant PERMANOVA can reflect group differences in *spread* rather than
    location; this checks for that. With ``subject`` the permutation is
    restricted the same way as in :func:`test_confounder` (``scheme="auto"``
    classifies ``group`` as between- or within-subject) -- including the same
    automatic ``reduce="medoid"`` fallback (with a ``UserWarning``) when a
    resolved ``"between"`` scheme meets an unbalanced subject structure; the
    dispersion itself is then computed on the reduced (1 sample/subject) data,
    so ``.attrs["group_dist_to_centroid"]`` and the F-test stay consistent
    with what was actually permuted. Returns a Series ``{"F", "pval"}`` with
    ``.attrs["group_dist_to_centroid"]`` and ``.attrs["scheme"]`` (the latter
    reads ``"between->reduce:medoid"`` when the fallback fired).
    """
    ro, *_ = require_rpy2()
    r_stats = r_package("stats")
    vegan = r_package("vegan")

    distmat = square_array(distmat)
    group = pd.Series(np.asarray(group)).reset_index(drop=True)

    if subject is None:
        ctrl, used = int(n_perm), "free"
    else:
        subj = _subject_array(distmat.shape[0], subject)
        used = _resolve_scheme(scheme, group, subj) or "between"
        try:
            ctrl = _control(used, subj, n_perm)
        except ValueError as e:
            if used != "between":
                raise
            warnings.warn(
                f"{e} Falle automatisch auf reduce='medoid' zurück (1 Probe/Subject, "
                "anschließend gewöhnliche freie Permutation)."
            )
            distmat, group, _ = _reduce_to_subject(distmat, group, subj, "medoid")
            ctrl, used = int(n_perm), "between->reduce:medoid"

    d = r_stats.as_dist(numpy_to_rpy2(distmat))
    g = _r_factor(group)
    bd = vegan.betadisper(d, g)

    ro.r("set.seed")(seed)
    pt = vegan.permutest(bd, permutations=ctrl)

    tab = rpy2_to_pandas(pt.rx2("tab"))
    means_r = ro.r("function(b) tapply(b$distances, b$group, mean)")(bd)
    means = pd.Series(np.asarray(means_r), index=list(ro.r["names"](means_r)))

    out = pd.Series({"F": float(tab.loc["Groups", "F"]),
                    "pval": float(tab.loc["Groups", "Pr(>F)"])})
    out.attrs["group_dist_to_centroid"] = means
    out.attrs["scheme"] = used
    return out


# --------------------------------------------------------------------------- #
# descriptive PERMANOVA-R2 mediation decomposition
# --------------------------------------------------------------------------- #
def _align_complete_case(distmat, exposure, mediator, subj):
    """Joint-complete-case subset so total/direct models share one sample set."""
    exposure = pd.Series(exposure).reset_index(drop=True)
    mediator = pd.Series(mediator).reset_index(drop=True)
    joint_na = exposure.isna().to_numpy() | mediator.isna().to_numpy()
    if not joint_na.any():
        return distmat, exposure, mediator, subj
    idx = np.arange(distmat.shape[0])[~joint_na]
    d = distmat[np.ix_(idx, idx)]
    e = exposure.iloc[idx].reset_index(drop=True)
    m = mediator.iloc[idx].reset_index(drop=True)
    s = None if subj is None else subj[idx]
    return d, e, m, s


def mediation_decompose(
    distmat,
    exposure: pd.Series,
    mediator: pd.Series,
    seed: int = 42,
    *,
    subject=None,
    scheme: str = "auto",
    n_perm: int = 999,
    reduce: str | None = None,
):
    """Descriptive PERMANOVA-R2 mediation decomposition of ``exposure``.

    Splits ``exposure``'s community-level association (``R2_total``, from
    the plain :func:`test_confounder`) into a part that survives adjusting
    for ``mediator`` (``R2_direct``, from :func:`test_confounder_adjusted`)
    and the remainder (``R2_indirect = R2_total - R2_direct``). Both models
    are fit on the same joint-complete-case subset (rows where neither
    ``exposure`` nor ``mediator`` is missing), so the two R2 values are
    comparable.

    Not a formal causal mediation estimate -- see the caveats in
    :func:`test_confounder_adjusted`. Use this as a fast first look at
    whether a candidate mediator changes ``exposure``'s PERMANOVA R2 at all;
    follow up with :func:`bootstrap_mediation` for a resampling CI on
    ``R2_indirect``, and with per-taxon causal mediation for anything you
    plan to report as a mediation finding.

    Returns a ``pandas.Series`` with ``R2_total``, ``pval_total``,
    ``R2_direct``, ``pval_direct``, ``R2_indirect`` and ``R2_indirect_frac``
    (``R2_indirect / R2_total``; ``nan`` if ``R2_total`` is ~0).
    ``.attrs["scheme_total"]`` / ``["scheme_direct"]`` record what each
    sub-model actually ran (see ``test_confounder``/``test_confounder_adjusted``).
    """
    distmat = square_array(distmat)
    subj = _subject_array(distmat.shape[0], subject)
    distmat, exposure, mediator, subj = _align_complete_case(distmat, exposure, mediator, subj)

    total = test_confounder(distmat, exposure, seed, subject=subj,
                             scheme=scheme, n_perm=n_perm, reduce=reduce)
    direct = test_confounder_adjusted(distmat, exposure, mediator, seed,
                                       subject=subj, scheme=scheme,
                                       n_perm=n_perm, reduce=reduce)

    r2_total, r2_direct = float(total["R2"]), float(direct["R2"])
    r2_indirect = r2_total - r2_direct
    out = pd.Series({
        "R2_total": r2_total,
        "pval_total": float(total["Pr(>F)"]),
        "R2_direct": r2_direct,
        "pval_direct": float(direct["Pr(>F)"]),
        "R2_indirect": r2_indirect,
        "R2_indirect_frac": r2_indirect / r2_total if r2_total > 1e-12 else np.nan,
    })
    out.attrs["scheme_total"] = total.attrs.get("scheme")
    out.attrs["scheme_direct"] = direct.attrs.get("scheme")
    return out


def bootstrap_mediation(
    distmat,
    exposure: pd.Series,
    mediator: pd.Series,
    seed: int = 42,
    *,
    subject=None,
    scheme: str = "auto",
    n_boot: int = 1000,
    reduce: str | None = None,
    ci: float = 0.95,
    verbose: bool = False,
):
    """Cluster-bootstrap percentile CI for :func:`mediation_decompose`.

    Resamples the resampling *unit* with replacement ``n_boot`` times --
    subjects if ``subject`` is given, individual samples otherwise -- refits
    the total/direct models on each resample, and collects the bootstrap
    distribution of ``R2_total``, ``R2_direct`` and ``R2_indirect``.

    Each bootstrap replicate uses ``n_perm=2``, the minimum ``GUniFrac::adonis3``
    needs to build a well-formed sequential ANOVA table for a 2/3-term model
    (``n_perm=1`` makes it emit a malformed table and error for the
    two-covariate ``m + x`` formulas this module uses); an adonis R2/pseudo-F
    is otherwise a deterministic function of the (resampled) distance matrix
    and design, computed before any permutation runs -- only the *p-value*
    needs many permutations, and the p-value is not used here, so this still
    keeps 1000+ replicates fast. Never read the per-replicate p-value as
    meaningful; only ``observed`` (fit with the real ``n_perm``) has a valid
    one. Warns if fewer than half the replicates succeed, since a replicate
    that raises (e.g. a degenerate resample) is silently dropped rather than
    failing the whole call -- see ``n_boot_ok`` in ``"summary"``.

    When a subject is drawn more than once in a replicate, its copies are
    relabelled (``f"{subject}__{k}"`` for the k-th draw) before refitting --
    otherwise the ``within`` scheme's ``subj`` blocking factor or the
    ``between`` scheme's whole-subject permutation would silently merge two
    independent resampled copies back into one block, which is not what a
    cluster bootstrap means.

    Returns a dict: ``"observed"`` (the real, ``n_perm``-permutation
    :func:`mediation_decompose` result), ``"boot"`` (a DataFrame with one row
    per successful replicate: ``R2_total``, ``R2_direct``, ``R2_indirect``),
    and ``"summary"`` (a DataFrame indexed by those three columns with
    ``mean``, ``sd``, the ``ci``-level percentile interval, and
    ``n_boot_ok``, the number of replicates that didn't error out).

    This is a percentile bootstrap on a PERMANOVA R2 -- useful for "how much
    does R2_indirect move around under resampling", not a substitute for a
    causal-mediation sensitivity analysis (it says nothing about unmeasured
    mediator-outcome confounding).
    """
    distmat = square_array(distmat)
    subj = _subject_array(distmat.shape[0], subject)
    distmat, exposure, mediator, subj = _align_complete_case(distmat, exposure, mediator, subj)

    observed = mediation_decompose(distmat, exposure, mediator, seed,
                                    subject=subj, scheme=scheme, n_perm=999,
                                    reduce=reduce)

    n = distmat.shape[0]
    rng = np.random.default_rng(seed)
    units = pd.unique(subj) if subj is not None else np.arange(n)

    rows = []
    for k in range(n_boot):
        draw = rng.choice(units, size=len(units), replace=True)
        if subj is None:
            idx = draw
            subj_boot = None
        else:
            idx_parts, subj_parts = [], []
            for j, u in enumerate(draw):
                pos = np.where(subj == u)[0]
                idx_parts.append(pos)
                subj_parts.append(np.full(len(pos), f"{u}__{j}", dtype=object))
            idx = np.concatenate(idx_parts)
            subj_boot = np.concatenate(subj_parts)

        d_b = distmat[np.ix_(idx, idx)]
        exp_b = exposure.iloc[idx].reset_index(drop=True)
        med_b = mediator.iloc[idx].reset_index(drop=True)
        try:
            dec = mediation_decompose(d_b, exp_b, med_b, seed + k + 1,
                                       subject=subj_boot, scheme=scheme,
                                       n_perm=2, reduce=reduce)
            rows.append(dec[["R2_total", "R2_direct", "R2_indirect"]])
        except Exception as e:
            if verbose:
                print(f"bootstrap replicate {k} skipped: {e}")
            continue

    boot = pd.DataFrame(rows).reset_index(drop=True)
    if len(boot) < n_boot / 2:
        warnings.warn(
            f"only {len(boot)}/{n_boot} bootstrap replicates succeeded; "
            "the CI in 'summary' is based on fewer replicates than requested "
            "(see the per-replicate errors with verbose=True)."
        )
    alpha = (1 - ci) / 2
    summary = {}
    for col in ("R2_total", "R2_direct", "R2_indirect"):
        vals = boot[col].dropna().to_numpy() if col in boot else np.array([])
        summary[col] = {
            "mean": float(np.mean(vals)) if len(vals) else np.nan,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            f"ci{int(ci * 100)}_lo": float(np.quantile(vals, alpha)) if len(vals) else np.nan,
            f"ci{int(ci * 100)}_hi": float(np.quantile(vals, 1 - alpha)) if len(vals) else np.nan,
            "n_boot_ok": int(len(vals)),
        }

    return {"observed": observed, "boot": boot, "summary": pd.DataFrame(summary).T}
