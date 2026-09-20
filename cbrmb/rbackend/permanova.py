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
