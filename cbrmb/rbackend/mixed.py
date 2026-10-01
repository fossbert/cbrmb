"""Mixed-effects models for alpha diversity in longitudinal / repeated-measures data.

One row per sample, several samples per ``subject``. The response is a single
alpha-diversity measure; the family follows from what kind of number it is:

* ``"lognormal"`` -- positive continuous measures such as the **Shannon effective
  number** (``exp(H)``). Fits ``log(y)`` with ``lmerTest::lmer`` (Satterthwaite
  df); ``exp(beta)`` is a *ratio* of geometric means. Note ``log(Shannon.Effective)``
  is just the Shannon index, so this is the classic Shannon LMM, reported on the
  effective-number scale.
* ``"negbin"`` -- counts such as (observed / rarefied) **richness**. Negative-
  binomial GLMM via ``lme4::glmer.nb`` (log link); ``exp(beta)`` is a rate ratio.
* ``"poisson"`` / ``"gaussian"`` -- for completeness (``glmer`` / ``lmer`` on
  the raw scale).

Random effects: ``random="intercept"`` -> ``(1 | subject)``, ``"slope"`` ->
``(1 + time | subject)``, or any lme4 random-effect string. Serial correlation
beyond the random effects: ``correlation="car1"`` switches the (log-)normal
families to ``nlme::lme`` with a continuous-time AR(1) (``corCAR1``), which
also handles irregular visit spacing.

Sequencing depth: pass ``depth=`` (a column of read counts) to adjust for
``log(depth)`` (centered, so the intercept refers to the mean log depth), or
``depth_as="offset"`` for count families. Rarefied richness needs neither.

Term tests (:attr:`AlphaMixedFit.terms`) respect marginality -- with
``group * time`` only the interaction is tested, not the main effects. They are
Satterthwaite F-tests for ``lmer`` fits and likelihood-ratio tests (ML refits of
the model without the term) otherwise.

Needs the ``r`` extra plus the R packages ``lme4``, ``lmerTest`` and ``nlme``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..pvalues import fdr
from . import require_rpy2, r_package
from ._bridge import pandas_to_rpy2, rpy2_to_pandas

__all__ = ["alpha_mixed", "alpha_mixed_screen", "AlphaMixedFit"]

FAMILIES = {"lognormal", "gaussian", "negbin", "poisson"}
_LOG_FAMILIES = {"lognormal", "negbin", "poisson"}
_COUNT_FAMILIES = {"negbin", "poisson"}
_DEPTH_COL = ".log_depth"

_R_CODE = r"""
local({
cbrmb_collect <- function(expr) {
    warns <- character(0)
    val <- withCallingHandlers(expr,
        warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
        message = function(m) { warns <<- c(warns, trimws(conditionMessage(m))); invokeRestart("muffleMessage") })
    list(value = val, warnings = unique(warns))
}

cbrmb_fitter <- function(family, engine, random, subject, time, corr, data) {
    function(fixed, reml = TRUE) {
        if (engine == "lme") {
            # nlme cannot parse backticked grouping names: use syntactic copies
            rnd <- if (random == "slope") ~ .time | .subject else ~ 1 | .subject
            cs <- nlme::corCAR1(form = ~ .time | .subject)
            return(nlme::lme(fixed, random = rnd, correlation = cs, data = data,
                             method = if (reml) "REML" else "ML",
                             control = nlme::lmeControl(opt = "optim")))
        }
        f <- update(fixed, as.formula(paste(". ~ . +", random)))
        environment(f) <- environment()
        if (family %in% c("lognormal", "gaussian")) {
            lmerTest::lmer(f, data = data, REML = reml)
        } else if (family == "poisson") {
            lme4::glmer(f, data = data, family = poisson(),
                        control = lme4::glmerControl(optimizer = "bobyqa"))
        } else {
            lme4::glmer.nb(f, data = data,
                           control = lme4::glmerControl(optimizer = "bobyqa"))
        }
    }
}

cbrmb_coef <- function(fit, engine, conf) {
    if (engine == "lme") {
        tt <- summary(fit)$tTable
        est <- tt[, "Value"]; se <- tt[, "Std.Error"]; df <- tt[, "DF"]
        stat <- tt[, "t-value"]; p <- tt[, "p-value"]
    } else {
        tt <- summary(fit)$coefficients
        est <- tt[, 1]; se <- tt[, 2]; stat <- tt[, 3]; p <- tt[, ncol(tt)]
        df <- if ("df" %in% colnames(tt)) tt[, "df"] else rep(NA_real_, nrow(tt))
    }
    q <- ifelse(is.na(df), qnorm(1 - (1 - conf) / 2), qt(1 - (1 - conf) / 2, pmax(df, 1)))
    data.frame(term = rownames(tt), estimate = est, se = se, df = df, stat = stat,
               pval = p, ci_low = est - q * se, ci_high = est + q * se,
               row.names = NULL, stringsAsFactors = FALSE)
}

cbrmb_terms <- function(fit, fitter, fixed, engine, family) {
    scope <- drop.scope(terms(fixed))
    if (length(scope) == 0) return(data.frame())
    if (engine == "lmer") {
        d <- drop1(fit)
        return(data.frame(term = rownames(d), test = "F (Satterthwaite)",
                          stat = d[["F value"]], df = d[["NumDF"]], den_df = d[["DenDF"]],
                          pval = d[["Pr(>F)"]], row.names = NULL, stringsAsFactors = FALSE))
    }
    full <- if (engine == "lme") fitter(fixed, reml = FALSE) else fit
    ll_full <- logLik(full)
    rows <- lapply(scope, function(tm) {
        r <- cbrmb_collect(fitter(update(fixed, as.formula(paste(". ~ . -", tm))), reml = FALSE))
        for (w in r$warnings) warning(sprintf("[refit without %s] %s", tm, w), call. = FALSE)
        ll_red <- logLik(r$value)
        chisq <- max(0, 2 * (as.numeric(ll_full) - as.numeric(ll_red)))
        ddf <- attr(ll_full, "df") - attr(ll_red, "df")
        data.frame(term = tm, test = "LRT", stat = chisq, df = ddf, den_df = NA_real_,
                   pval = pchisq(chisq, ddf, lower.tail = FALSE), stringsAsFactors = FALSE)
    })
    do.call(rbind, rows)
}

cbrmb_random <- function(fit, engine) {
    if (engine == "lme") {
        vc <- nlme::VarCorr(fit)
        sds <- suppressWarnings(as.numeric(vc[, "StdDev"]))
        nm <- rownames(vc)
        grp <- ifelse(nm == "Residual", "Residual", "subject")
        return(data.frame(group = grp, name = nm, sd = sds, stringsAsFactors = FALSE))
    }
    vc <- as.data.frame(lme4::VarCorr(fit))
    vc <- vc[is.na(vc$var2), ]
    data.frame(group = ifelse(vc$grp == "Residual", "Residual", "subject"),
               name = ifelse(is.na(vc$var1), "Residual", vc$var1),
               sd = vc$sdcor, stringsAsFactors = FALSE)
}

cbrmb_info <- function(fit, engine, family, data, subject) {
    singular <- if (engine == "lme") NA_real_ else as.numeric(lme4::isSingular(fit))
    theta <- if (family == "negbin") lme4::getME(fit, "glmer.nb.theta") else NA_real_
    phi <- NA_real_
    if (engine == "lme") phi <- as.numeric(coef(fit$modelStruct$corStruct, unconstrained = FALSE))
    disp <- NA_real_
    if (family == "poisson") disp <- sum(residuals(fit, type = "pearson")^2) / df.residual(fit)
    list(n_obs = nrow(data), n_subjects = length(unique(data[[subject]])),
         aic = AIC(fit), loglik = as.numeric(logLik(fit)), singular = singular,
         theta = theta, car1_phi = phi, pearson_dispersion = disp)
}

cbrmb_alpha_fit <- function(data, lhs, fixed_rhs, family, engine, random, subject, time, conf) {
    fixed <- as.formula(paste(lhs, "~", fixed_rhs))
    vars <- intersect(unique(c(all.vars(fixed), subject, time)), names(data))
    data <- data[complete.cases(data[, vars, drop = FALSE]), , drop = FALSE]
    data[[subject]] <- factor(data[[subject]])
    data$.subject <- data[[subject]]
    if (!is.null(time)) data$.time <- data[[time]]
    fitter <- cbrmb_fitter(family, engine, random, subject, time, NULL, data)
    res <- cbrmb_collect({
        fit <- fitter(fixed)
        list(fit = fit, coef = cbrmb_coef(fit, engine, conf),
             terms = cbrmb_terms(fit, fitter, fixed, engine, family),
             random = cbrmb_random(fit, engine),
             info = cbrmb_info(fit, engine, family, data, subject))
    })
    rhs_terms <- delete.response(terms(fixed))
    xlev <- .getXlevels(rhs_terms, model.frame(rhs_terms, data))
    c(res$value, list(warnings = res$warnings, rhs_terms = rhs_terms, xlev = xlev))
}

cbrmb_alpha_predict <- function(obj, newdata, engine, conf) {
    tt <- obj$rhs_terms
    mf <- model.frame(tt, newdata, xlev = obj$xlev, na.action = na.pass)
    X <- model.matrix(tt, mf)
    fit <- obj$fit
    if (engine == "lme") { beta <- nlme::fixef(fit); V <- fit$varFix }
    else { beta <- lme4::fixef(fit); V <- as.matrix(vcov(fit)) }
    X <- X[, names(beta), drop = FALSE]
    off <- model.offset(mf); if (is.null(off)) off <- 0
    eta <- as.vector(X %*% beta) + off
    se <- sqrt(rowSums((X %*% V) * X))
    z <- qnorm(1 - (1 - conf) / 2)
    data.frame(eta = eta, se = se, eta_low = eta - z * se, eta_high = eta + z * se)
}

assign("cbrmb_alpha_fit", cbrmb_alpha_fit, envir = globalenv())
assign("cbrmb_alpha_predict", cbrmb_alpha_predict, envir = globalenv())
})
"""

_R_READY = False


def _r_env():
    """Load lme4 / lmerTest / nlme and define the R helpers once per session."""
    global _R_READY
    ro, *_ = require_rpy2()
    if not _R_READY:
        for pkg in ("lme4", "lmerTest", "nlme"):
            r_package(pkg)
        ro.r(_R_CODE)
        _R_READY = True
    return ro


def _bt(name: str) -> str:
    """Backtick-quote a column name for an R formula."""
    return "`" + str(name).replace("`", "") + "`"


def _is_count(y: pd.Series) -> bool:
    y = y.dropna()
    return len(y) > 0 and bool(np.all(y >= 0)) and bool(np.allclose(y, np.round(y)))


def _resolve_family(y: pd.Series, family: str) -> str:
    if family == "auto":
        return "negbin" if _is_count(y) else "lognormal"
    if family not in FAMILIES:
        raise ValueError(f"family must be 'auto' or one of {sorted(FAMILIES)}, got {family!r}")
    return family


def _random_term(random: str, subject: str, time) -> str:
    if random == "intercept":
        return f"(1 | {_bt(subject)})"
    if random == "slope":
        if time is None:
            raise ValueError("random='slope' needs time=")
        return f"(1 + {_bt(time)} | {_bt(subject)})"
    if "|" not in random:
        raise ValueError(
            "random must be 'intercept', 'slope' or an lme4 term such as '(1 | subject)'"
        )
    return random


def _prepare(data, response, subject, time, depth, depth_as, family):
    """Copy ``data`` with a string subject column and the centered log-depth term."""
    for col in [response, subject, time, depth]:
        if col is not None and col not in data.columns:
            raise KeyError(f"column {col!r} not in data")
    df = data.copy()
    df[subject] = df[subject].astype("string").astype(object)
    extra_rhs, offset = None, None
    if depth is not None:
        d = pd.to_numeric(df[depth], errors="coerce").astype(float)
        if (d <= 0).any():
            raise ValueError(f"depth column {depth!r} must be > 0")
        logd = np.log(d)
        if depth_as == "covariate":
            df[_DEPTH_COL] = logd - logd.mean()
            extra_rhs = _DEPTH_COL
        elif depth_as == "offset":
            if family not in _COUNT_FAMILIES:
                raise ValueError("depth_as='offset' only makes sense for count families")
            df[_DEPTH_COL] = logd
            offset = logd.mean()
            extra_rhs = f"offset({_DEPTH_COL})"
        else:
            raise ValueError("depth_as must be 'covariate' or 'offset'")
    return df, extra_rhs, offset


@dataclass
class AlphaMixedFit:
    """Result of :func:`alpha_mixed`.

    Attributes
    ----------
    coef : DataFrame
        Fixed effects on the link scale (``estimate``, ``se``, ``df``, ``stat``,
        ``pval``, Wald ``ci_low``/``ci_high``); log-link / lognormal fits also carry
        ``ratio``, ``ratio_low``, ``ratio_high`` = ``exp(...)``.
    terms : DataFrame
        One test per droppable model term (marginality respected).
    random : DataFrame
        Random-effect and residual standard deviations.
    info : Series
        ``n_obs``, ``n_subjects``, ``aic``, ``loglik``, ``singular``, ``theta`` (NB),
        ``car1_phi``, ``pearson_dispersion`` (Poisson), ``icc`` ((log-)normal,
        intercept-only), plus ``family`` / ``engine`` / ``formula``.
    warnings : list of str
        Convergence / singularity messages from R.
    r_fit :
        The underlying R model object (rpy2), for anything not wrapped here.
    """

    response: str
    family: str
    engine: str
    formula: str
    coef: pd.DataFrame
    terms: pd.DataFrame
    random: pd.DataFrame
    info: pd.Series
    warnings: list = field(default_factory=list)
    r_fit: object = field(default=None, repr=False)
    _r_obj: object = field(default=None, repr=False)
    _depth: dict = field(default_factory=dict, repr=False)
    _subject: str = field(default="", repr=False)

    def predict(self, newdata: pd.DataFrame, *, conf_level: float = 0.95) -> pd.DataFrame:
        """Population-level (random effects = 0) predictions with Wald CIs.

        ``newdata`` needs the fixed-effect columns; factor levels are matched to
        the fitted data. A depth covariate / offset missing from ``newdata`` is
        set to the mean log depth of the fitted data. Returns ``newdata`` with
        ``fit``, ``fit_low``, ``fit_high`` on the response scale -- for
        ``"lognormal"`` that is the geometric mean, for count families the
        expected count.
        """
        ro = _r_env()
        nd = newdata.copy()
        dep = self._depth
        if dep.get("col") is not None:
            if dep["col"] in nd.columns:
                logd = np.log(pd.to_numeric(nd[dep["col"]], errors="coerce").astype(float))
                nd[_DEPTH_COL] = logd - dep["center"] if dep["as"] == "covariate" else logd
            else:
                nd[_DEPTH_COL] = 0.0 if dep["as"] == "covariate" else dep["center"]
        out = rpy2_to_pandas(
            ro.r["cbrmb_alpha_predict"](
                self._r_obj, pandas_to_rpy2(nd.reset_index(drop=True)), self.engine, conf_level
            )
        )
        inv = np.exp if self.family in _LOG_FAMILIES else (lambda v: v)
        res = newdata.copy()
        res["fit"] = inv(out["eta"].to_numpy())
        res["fit_low"] = inv(out["eta_low"].to_numpy())
        res["fit_high"] = inv(out["eta_high"].to_numpy())
        return res


def alpha_mixed(
    data: pd.DataFrame,
    response: str,
    fixed: str,
    subject: str,
    *,
    family: str = "auto",
    random: str = "intercept",
    time: str | None = None,
    depth: str | None = None,
    depth_as: str = "covariate",
    correlation: str | None = None,
    conf_level: float = 0.95,
) -> AlphaMixedFit:
    """Fit a mixed-effects model for one alpha-diversity measure.

    Parameters
    ----------
    data : DataFrame
        One row per sample. Rows with NA in any model variable are dropped.
    response : str
        Column with the alpha-diversity measure, e.g. ``"Shannon.Effective"`` or
        ``"Richness"``.
    fixed : str
        Right-hand side of the fixed-effects formula in R syntax, e.g.
        ``"group * time + age"``. Non-syntactic column names need backticks.
    subject : str
        Column with the subject / patient id (random-effect grouping).
    family : {"auto", "lognormal", "negbin", "poisson", "gaussian"}
        ``"auto"``: ``"negbin"`` for non-negative integers, else ``"lognormal"``.
    random : {"intercept", "slope"} or str
        ``"slope"`` = random intercept + random ``time`` slope per subject. Any
        other string is used verbatim as the lme4 random-effect term(s).
    time : str, optional
        Numeric time column; needed for ``random="slope"`` and ``correlation``.
        Centering/scaling it (e.g. years since baseline) helps convergence.
    depth : str, optional
        Column with read counts per sample; adjusts for ``log(depth)``.
    depth_as : {"covariate", "offset"}
        ``"offset"`` (count families only) models diversity per read.
    correlation : {None, "car1"}
        ``"car1"`` adds a continuous-time AR(1) residual correlation within
        subject (``nlme::lme``; lognormal / gaussian only; ``random`` must be
        ``"intercept"`` or ``"slope"``).
    conf_level : float

    Returns
    -------
    AlphaMixedFit
    """
    ro = _r_env()
    family = _resolve_family(data[response], family)
    if correlation not in (None, "car1"):
        raise ValueError("correlation must be None or 'car1'")
    if correlation == "car1":
        if family in _COUNT_FAMILIES:
            raise ValueError("correlation='car1' is only available for lognormal / gaussian")
        if time is None:
            raise ValueError("correlation='car1' needs time=")
        if random not in ("intercept", "slope"):
            raise ValueError("with correlation='car1', random must be 'intercept' or 'slope'")
        engine = "lme"
    else:
        engine = "lmer" if family in ("lognormal", "gaussian") else "glmer"

    y = pd.to_numeric(data[response], errors="coerce")
    if family == "lognormal" and (y.dropna() <= 0).any():
        raise ValueError(f"{response!r} has values <= 0; lognormal needs strictly positive data")

    df, extra_rhs, offset = _prepare(data, response, subject, time, depth, depth_as, family)
    lhs = f"log({_bt(response)})" if family == "lognormal" else _bt(response)
    rhs = fixed if extra_rhs is None else f"{fixed} + {extra_rhs}"
    rnd = random if engine == "lme" else _random_term(random, subject, time)

    obj = ro.r["cbrmb_alpha_fit"](
        pandas_to_rpy2(df.reset_index(drop=True)),
        lhs, rhs, family, engine, rnd, subject, time if time is not None else ro.NULL,
        conf_level,
    )

    coef = rpy2_to_pandas(obj.rx2("coef"))
    if family in _LOG_FAMILIES:
        coef["ratio"] = np.exp(coef["estimate"])
        coef["ratio_low"] = np.exp(coef["ci_low"])
        coef["ratio_high"] = np.exp(coef["ci_high"])
    terms_r = obj.rx2("terms")
    terms = rpy2_to_pandas(terms_r) if len(terms_r) else pd.DataFrame(
        columns=["term", "test", "stat", "df", "den_df", "pval"]
    )
    rand = rpy2_to_pandas(obj.rx2("random"))

    info_r = obj.rx2("info")
    info = {k: np.asarray(info_r.rx2(k)).ravel()[0] for k in info_r.names}
    info = {k: (bool(v) if k == "singular" and not pd.isna(v) else v) for k, v in info.items()}
    info["icc"] = np.nan
    if family in ("lognormal", "gaussian") and random == "intercept":
        sd_s = rand.loc[rand["group"] == "subject", "sd"]
        sd_e = rand.loc[rand["group"] == "Residual", "sd"]
        if len(sd_s) == 1 and len(sd_e) == 1:
            info["icc"] = float(sd_s.iloc[0] ** 2 / (sd_s.iloc[0] ** 2 + sd_e.iloc[0] ** 2))
    formula = f"{lhs} ~ {rhs}" + (f" + {rnd}" if engine != "lme" else "")
    info.update(family=family, engine=engine, formula=formula)

    warns = [str(w) for w in obj.rx2("warnings")]
    if family == "negbin" and info["theta"] > 1e3:
        warns.append(
            f"theta = {info['theta']:.3g}: no overdispersion (the data may be "
            "underdispersed); family='poisson' is the better (and conservative) choice"
        )
    if warns:
        warnings.warn(f"{response}: " + " | ".join(warns), RuntimeWarning, stacklevel=2)

    depth_meta = {"col": depth, "as": depth_as}
    if depth is not None:
        logd = np.log(pd.to_numeric(data[depth], errors="coerce").astype(float))
        depth_meta["center"] = float(logd.mean()) if offset is None else float(offset)

    return AlphaMixedFit(
        response=response, family=family, engine=engine, formula=formula,
        coef=coef, terms=terms, random=rand, info=pd.Series(info), warnings=warns,
        r_fit=obj.rx2("fit"), _r_obj=obj, _depth=depth_meta, _subject=subject,
    )


def alpha_mixed_screen(
    data: pd.DataFrame,
    responses,
    fixed: str,
    subject: str,
    **kwargs,
):
    """Fit :func:`alpha_mixed` for several alpha-diversity measures.

    ``responses`` is a list of columns (family ``"auto"`` each) or a dict
    ``{column: family}``. Remaining keyword arguments go to :func:`alpha_mixed`.

    Returns ``(coef, terms, fits)``: long ``coef`` / ``terms`` tables with a
    ``response`` column and an ``fdr`` column (Benjamini-Hochberg per term/
    coefficient across responses; the intercept is left out), and the
    ``{response: AlphaMixedFit}`` dict.
    """
    if not isinstance(responses, dict):
        responses = {r: kwargs.get("family", "auto") for r in responses}
    kwargs.pop("family", None)

    fits = {
        resp: alpha_mixed(data, resp, fixed, subject, family=fam, **kwargs)
        for resp, fam in responses.items()
    }

    def _long(attr):
        parts = []
        for resp, fit in fits.items():
            tab = getattr(fit, attr).copy()
            tab.insert(0, "family", fit.family)
            tab.insert(0, "response", resp)
            parts.append(tab)
        out = pd.concat(parts, ignore_index=True)
        out["fdr"] = np.nan
        keep = (out["term"] != "(Intercept)") & out["pval"].notna()
        for _, idx in out[keep].groupby("term").groups.items():
            out.loc[idx, "fdr"] = fdr(out.loc[idx, "pval"].to_numpy())
        cols = list(out.columns)
        cols.insert(cols.index("pval") + 1, cols.pop(cols.index("fdr")))
        return out[cols]

    return _long("coef"), _long("terms"), fits
