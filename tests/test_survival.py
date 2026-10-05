"""cbrmb.survival: added value of features over a clinical Cox model."""
import warnings

import numpy as np
import pandas as pd
import pytest

from cbrmb.survival import (
    _fit_cox,
    concordance_index,
    cox_added_value,
    cox_screen,
    outcome_free_score,
)


def _cohort(rng, n=60, effect=0.8):
    idx = [f"p{i}" for i in range(n)]
    clin = pd.DataFrame({
        "stage": rng.binomial(1, 0.6, n).astype(float),
        "ca19_9": np.exp(rng.normal(5, 1.5, n)),
        "regimen": rng.choice(["FOLFIRINOX", "GNP"], n),
    }, index=idx)
    feat = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n)}, index=idx)
    lp = 1.0 * clin["stage"] + effect * feat["signal"]
    clin["time"] = np.round(rng.exponential(np.exp(-lp)) * 12, 1) + 0.1
    clin["event"] = (rng.random(n) < 0.75).astype(int)
    return feat, clin


# --------------------------------------------------------------------------- #
def test_concordance_by_hand():
    # times 1..4, all events; risk perfectly reversed -> C = 1
    assert concordance_index([1, 2, 3, 4], [1, 1, 1, 1], [4, 3, 2, 1]) == 1.0
    assert concordance_index([1, 2, 3, 4], [1, 1, 1, 1], [1, 2, 3, 4]) == 0.0
    # tied risk counts 1/2; censored first time is not a usable anchor
    assert concordance_index([1, 2], [1, 1], [5, 5]) == 0.5
    assert np.isnan(concordance_index([1, 2], [0, 0], [1, 2]))
    # event and censoring at the same time are comparable
    assert concordance_index([2, 2], [1, 0], [3, 1]) == 1.0
    # strata: only within-stratum pairs
    c = concordance_index([1, 2, 1, 2], [1, 1, 1, 1], [2, 1, 1, 2], strata=["a", "a", "b", "b"])
    assert c == 0.5


def test_cox_screen_finds_signal(rng):
    feat, clin = _cohort(rng, n=120)
    res = cox_screen(feat, clin, time="time", event="event", adjust="stage")
    assert res.index[0] == "signal"
    assert res.loc["signal", "hr"] > 1.5 and res.loc["signal", "pval"] < 0.01
    assert {"hr_low", "hr_high", "fdr", "epv"} <= set(res.columns)
    assert res.loc["signal", "epv"] == pytest.approx(res.loc["signal", "events"] / 2)


def test_cox_screen_categorical_adjust_and_strata(rng):
    feat, clin = _cohort(rng)
    a = cox_screen(feat, clin, time="time", event="event", adjust=["stage", "regimen"])
    b = cox_screen(feat, clin, time="time", event="event", adjust="stage", strata="regimen")
    assert a.loc["signal", "epv"] == pytest.approx(a.loc["signal", "events"] / 3)
    assert a.loc["signal", "hr"] != b.loc["signal", "hr"]


def test_cox_added_value_table(rng):
    feat, clin = _cohort(rng, n=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cox_added_value(feat, clin, time="time", event="event", base="stage", n_boot=50)
    t = res.table
    assert list(t.index) == ["signal", "noise"]
    assert t.loc["signal", "pval"] < t.loc["noise", "pval"]
    assert t.loc["signal", "delta_c"] > t.loc["noise", "delta_c"]
    assert (t["n_boot_ok"] == 50).all()
    # optimism shrinks the apparent gain for pure noise
    assert t.loc["noise", "delta_c_corr"] < t.loc["noise", "delta_c"]
    assert list(res.base.index) == ["stage"] and 0.5 < res.base_c < 1


def test_cox_added_value_null_base_and_missing(rng):
    feat, clin = _cohort(rng)
    feat.iloc[:5, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cox_added_value(feat, clin, time="time", event="event", n_boot=0)
    t = res.table
    assert t.loc["signal", "n"] == len(feat) - 5 and t.loc["noise", "n"] == len(feat)
    assert (t["c_base"] == 0.5).all()
    assert "c_full_corr" not in t


def test_duplicate_index_is_rejected(rng):
    feat, clin = _cohort(rng)
    feat = pd.concat([feat, feat.iloc[:2]])
    with pytest.raises(ValueError, match="duplicate index"):
        cox_screen(feat, clin, time="time", event="event")


def test_cox_added_value_warns_on_low_epv(rng):
    feat, clin = _cohort(rng, n=20)
    with pytest.warns(UserWarning, match="events per coefficient"):
        cox_added_value(feat, clin, time="time", event="event", base="stage", n_boot=0)


def test_cox_added_value_groups_resample_patients(rng):
    feat, clin = _cohort(rng, n=40)
    clin["patient"] = [f"pt{i // 2}" for i in range(len(clin))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cox_added_value(feat, clin, time="time", event="event", base="stage",
                              n_boot=20, groups="patient")
    assert (res.table["n_boot_ok"] > 0).all()


def test_outcome_free_score(rng):
    n = 50
    severity = rng.normal(size=n)
    clin = pd.DataFrame({
        "ca19_9": np.exp(5 + severity + rng.normal(0, .5, n)),
        "crp": np.exp(1 + severity + rng.normal(0, .5, n)),
        "albumin": 40 - 3 * severity + rng.normal(0, 1, n),
    })
    clin.iloc[0, 1] = np.nan
    sc = outcome_free_score(clin, ["ca19_9", "crp", "albumin"], log=["ca19_9", "crp"], orient="ca19_9")
    assert sc.loadings["ca19_9"] > 0 and sc.loadings["albumin"] < 0
    assert sc.n_imputed["crp"] == 1
    assert sc.score.mean() == pytest.approx(0, abs=1e-12) and sc.score.std() == pytest.approx(1)
    assert np.corrcoef(sc.score, severity)[0, 1] > 0.8
    assert sc.explained_variance > 0.6
    knn = outcome_free_score(clin, ["ca19_9", "crp", "albumin"], log=["ca19_9", "crp"],
                             orient="ca19_9", impute="knn")
    assert np.corrcoef(knn.score, sc.score)[0, 1] > 0.99
    with pytest.raises(ValueError):
        outcome_free_score(clin, ["ca19_9"], log=["crp"])


# --------------------------------------------------------------------------- #
# Equivalence with R survival, skipped without R
# --------------------------------------------------------------------------- #
def _survival_ready():
    try:
        from rpy2.robjects.packages import importr

        importr("survival")
        return True
    except Exception:
        return False


needs_r = pytest.mark.skipif(not _survival_ready(), reason="needs rpy2 + R survival")


def _to_r(name, df):
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv[name] = df


@needs_r
@pytest.mark.parametrize("strata", [False, True])
def test_cox_matches_r_survival(rng, strata):
    from rpy2 import robjects as ro

    ro.r("suppressMessages(library(survival))")
    for _ in range(8):
        n = int(rng.integers(20, 60))
        df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.binomial(1, .4, n).astype(float)})
        df["time"] = np.round(rng.exponential(np.exp(-0.8 * df.x1)) * 12) + 1   # many tied times
        df["event"] = (rng.random(n) < 0.7).astype(int)
        df["st"] = rng.choice(["A", "B"], n)
        _to_r("d", df)
        ro.r(f"f <- coxph(Surv(time, event) ~ x1 + x2{' + strata(st)' if strata else ''}, d, ties='efron')")
        s = df["st"].to_numpy() if strata else None
        fit = _fit_cox(df.time.to_numpy(float), df.event.to_numpy(), df[["x1", "x2"]].to_numpy(float), s)
        np.testing.assert_allclose(fit.params, np.array(ro.r("coef(f)")), atol=1e-7)
        np.testing.assert_allclose(fit.bse, np.sqrt(np.diag(np.array(ro.r("vcov(f)")))), atol=1e-7)
        assert fit.llf == pytest.approx(float(ro.r("f$loglik[2]")[0]), abs=1e-9)
        c = concordance_index(df.time, df.event, fit.lp, s)
        assert c == pytest.approx(float(ro.r("concordance(f)$concordance")[0]), abs=1e-12)


@needs_r
def test_bootstrap_optimism_matches_r(rng):
    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    ro.r("suppressMessages(library(survival))")
    feat, clin = _cohort(rng, n=40)
    B, seed = 60, 11
    res = cox_added_value(feat[["signal"]], clin, time="time", event="event", base="stage",
                          n_boot=B, random_state=seed, standardize=False)
    r = np.random.default_rng(seed)
    idx = np.array([r.choice(len(clin), size=len(clin), replace=True) for _ in range(B)]) + 1
    _to_r("d", clin.join(feat).reset_index(drop=True))
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["idx"] = idx
    out = np.array(ro.r("""
      cidx <- function(f, data) concordance(Surv(time, event) ~ predict(f, newdata=data, type='lp'),
                                            data=data, reverse=TRUE)$concordance
      fb <- coxph(Surv(time, event) ~ stage, d, ties='efron')
      ff <- coxph(Surv(time, event) ~ stage + signal, d, ties='efron')
      ob <- of <- c()
      for (b in seq_len(nrow(idx))) {
        db <- d[idx[b, ], ]
        if (sum(db$event) < 2) next
        gb <- coxph(Surv(time, event) ~ stage, db, ties='efron')
        gf <- coxph(Surv(time, event) ~ stage + signal, db, ties='efron')
        ob <- c(ob, cidx(gb, db) - cidx(gb, d)); of <- c(of, cidx(gf, db) - cidx(gf, d))
      }
      c((cidx(ff, d) - cidx(fb, d)) - mean(of - ob), anova(fb, ff)$Chisq[2])
    """))
    row = res.table.loc["signal"]
    assert row["delta_c_corr"] == pytest.approx(out[0], abs=1e-12)
    assert row["lr_stat"] == pytest.approx(out[1], abs=1e-9)
