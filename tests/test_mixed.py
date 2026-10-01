"""Alpha-diversity mixed models (cbrmb.rbackend.mixed)."""
import numpy as np
import pandas as pd
import pytest

from cbrmb.rbackend.mixed import _prepare, _random_term, _resolve_family


# --- pure Python ---------------------------------------------------------
def test_resolve_family_auto():
    assert _resolve_family(pd.Series([3, 10, 0, np.nan]), "auto") == "negbin"
    assert _resolve_family(pd.Series([3.2, 10.5]), "auto") == "lognormal"
    assert _resolve_family(pd.Series([3, 10]), "poisson") == "poisson"
    with pytest.raises(ValueError, match="family"):
        _resolve_family(pd.Series([1.0]), "beta")


def test_random_term():
    assert _random_term("intercept", "MT-ID", None) == "(1 | `MT-ID`)"
    assert _random_term("slope", "id", "t") == "(1 + `t` | `id`)"
    assert _random_term("(1 | id) + (1 | site)", "id", None) == "(1 | id) + (1 | site)"
    with pytest.raises(ValueError, match="time="):
        _random_term("slope", "id", None)
    with pytest.raises(ValueError, match="random must"):
        _random_term("sloppy", "id", None)


def test_prepare_depth():
    df = pd.DataFrame({"y": [1, 2, 3], "id": [1, 1, 2], "n": [10.0, 100.0, 1000.0]})
    out, rhs, off = _prepare(df, "y", "id", None, "n", "covariate", "negbin")
    assert rhs == ".log_depth" and off is None
    assert np.isclose(out[".log_depth"].mean(), 0)
    assert out["id"].tolist() == ["1", "1", "2"]
    out, rhs, off = _prepare(df, "y", "id", None, "n", "offset", "negbin")
    assert rhs == "offset(.log_depth)" and np.isclose(off, np.log(100))
    with pytest.raises(ValueError, match="count families"):
        _prepare(df, "y", "id", None, "n", "offset", "lognormal")
    with pytest.raises(KeyError):
        _prepare(df, "y", "subject", None, None, "covariate", "negbin")


# --- needs lme4 / lmerTest / nlme ---------------------------------------------
def _r_ready():
    try:
        from cbrmb.rbackend import r_package
        for p in ("lme4", "lmerTest", "nlme"):
            r_package(p)
        return True
    except Exception:
        return False


rmixed = pytest.mark.skipif(not _r_ready(), reason="needs R packages lme4, lmerTest, nlme")


@pytest.fixture(scope="module")
def adf():
    rng = np.random.default_rng(1)
    nsub, nt = 40, 5
    n = nsub * nt
    subj = np.repeat([f"P{i:02d}" for i in range(nsub)], nt)
    t = np.tile(np.arange(nt, dtype=float), nsub)
    ibd = np.repeat(rng.choice([0, 1], nsub), nt)
    u0 = np.repeat(rng.normal(0, 0.3, nsub), nt)
    depth = rng.integers(8000, 40000, n)
    ld = np.log(depth) - np.log(depth).mean()
    log_h = np.log(50) - 0.2 * ibd - 0.08 * t * ibd + u0 + rng.normal(0, 0.15, n)
    mu = np.exp(np.log(130) - 0.3 * ibd - 0.1 * t * ibd + u0 + 0.3 * ld)
    rich = rng.negative_binomial(20, 20 / (20 + mu))
    return pd.DataFrame({
        "MT-ID": subj, "time": t,
        "group": pd.Categorical(np.where(ibd, "IBD", "ctrl"), ["ctrl", "IBD"]),
        "Shannon.Effective": np.exp(log_h), "Richness": rich, "Nreads": depth,
    })


@rmixed
def test_lognormal_matches_statsmodels(adf):
    import statsmodels.formula.api as smf
    from cbrmb.rbackend.mixed import alpha_mixed

    fit = alpha_mixed(adf, "Shannon.Effective", "group * time", "MT-ID")
    assert fit.family == "lognormal" and fit.engine == "lmer"
    d = adf.rename(columns={"MT-ID": "sid"}).assign(lh=lambda x: np.log(x["Shannon.Effective"]))
    sm = smf.mixedlm("lh ~ group * time", d, groups="sid").fit(reml=True)
    np.testing.assert_allclose(fit.coef["estimate"], sm.fe_params.values, atol=1e-5)
    np.testing.assert_allclose(fit.coef["se"], sm.bse_fe.values, rtol=1e-3)
    icc_sm = sm.cov_re.iloc[0, 0] / (sm.cov_re.iloc[0, 0] + sm.scale)
    assert abs(fit.info["icc"] - icc_sm) < 1e-3
    np.testing.assert_allclose(fit.coef["ratio"], np.exp(fit.coef["estimate"]))
    # marginality: only the interaction is tested
    assert fit.terms["term"].tolist() == ["group:time"]
    assert fit.terms["pval"].iloc[0] < 1e-3


@rmixed
def test_negbin_richness_with_depth(adf):
    from cbrmb.rbackend.mixed import alpha_mixed

    fit = alpha_mixed(adf, "Richness", "group * time", "MT-ID", depth="Nreads")
    assert fit.family == "negbin" and fit.engine == "glmer"
    c = fit.coef.set_index("term")
    assert abs(c.loc[".log_depth", "estimate"] - 0.3) < 0.15
    assert c.loc["groupIBD:time", "ci_low"] < -0.1 < c.loc["groupIBD:time", "ci_high"]
    assert 5 < fit.info["theta"] < 100
    assert set(fit.terms["term"]) == {".log_depth", "group:time"}
    assert (fit.terms["test"] == "LRT").all()


@rmixed
def test_slope_and_car1(adf):
    from cbrmb.rbackend.mixed import alpha_mixed

    s = alpha_mixed(adf, "Shannon.Effective", "group * time", "MT-ID",
                    random="slope", time="time")
    assert set(s.random["name"]) == {"(Intercept)", "time", "Residual"}
    c = alpha_mixed(adf, "Shannon.Effective", "group * time", "MT-ID",
                    time="time", correlation="car1")
    assert c.engine == "lme" and 0 <= c.info["car1_phi"] < 1
    est = c.coef.set_index("term").loc["groupIBD:time", "estimate"]
    assert abs(est - (-0.08)) < 0.03
    with pytest.raises(ValueError, match="lognormal / gaussian"):
        alpha_mixed(adf, "Richness", "group", "MT-ID", time="time", correlation="car1")


@rmixed
def test_predict_population(adf):
    from cbrmb.rbackend.mixed import alpha_mixed

    fit = alpha_mixed(adf, "Richness", "group * time", "MT-ID", depth="Nreads")
    nd = pd.DataFrame({"group": ["ctrl", "IBD"], "time": [0.0, 0.0]})
    pr = fit.predict(nd)
    b = fit.coef.set_index("term")["estimate"]
    np.testing.assert_allclose(pr["fit"], np.exp([b["(Intercept)"], b["(Intercept)"] + b["groupIBD"]]))
    assert (pr["fit_low"] < pr["fit"]).all() and (pr["fit"] < pr["fit_high"]).all()


@rmixed
def test_screen_fdr(adf):
    from cbrmb.rbackend.mixed import alpha_mixed_screen

    coef, terms, fits = alpha_mixed_screen(
        adf, ["Shannon.Effective", "Richness"], "group * time", "MT-ID", depth="Nreads"
    )
    assert {f.family for f in fits.values()} == {"lognormal", "negbin"}
    assert coef.loc[coef["term"] == "(Intercept)", "fdr"].isna().all()
    it = terms[terms["term"] == "group:time"]
    assert len(it) == 2 and (it["fdr"] >= it["pval"]).all()
    assert list(terms.columns).index("fdr") == list(terms.columns).index("pval") + 1
