"""Repeated-measures confounder screening (cbrmb.rbackend.permanova)."""
import numpy as np
import pandas as pd
import pytest

from cbrmb.rbackend.permanova import _reduce_to_subject, subject_variation


# --- pure Python: no R needed ---------------------------------------------
def test_subject_variation_levels():
    subject = np.repeat([f"P{i}" for i in range(6)], 3)          # 6 subjects x 3
    meta = pd.DataFrame({
        "sex":       np.repeat(["F", "M", "F", "M", "F", "M"], 3),      # between
        "timepoint": np.tile(["t0", "t1", "t2"], 6),                    # within
        "crp":       np.arange(18.0),                                   # within (numeric)
        "batch":     ["b"] * 18,                                        # constant
        "sample_id": [f"s{i}" for i in range(18)],                      # id_like
    })
    # medication turns on partway through for 2 of 6 subjects -> mixed
    med = np.array(["no"] * 18, dtype=object)
    med[[1, 2, 4, 5]] = "yes"
    meta["med"] = med

    lv = subject_variation(meta, subject)["level"].to_dict()
    assert lv["sex"] == "between"
    assert lv["timepoint"] == "within"
    assert lv["crp"] == "within"
    assert lv["batch"] == "constant"
    assert lv["sample_id"] == "id_like"
    assert lv["med"] == "mixed"


def test_subject_variation_without_subject_is_between():
    meta = pd.DataFrame({"x": [1, 2, 3, 4], "c": [1, 1, 1, 1]})
    lv = subject_variation(meta)["level"].to_dict()
    assert lv["x"] == "between"
    assert lv["c"] == "constant"


def test_reduce_to_subject_medoid_and_first():
    subject = np.array(["A", "A", "A", "B", "B"])
    # A: point 1 is the medoid (equidistant to 0 and 2); B: symmetric
    D = np.array([
        [0, 1, 2, 9, 9],
        [1, 0, 1, 9, 9],
        [2, 1, 0, 9, 9],
        [9, 9, 9, 0, 3],
        [9, 9, 9, 3, 0],
    ], dtype=float)
    vals = pd.Series(["a0", "a1", "a2", "b0", "b1"])

    Dm, vm, order = _reduce_to_subject(D, vals, subject, "medoid")
    assert list(order) == ["A", "B"]
    assert Dm.shape == (2, 2)
    assert vm.iloc[0] == "a1"                     # medoid of A

    Df, vf, _ = _reduce_to_subject(D, vals, subject, "first")
    assert vf.tolist() == ["a0", "b0"]


# --- R-backed: skip if rpy2 / R packages unavailable ---------------------
def _r_ready():
    try:
        from cbrmb.rbackend import r_package
        for pkg in ("GUniFrac", "permute", "vegan"):
            r_package(pkg)
        return True
    except Exception:
        return False


r_backend = pytest.mark.skipif(not _r_ready(), reason="needs rpy2 + R (GUniFrac, permute, vegan)")


@pytest.fixture
def rm_data():
    rng = np.random.default_rng(1)
    nsub, per = 30, 3
    subject = np.repeat([f"P{i}" for i in range(nsub)], per)
    off = rng.normal(0, 3, size=(nsub, 8))
    X = off[np.repeat(np.arange(nsub), per)] + rng.normal(0, 0.4, size=(nsub * per, 8))
    grp = np.repeat(rng.choice(["a", "b"], nsub), per)          # between, REAL
    X = X + (grp[:, None] == "b") * 1.5
    sex = np.repeat(rng.choice(["F", "M"], nsub), per)          # between, NULL
    tp = np.tile(["t0", "t1", "t2"], nsub)                      # within, NULL
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X))
    return D, subject, pd.DataFrame({"group": grp, "sex": sex, "timepoint": tp})


@r_backend
def test_free_permutation_is_anticonservative_vs_restricted(rm_data):
    from cbrmb.rbackend.permanova import test_confounder
    D, subject, meta = rm_data
    free = test_confounder(D, meta["sex"])                       # NULL, free perm
    restr = test_confounder(D, meta["sex"], subject=subject)     # NULL, subject-aware
    assert free.attrs["scheme"] == "free"
    assert restr.attrs["scheme"] == "between"
    # same F/R2, only the permutation p changes; restricted must be far less extreme
    assert restr["Pr(>F)"] > free["Pr(>F)"]
    assert restr["Pr(>F)"] > 0.15


@r_backend
def test_between_effect_detected_and_reduce_paths(rm_data):
    from cbrmb.rbackend.permanova import test_confounder
    D, subject, meta = rm_data
    auto = test_confounder(D, meta["group"], subject=subject)
    assert auto.attrs["scheme"] == "between"
    assert auto["Pr(>F)"] < 0.05
    for how in ("medoid", "first"):
        r = test_confounder(D, meta["group"], subject=subject, reduce=how)
        assert r.attrs["scheme"] == f"reduce:{how}"
        assert np.isfinite(r["Pr(>F)"])


@r_backend
def test_within_covariate_scheme(rm_data):
    from cbrmb.rbackend.permanova import test_confounder
    D, subject, meta = rm_data
    r = test_confounder(D, meta["timepoint"], subject=subject)
    assert r.attrs["scheme"] == "within"
    assert np.isfinite(r["R2"]) and np.isfinite(r["Pr(>F)"])


@r_backend
def test_screen_confounder_output_shapes(rm_data):
    from cbrmb.rbackend.permanova import screen_confounder
    D, subject, meta = rm_data

    legacy = screen_confounder(D, meta, verbose=False)
    assert list(legacy.columns) == ["var", "r2", "pval"]

    rm = screen_confounder(D, meta, subject=subject, verbose=False)
    assert list(rm.columns) == ["var", "level", "n", "scheme", "r2", "pval", "fdr"]
    assert rm["pval"].is_monotonic_increasing
    assert rm.loc[rm["var"] == "group", "level"].iloc[0] == "between"


@r_backend
def test_betadisper_runs(rm_data):
    from cbrmb.rbackend.permanova import betadisper
    D, subject, meta = rm_data
    out = betadisper(D, meta["group"], subject=subject)
    assert set(out.index) == {"F", "pval"}
    assert out.attrs["scheme"] == "between"
    assert "group_dist_to_centroid" in out.attrs


# --- mixed-type covariate coercion (_r_ready_series) --------------------------
def test_r_ready_series_leaves_numeric_and_stringifies_the_rest():
    from cbrmb.rbackend.permanova import _r_ready_series

    num = pd.Series([1.0, 2.0, np.nan, 4.0])
    pd.testing.assert_series_equal(_r_ready_series(num), num)

    mixed = pd.Series([1, "IV", 3, np.nan, "unknown"], dtype=object)
    out = _r_ready_series(mixed)
    assert out[out.notna()].map(type).eq(str).all()     # homogeneous str
    assert out.isna().tolist() == [False, False, False, True, False]
    assert out.tolist()[:3] == ["1", "IV", "3"]         # no "nan" string leaks


@r_backend
def test_screen_confounder_survives_mixed_type_column(rm_data):
    from cbrmb.rbackend.permanova import screen_confounder, test_confounder

    D, subject, meta = rm_data
    meta = meta.copy()
    # a clinical-style column: numeric codes with a couple of string escapes
    stage = np.where(np.arange(len(meta)) % 4 == 0, "n.a.", np.arange(len(meta)) % 3)
    meta["stage"] = pd.Series(stage, index=meta.index, dtype=object)

    res = test_confounder(D, meta["stage"])
    assert np.isfinite(res["R2"]) and np.isfinite(res["Pr(>F)"])

    out = screen_confounder(D, meta, verbose=False)
    assert "stage" in set(out["var"])
    assert out["pval"].notna().all()


# --- mediation decomposition ------------------------------------------------
def test_confounder_adjusted_skips_constant_target():
    # pure Python: the constant/id_like check short-circuits before any R call
    from cbrmb.rbackend.permanova import test_confounder_adjusted

    D = np.eye(6)
    target = pd.Series(["a"] * 6)                       # constant -> untestable
    adjust_for = pd.Series(["x", "y", "x", "y", "x", "y"])
    out = test_confounder_adjusted(D, target, adjust_for)
    assert np.isnan(out["R2"]) and np.isnan(out["Pr(>F)"])
    assert out.attrs["scheme"].startswith("skip:")


def test_reduce_to_subject_multi_matches_single():
    from cbrmb.rbackend.permanova import _reduce_to_subject, _reduce_to_subject_multi

    subject = np.array(["A", "A", "A", "B", "B"])
    D = np.array([
        [0, 1, 2, 9, 9],
        [1, 0, 1, 9, 9],
        [2, 1, 0, 9, 9],
        [9, 9, 9, 0, 3],
        [9, 9, 9, 3, 0],
    ], dtype=float)
    vals = pd.Series(["a0", "a1", "a2", "b0", "b1"])
    vals2 = pd.Series(["x0", "x1", "x2", "y0", "y1"])

    Dm, vm, order = _reduce_to_subject(D, vals, subject, "medoid")
    Dm2, (vm2, vm2b), order2 = _reduce_to_subject_multi(D, [vals, vals2], subject, "medoid")
    np.testing.assert_array_equal(Dm, Dm2)
    assert vm.tolist() == vm2.tolist()
    assert list(order) == list(order2)
    assert vm2b.tolist() == ["x1", "y0"]


@pytest.fixture
def mediation_data():
    """No repeated measures: `antibiotics` genuinely mediates part of `group`'s
    community effect (shifts 6/10 features), `group` also has a small
    residual/independent effect on the other 4 -> R2_indirect should be > 0."""
    rng = np.random.default_rng(7)
    nsub = 40
    grp = rng.choice(["ctrl", "case"], nsub, p=[0.5, 0.5])
    med_prob = np.where(grp == "case", 0.8, 0.2)
    med = (rng.random(nsub) < med_prob).astype(int)
    med_lab = np.where(med == 1, "yes", "no")

    off = rng.normal(0, 3, size=(nsub, 10))
    off = off + (med[:, None] == 1) * np.r_[np.full(6, 2.0), np.zeros(4)]
    off = off + (grp[:, None] == "case") * np.r_[np.zeros(6), np.full(4, 1.0)]

    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(off))
    meta = pd.DataFrame({"group": grp, "antibiotics": med_lab})
    return D, meta


@r_backend
def test_confounder_adjusted_reduces_shared_variance(mediation_data):
    from cbrmb.rbackend.permanova import test_confounder, test_confounder_adjusted

    D, meta = mediation_data
    total = test_confounder(D, meta["group"])
    direct = test_confounder_adjusted(D, meta["group"], meta["antibiotics"])
    assert total.attrs["scheme"] == "free"
    assert direct.attrs["scheme"] == "free"
    assert np.isfinite(direct["R2"])
    assert direct["R2"] < total["R2"]  # antibiotics carries a real chunk of group's total R2


@r_backend
def test_mediation_decompose_shape(mediation_data):
    from cbrmb.rbackend.permanova import mediation_decompose

    D, meta = mediation_data
    dec = mediation_decompose(D, meta["group"], meta["antibiotics"])
    for k in ("R2_total", "pval_total", "R2_direct", "pval_direct",
              "R2_indirect", "R2_indirect_frac"):
        assert k in dec.index
    assert dec["R2_indirect"] > 0


@r_backend
def test_mediation_decompose_with_subject(rm_data):
    from cbrmb.rbackend.permanova import mediation_decompose

    D, subject, meta = rm_data
    dec = mediation_decompose(D, meta["group"], meta["sex"], subject=subject)
    assert dec.attrs["scheme_total"] == "between"
    assert np.isfinite(dec["R2_indirect"])


@r_backend
def test_confounder_adjusted_between_target_within_mediator(rm_data):
    # scheme="auto" must key off target's own level, not adjust_for's -- a
    # between-subject target (group) adjusted for a within-subject mediator
    # (timepoint) used to force the within-subject-blocked model, where the
    # between-subject target has zero residual df and gets aliased out of the
    # ANOVA table entirely (KeyError on tab.loc["x"])
    from cbrmb.rbackend.permanova import test_confounder_adjusted, mediation_decompose

    D, subject, meta = rm_data
    out = test_confounder_adjusted(D, meta["group"], meta["timepoint"], subject=subject)
    assert out.attrs["scheme"] == "between"
    assert np.isfinite(out["R2"]) and np.isfinite(out["Pr(>F)"])

    dec = mediation_decompose(D, meta["group"], meta["timepoint"], subject=subject)
    assert dec.attrs["scheme_direct"] == "between"
    assert np.isfinite(dec["R2_indirect"])


@r_backend
def test_bootstrap_mediation_shapes(mediation_data):
    from cbrmb.rbackend.permanova import bootstrap_mediation

    D, meta = mediation_data
    out = bootstrap_mediation(D, meta["group"], meta["antibiotics"], n_boot=20)
    assert set(out) == {"observed", "boot", "summary"}
    assert set(out["boot"].columns) == {"R2_total", "R2_direct", "R2_indirect"}
    assert len(out["boot"]) <= 20
    assert set(out["summary"].index) == {"R2_total", "R2_direct", "R2_indirect"}
    assert (out["summary"]["n_boot_ok"] > 0).all()
    # n_perm=1 per replicate made every single replicate error out inside
    # GUniFrac::adonis3 (malformed ANOVA table for the two-covariate formula);
    # with the n_perm=2 floor almost all 20 replicates should now go through
    assert (out["summary"]["n_boot_ok"] >= 15).all()


@r_backend
def test_bootstrap_mediation_with_subject_relabels_duplicates(rm_data):
    from cbrmb.rbackend.permanova import bootstrap_mediation

    D, subject, meta = rm_data
    out = bootstrap_mediation(D, meta["group"], meta["sex"], subject=subject, n_boot=10)
    assert len(out["boot"]) <= 10
    assert out["summary"]["n_boot_ok"].min() >= 0
