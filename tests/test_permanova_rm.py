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
