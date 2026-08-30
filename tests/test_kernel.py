"""MiRKAT kernel association wrappers (cbrmb.rbackend.kernel)."""
import numpy as np
import pandas as pd
import pytest

from cbrmb.rbackend.kernel import _align_drop_na


# --- pure Python ---------------------------------------------------------
def test_align_drop_na_drops_incomplete_rows():
    D = np.arange(36, dtype=float).reshape(6, 6)
    D = (D + D.T) / 2
    y = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    subject = ["a", "a", "b", "b", "c", "c"]
    cov = pd.DataFrame({"x": [0.0, 1.0, 2.0, np.nan, 4.0, 5.0]})

    d2, y2, cov2, subj2 = _align_drop_na(D, y, cov, subject)
    assert d2.shape == (4, 4)                       # rows 2 and 3 dropped
    assert y2.tolist() == [1.0, 2.0, 5.0, 6.0]
    assert list(subj2) == ["a", "a", "c", "c"]
    assert cov2["x"].tolist() == [0.0, 1.0, 4.0, 5.0]


def test_align_drop_na_requires_subject():
    with pytest.raises(ValueError, match="subject="):
        _align_drop_na(np.zeros((3, 3)), [1.0, 2.0, 3.0], None, None)


# --- needs MiRKAT ------------------------------------------------------------
def _mirkat_ready():
    try:
        from cbrmb.rbackend import r_package
        r_package("MiRKAT")
        return True
    except Exception:
        return False


mirkat = pytest.mark.skipif(not _mirkat_ready(), reason="needs R package MiRKAT")


@pytest.fixture
def kdata():
    rng = np.random.default_rng(3)
    nsub, per = 28, 3
    subject = np.repeat([f"P{i}" for i in range(nsub)], per)
    idv = np.repeat(np.arange(nsub), per)
    off = rng.normal(0, 3, (nsub, 7))
    Z = off[idv] + rng.normal(0, 0.4, (nsub * per, 7))
    grp = np.repeat(rng.choice([0, 1], nsub), per)
    Z = Z + grp[:, None] * 1.3
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(Z))
    cov = pd.DataFrame({"age": rng.normal(50, 10, nsub * per),
                        "sex": np.repeat(rng.choice(["F", "M"], nsub), per)})
    return D, subject, grp, cov, rng


@mirkat
def test_glmm_mirkat_gaussian_and_cskat(kdata):
    from cbrmb.rbackend.kernel import cskat, glmm_mirkat
    D, subject, grp, cov, rng = kdata
    bmi = 25 + 2 * grp + rng.normal(0, 2, len(grp))

    out = glmm_mirkat(D, bmi, cov, subject, model="gaussian")
    assert set(out.index) == {"pval", "omnibus_p"}
    assert 0.0 <= out["pval"] <= 1.0

    cs = cskat(D, bmi, cov, subject)
    assert 0.0 <= cs["pval"] <= 1.0


@mirkat
def test_glmm_mirkat_binomial_detects_signal(kdata):
    from cbrmb.rbackend.kernel import glmm_mirkat
    D, subject, grp, cov, _ = kdata
    out = glmm_mirkat(D, grp.astype(float), cov[["age"]], subject, model="binomial")
    assert out["pval"] < 0.10          # microbiome was shifted by grp


@mirkat
def test_glmm_mirkat_null_outcome_not_significant(kdata):
    from cbrmb.rbackend.kernel import cskat
    D, subject, _, cov, rng = kdata
    y = rng.normal(0, 1, D.shape[0])
    assert cskat(D, y, cov, subject)["pval"] > 0.15


@mirkat
def test_glmm_mirkat_handles_nan(kdata):
    from cbrmb.rbackend.kernel import glmm_mirkat
    D, subject, grp, cov, rng = kdata
    y = pd.Series(25 + 2 * grp + rng.normal(0, 2, len(grp)))
    y.iloc[5] = np.nan
    out = glmm_mirkat(D, y, cov, subject, model="gaussian")
    assert np.isfinite(out["pval"])
