import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from cbrmb.ordination import calc_mds, calc_umap, ordination_report


def _r_ready():
    try:
        from cbrmb.rbackend import r_package

        for pkg in ("GUniFrac", "permute", "vegan"):
            r_package(pkg)
        return True
    except Exception:
        return False


r_backend = pytest.mark.skipif(
    not _r_ready(), reason="needs rpy2 + R (GUniFrac, permute, vegan)"
)


@pytest.fixture
def dist(rng):
    pts = rng.normal(size=(15, 4))
    labels = [f"s{i}" for i in range(15)]
    return squareform(pdist(pts)), labels


@pytest.fixture
def three_group_dist(rng):
    """45 samples, 3 groups separated along one axis; returns (D, index, groups)."""
    grp = np.repeat(["A", "B", "C"], 15)
    X = rng.normal(size=(45, 8))
    X[:, 0] += np.repeat([0.0, 2.5, 5.0], 15)
    idx = [f"s{i}" for i in range(45)]
    return squareform(pdist(X)), idx, grp


def test_calc_mds_shape_index_and_determinism(dist):
    d, labels = dist
    out = calc_mds(d, index=labels)
    assert list(out.columns) == ["Dim1", "Dim2"]
    assert list(out.index) == labels
    pd.testing.assert_frame_equal(out, calc_mds(d, index=labels))


def test_calc_mds_accepts_dataframe_and_s1(dist):
    d, labels = dist
    df = pd.DataFrame(d, index=labels, columns=labels)
    groups = ["A"] * 7 + ["B"] * 8
    out = calc_mds(df, s1=groups)
    assert list(out.index) == labels           # index taken from the frame
    assert list(out.columns) == ["Dim1", "Dim2", "s1"]
    assert out["s1"].tolist() == groups


def test_calc_umap_smoke(dist):
    pytest.importorskip("umap")
    d, labels = dist
    out = calc_umap(d, index=labels, n_neighbors=5, min_dist=0.3)
    assert out.shape == (15, 2)
    assert list(out.index) == labels


# --- ordination_report --------------------------------------------------------
def test_ordination_report_bare_matrix_no_test(three_group_dist):
    D, idx, grp = three_group_dist
    rep = ordination_report(
        pd.DataFrame(D, index=idx, columns=idx),
        group=grp, method="mds", test=False, metric=True,
    )
    assert list(rep.embedding.columns) == ["Dim1", "Dim2", "group"]
    assert list(rep.embedding.index) == idx
    assert rep.embedding["group"].tolist() == list(grp)
    assert rep.permanova is None
    assert rep.annotation() == ""
    assert "no test" in repr(rep)


def test_ordination_report_bad_method(three_group_dist):
    D, idx, grp = three_group_dist
    with pytest.raises(ValueError, match="method must be one of"):
        ordination_report(D, group=grp, method="tsne", test=False)


def test_ordination_report_bare_matrix_needs_group_sequence(three_group_dist):
    D, _, _ = three_group_dist
    with pytest.raises(TypeError, match="group` must be a sequence"):
        ordination_report(D, method="mds", test=False)


def test_ordination_report_anndata_path_names_the_covariate(three_group_dist):
    ad = pytest.importorskip("anndata")
    D, idx, grp = three_group_dist
    adata = ad.AnnData(
        X=np.zeros((45, 2)),
        obs=pd.DataFrame({"arm": grp}, index=idx),
    )
    adata.obsp["gunifrac"] = D
    rep = ordination_report(adata, dist_key="gunifrac", group="arm",
                            method="mds", test=False, metric=True)
    assert list(rep.embedding.columns) == ["Dim1", "Dim2", "arm"]
    assert list(rep.embedding.index) == idx
    assert rep.group == "arm"


def test_ordination_report_missing_r_extra_warns(monkeypatch, three_group_dist):
    import sys

    D, idx, grp = three_group_dist
    # make `from cbrmb.rbackend.permanova import ...` raise ImportError
    monkeypatch.setitem(sys.modules, "cbrmb.rbackend.permanova", None)

    with pytest.warns(UserWarning, match="needs the 'r' extra"):
        rep = ordination_report(D, group=grp, method="mds", test=True, metric=True)
    assert rep.permanova is None


@r_backend
def test_ordination_report_permanova_and_betadisper(three_group_dist):
    D, idx, grp = three_group_dist
    rep = ordination_report(
        pd.DataFrame(D, index=idx, columns=idx),
        group=grp, method="mds", n_perm=199, seed=1, metric=True,
    )
    p = rep.permanova
    assert set(p.index) == {
        "n", "n_groups", "scheme", "R2", "pval",
        "betadisper_F", "betadisper_pval",
    }
    assert p["n"] == 45 and p["n_groups"] == 3
    assert p["scheme"] == "free"
    assert 0.0 <= p["R2"] <= 1.0
    assert 0.0 < p["pval"] <= 1.0
    assert p["R2"] > 0.1  # the groups are separated by construction
    assert "R$^2$" in rep.annotation()


@r_backend
def test_ordination_report_subject_switches_scheme(rng):
    # 12 subjects x 3 timepoints; group is constant within subject -> "between"
    subject = np.repeat([f"P{i}" for i in range(12)], 3)
    grp = np.repeat(np.where(np.arange(12) % 2 == 0, "ctrl", "case"), 3)
    X = rng.normal(size=(36, 6))
    X[:, 0] += (grp == "case") * 2.0
    D = squareform(pdist(X))
    rep = ordination_report(D, group=grp, subject=subject, method="mds",
                            n_perm=199, seed=1, metric=True)
    assert rep.permanova["scheme"] == "between"


@r_backend
def test_ordination_report_plot_annotates(three_group_dist):
    pytest.importorskip("cbrviz")
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    D, idx, grp = three_group_dist
    rep = ordination_report(D, group=grp, method="mds", n_perm=199, metric=True)
    fig, ax = plt.subplots()
    out = rep.plot(ax=ax, colors=["#4C72B0", "#DD8452", "#55A868"])
    assert out is ax
    assert ax.get_legend() is not None
    texts = [t.get_text() for t in ax.texts]
    assert any("PERMANOVA" in t for t in texts)
    plt.close(fig)
