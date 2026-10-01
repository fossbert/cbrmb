import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

import cbrmb as mb


@pytest.fixture
def paired_meta():
    """6 patients x 2 sample types x visits 1/2; p5 lacks V2 stool, p6 has two V1 stool samples."""
    rows = []
    for p in range(1, 7):
        for st in ["feces", "saliva"]:
            for v in [1.0, 2.0]:
                if p == 5 and st == "feces" and v == 2.0:
                    continue
                rows.append({"sample_id": f"p{p}_{st}_v{int(v)}", "case_id": f"p{p}", "sample_type": st,
                             "Visit": v, "mt_id": f"MT{p}{st[0]}", "nreads": 1000})
    rows.append({"sample_id": "p6_feces_v1_rerun", "case_id": "p6", "sample_type": "feces",
                 "Visit": 1.0, "mt_id": "MT6f", "nreads": 5000})
    rows.append({"sample_id": "control", "case_id": np.nan, "sample_type": "feces",
                 "Visit": np.nan, "mt_id": "MTc", "nreads": 1000})
    return pd.DataFrame(rows).set_index("sample_id")


@pytest.fixture
def values(paired_meta, rng):
    df = pd.DataFrame(rng.poisson(20, size=(len(paired_meta), 5)).astype(float),
                      index=paired_meta.index, columns=[f"otu{i}" for i in range(5)])
    df["shannon"] = rng.normal(3, 1, len(df))
    return df


def _pairs(meta, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.find_pairs(meta, "case_id", "Visit", [1, 2], by="sample_type", **kw)


def test_find_pairs_complete_only(paired_meta):
    pairs = _pairs(paired_meta)
    assert pairs.table.index.names == ["case_id", "sample_type"]
    assert ("p5", "feces") not in pairs.table.index
    assert ("p5", "saliva") in pairs.table.index
    assert len(pairs) == 11
    assert pairs.visits == [1, 2]


def test_find_pairs_duplicates(paired_meta):
    with pytest.warns(UserWarning, match="share"):
        mb.find_pairs(paired_meta, "case_id", "Visit", [1, 2], by="sample_type")
    with pytest.raises(ValueError, match="several samples"):
        mb.find_pairs(paired_meta, "case_id", "Visit", [1, 2], by="sample_type", duplicates="error")
    pairs = _pairs(paired_meta, prefer="nreads")
    assert pairs.table.loc[("p6", "feces"), 1] == "p6_feces_v1_rerun"


def test_min_visits(paired_meta):
    pairs = _pairs(paired_meta, min_visits=1)
    assert pd.isna(pairs.table.loc[("p5", "feces"), 2])
    assert pairs.complete().shape[0] == 11


def test_subset_split_samples_long(paired_meta):
    pairs = _pairs(paired_meta)
    stool = pairs.subset(sample_type="feces")
    assert stool.table.index.names == ["case_id"]
    assert set(pairs.split()) == {"feces", "saliva"}
    assert len(stool.samples) == 10
    long = stool.long()
    assert list(long.columns) == ["case_id", "Visit"]
    assert long.index.isin(paired_meta.index).all()


def test_delta_and_wide(paired_meta, values):
    stool = _pairs(paired_meta).subset(sample_type="feces")
    d = stool.delta(values, "shannon")
    exp = values.loc["p1_feces_v2", "shannon"] - values.loc["p1_feces_v1", "shannon"]
    assert d.loc["p1"] == pytest.approx(exp)
    w = stool.wide(values, "shannon")
    assert list(w.columns) == [1, 2]
    np.testing.assert_allclose(w[2] - w[1], d.loc[w.index])
    wm = stool.wide(values, ["shannon", "otu0"])
    assert wm.columns.nlevels == 2


def test_delta_transforms(paired_meta, values):
    stool = _pairs(paired_meta).subset(sample_type="feces")
    otus = values.filter(like="otu")
    clr = np.log(otus + 1)
    clr = clr.sub(clr.mean(axis=1), axis=0)
    d = stool.delta(otus, transform="clr")
    np.testing.assert_allclose(d.loc["p1"], clr.loc["p1_feces_v2"] - clr.loc["p1_feces_v1"])
    d2 = stool.delta(otus, "otu0", transform="log2", pseudocount=0.5)
    exp = np.log2(otus.loc["p2_feces_v2", "otu0"] + 0.5) - np.log2(otus.loc["p2_feces_v1", "otu0"] + 0.5)
    assert d2.loc["p2"] == pytest.approx(exp)


def test_delta_missing_samples_warn(paired_meta, values):
    stool = _pairs(paired_meta).subset(sample_type="feces")
    with pytest.warns(UserWarning, match="missing"):
        d = stool.delta(values.drop(index="p1_feces_v2"), "shannon")
    assert "p1" not in d.index


def test_distances(paired_meta, values):
    D = pd.DataFrame(squareform(pdist(values.filter(like="otu"))), index=values.index, columns=values.index)
    stool = _pairs(paired_meta).subset(sample_type="feces")
    w = stool.within_distance(D)
    assert w.loc["p3"] == pytest.approx(D.loc["p3_feces_v1", "p3_feces_v2"])
    # ndarray + labels gives the same
    w2 = stool.within_distance(D.values, labels=D.index)
    pd.testing.assert_series_equal(w, w2)
    sub = stool.distance_matrix(D)
    assert list(sub.index) == list(stool.samples)
    # Euclidean input -> PCoA shift vectors reproduce the raw-feature deltas' distances
    shift = stool.shift_distance(D)
    raw = mb.distance_matrix(stool.delta(values.filter(like="otu")))
    np.testing.assert_allclose(shift.loc[raw.index, raw.index], raw, atol=1e-8)


def test_align_samples(paired_meta):
    covars = pd.DataFrame({
        "mt_id_stool": ["MT1f", "MT1f", "MT2f", "MT9f"],
        "visit": [1, 2, "1", 1],
        "abx": [0, 1, 1, 0],
    })
    out = mb.align_samples(covars.assign(sample_type="feces"), paired_meta,
                           on={"mt_id_stool": "mt_id", "sample_type": "sample_type", "visit": "Visit"},
                           verbose=False)
    assert list(out.index) == ["p1_feces_v1", "p1_feces_v2", "p2_feces_v1"]
    assert out.attrs["unmatched"] == [("MT9f", "feces", "1")]
    with pytest.raises(ValueError, match="several samples"):
        mb.align_samples(pd.DataFrame({"mt_id": ["MT6f"], "Visit": [1]}), paired_meta,
                         on=["mt_id", "Visit"], verbose=False)


def test_pcoa_recovers_euclidean(rng):
    X = rng.normal(size=(12, 3))
    D = pd.DataFrame(squareform(pdist(X)))
    coords = mb.pcoa(D)
    assert coords.shape[1] == 3
    np.testing.assert_allclose(squareform(pdist(coords.values)), D.values, atol=1e-10)
    assert sum(coords.attrs["explained_variance_ratio"]) == pytest.approx(1)


@pytest.mark.parametrize("method,fn", [("spearman", spearmanr), ("pearson", pearsonr)])
def test_mantel_matches_scipy(rng, method, fn):
    labels = [f"u{i}" for i in range(15)]
    X = rng.normal(size=(15, 2))
    D1 = pd.DataFrame(squareform(pdist(X)), index=labels, columns=labels)
    D2 = pd.DataFrame(squareform(pdist(X + rng.normal(scale=0.5, size=X.shape))), index=labels, columns=labels)
    iu = np.triu_indices(15, 1)
    res = mb.mantel_test(D1, D2.iloc[::-1, ::-1], method=method, n_perm=499, random_state=1)
    assert res.r == pytest.approx(fn(D1.values[iu], D2.values[iu])[0])
    assert res.n == 15
    assert res.p < 0.01
    r, p, n = res  # unpacks like the old tuple


def test_mantel_null_and_screen(rng):
    labels = [f"u{i}" for i in range(20)]
    D1 = mb.distance_matrix(pd.DataFrame(rng.normal(size=(20, 3)), index=labels))
    table = pd.DataFrame({"null": rng.normal(size=20), "hit": D1.iloc[:, 0].values}, index=labels)
    ps = [mb.mantel_test(D1, mb.distance_matrix(pd.DataFrame(rng.normal(size=(20, 2)), index=labels)),
                         n_perm=199, random_state=i).p for i in range(30)]
    assert 0.2 < np.mean(ps) < 0.8
    scr = mb.mantel_screen(D1, table, n_perm=199, random_state=0)
    assert list(scr.columns) == ["r", "pval", "fdr", "n"]
    assert scr.index[0] == "hit"


def test_within_group_distances(rng):
    labels = list("abcde")
    D = pd.DataFrame(squareform(pdist(rng.normal(size=(5, 2)))), index=labels, columns=labels)
    groups = pd.Series({"a": "g1", "b": "g1", "c": "g1", "d": "g2", "e": "g3", "zz": "g1"})
    out = mb.within_group_distances(D, groups)
    assert len(out) == 3  # g1: 3 pairs, g2/g3 singletons
    row = out[(out.sample_a == "a") & (out.sample_b == "c")]
    assert row["distance"].iloc[0] == pytest.approx(D.loc["a", "c"])


@pytest.mark.parametrize("module", ["cbrmb.paired", "cbrmb.distance"])
def test_docstring_examples(module):
    """The examples in the docstrings are part of the documentation -- keep them true."""
    import doctest
    import importlib

    res = doctest.testmod(importlib.import_module(module), optionflags=doctest.NORMALIZE_WHITESPACE)
    assert res.failed == 0


def test_readme_examples():
    """The toy walkthrough in the README is executed as well."""
    import doctest
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "README.md"
    res = doctest.testfile(str(readme), module_relative=False, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert res.failed == 0
