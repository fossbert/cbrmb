"""k-TSP (cbrmb.ml.ktsp): behaviour + byte equivalence with switchBox."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cbrmb.ml import KTSP, KTSPClassifier, PrevalenceThreshold, tsp_scores, wilcoxon_filter


def _toy(rng, n=40, m=12, ties=False):
    y = np.array(["case"] * (n // 2) + ["ctrl"] * (n - n // 2))
    if ties:
        X = rng.poisson(rng.gamma(0.5, 4, size=m), size=(n, m)).astype(float)
        X[rng.random((n, m)) < 0.5] = 0
        X[y == "ctrl", 0] += 10          # f0 high in ctrl
    else:
        X = rng.normal(size=(n, m))
        X[y == "ctrl", 0] += 3           # f0 > f1 in ctrl ...
        X[y == "case", 1] += 3           # ... and f1 > f0 in case
    cols = [f"f{i}" for i in range(m)]
    return pd.DataFrame(X, columns=cols), y


def test_top_pair_orientation_votes_for_second_class(rng):
    X, y = _toy(rng)
    est = KTSP(k=1).fit(X, y)
    pairs = est.get_pairs()
    assert (pairs.loc[0, "a"], pairs.loc[0, "b"]) == ("f0", "f1")   # f0 > f1 votes "ctrl"
    clf = KTSPClassifier(k=1).fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.9
    assert list(clf.classes_) == ["case", "ctrl"]


def test_scores_antisymmetric_and_disjoint(rng):
    X, y = _toy(rng, ties=True)
    s, tv = tsp_scores(X.values, y)
    np.testing.assert_allclose(s, -s.T, atol=1e-12)
    est = KTSP(k=5).fit(X, y)
    flat = est.pairs_.ravel()
    assert len(set(flat)) == len(flat)
    assert np.all(np.diff(est.scores_) <= 1e-12)


def test_transform_names_and_index_in_pipeline(rng):
    X, y = _toy(rng, ties=True)
    pipe = Pipeline([("prev", PrevalenceThreshold(0.1)), ("ktsp", KTSP(k=3, handle_ties=True)),
                     ("clf", LogisticRegression())]).set_output(transform="pandas")
    pipe.fit(X, y)
    names = pipe[:-1].get_feature_names_out()
    assert len(names) == 3 and all(n.startswith("f") for n in names)
    Z = pipe[:-1].transform(X.iloc[:5])
    assert list(Z.index) == list(X.index[:5])
    assert set(np.unique(Z.values)) <= {0.0, 1.0}


def test_k_range_selection_and_clone(rng):
    X, y = _toy(rng)
    est = KTSP(k=[1, 3, 5]).fit(X, y)
    assert est.k_ in (1, 3, 5)
    assert clone(est).get_params()["k"] == [1, 3, 5]


def test_classifier_proba_and_decision(rng):
    X, y = _toy(rng)
    clf = KTSPClassifier(k=3).fit(X, y)
    p = clf.predict_proba(X)
    np.testing.assert_allclose(p.sum(1), 1)
    np.testing.assert_allclose(clf.decision_function(X), p[:, 1] * 3 - 1.5)


def test_wilcoxon_filter_keeps_shifted_features(rng):
    X, y = _toy(rng, m=30)
    keep = wilcoxon_filter(X.values, y, 6)
    assert {0, 1} <= set(keep.tolist())


def test_needs_two_classes(rng):
    X, _ = _toy(rng)
    with pytest.raises(ValueError):
        KTSP().fit(X, np.array(["a"] * len(X)))


# --------------------------------------------------------------------------- #
# Equivalence with R switchBox, skipped without R
# --------------------------------------------------------------------------- #
def _swb_ready():
    try:
        from rpy2.robjects.packages import importr

        importr("switchBox")
        return True
    except Exception:
        return False


needs_swb = pytest.mark.skipif(not _swb_ready(), reason="needs rpy2 + R switchBox")


def _r_train(X, y, krange, handle_ties, prefilter):
    from rpy2 import robjects as ro
    from rpy2.robjects.packages import importr

    swb = importr("switchBox")
    M = ro.r.matrix(ro.FloatVector(X.T.ravel(order="F")), nrow=X.shape[1])
    M = ro.r["rownames<-"](M, ro.StrVector([f"f{i}" for i in range(X.shape[1])]))
    kw = {"FilterFunc": ro.NULL}
    if prefilter is not None:
        kw = {"FilterFunc": swb.SWAP_Filter_Wilcoxon, "featureNo": prefilter}
    res = swb.SWAP_Train_KTSP(M, ro.r.factor(ro.StrVector(y)), krange=ro.IntVector(krange),
                              handleTies=handle_ties, **kw)
    tsps = np.array(ro.r["as.vector"](res.rx2("TSPs")), dtype=object).reshape(2, -1).T
    comps = swb.SWAP_KTSP_Statistics(M, res).rx2("comparisons")
    comps = np.array(ro.r["as.vector"](comps)).reshape(X.shape[0], -1, order="F")
    # switchBox pads with self-pairs / NA when disjoint pairs run out; we stop instead
    keep = [i for i, (a, b) in enumerate(tsps) if str(a) != "NA" and a != b]
    pairs = [(int(tsps[i, 0][1:]), int(tsps[i, 1][1:])) for i in keep]
    tie = list(ro.r["as.character"](res.rx2("tieVote")))
    return pairs, np.array(res.rx2("score"))[keep], [tie[i] for i in keep], comps[:, keep]


@needs_swb
@pytest.mark.parametrize("seed", range(4))
def test_matches_switchbox(seed):
    rng = np.random.default_rng(seed)
    lab = {0: "both", 1: "case", 2: "ctrl"}
    for trial in range(12):
        n, m = int(rng.integers(20, 70)), int(rng.integers(8, 50))
        y = np.array(["case"] * (n // 2) + ["ctrl"] * (n - n // 2))
        rng.shuffle(y)
        continuous = trial % 3 == 0
        if continuous:
            X = rng.normal(size=(n, m))
            X[y == "case", :3] += 1
        else:
            X = rng.poisson(rng.gamma(0.5, 4, size=m), size=(n, m)).astype(float)
            X[rng.random((n, m)) < 0.5] = 0
            X[y == "case", :4] *= 3
        handle = (not continuous) and bool(trial % 2)
        prefilter = 20 if trial % 4 == 3 else None
        # k selection deliberately differs from switchBox on tied data (module docstring)
        # (k <= n_features/2: switchBox itself fails on NA-padded classifiers)
        kmax = min(9, m // 2, prefilter // 2 if prefilter else m)
        krange = list(range(2, kmax + 1)) if continuous else [int(rng.integers(2, kmax + 1))]
        r_pairs, r_scores, r_ties, r_comps = _r_train(X, y, krange, handle, prefilter)
        est = KTSP(k=krange if len(krange) > 1 else krange[0], handle_ties=handle,
                   prefilter=prefilter).fit(X, y)
        assert [tuple(map(int, p)) for p in est.pairs_] == r_pairs
        assert np.array_equal(est.scores_, r_scores)
        assert [lab[t] for t in est.tie_votes_] == r_ties
        assert np.array_equal(est.transform(X).astype(bool), r_comps.astype(bool))
