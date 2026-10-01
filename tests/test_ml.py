"""Nested CV, cross-cohort validation, transformers, splitters (cbrmb.ml)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from cbrmb.ml import (
    CLR,
    KTSPClassifier,
    PrevalenceThreshold,
    RelativeAbundance,
    RepeatedStratifiedGroupKFold,
    cross_cohort,
    nested_cv,
    permutation_test,
)


def _cohort(rng, n, name, depth):
    y = np.array(["Control"] * (n // 2) + ["PDAC"] * (n - n // 2))
    X = rng.poisson(rng.gamma(0.5, 10, size=20), size=(n, 20)).astype(float)
    X[rng.random(X.shape) < 0.4] = 0
    X[y == "PDAC", :3] = X[y == "PDAC", :3] * 3 + rng.poisson(4, size=((y == "PDAC").sum(), 3))
    X *= depth
    return pd.DataFrame(X, index=[f"{name}{i}" for i in range(n)],
                        columns=[f"Z{j}" for j in range(20)]), y


@pytest.fixture
def cohorts(rng):
    parts = [_cohort(rng, 48, "A", 1.0), _cohort(rng, 40, "B", 3.0), _cohort(rng, 44, "C", 0.5)]
    X = pd.concat([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    cohort = np.repeat(["A", "B", "C"], [48, 40, 44])
    return X, y, cohort


def _lgr():
    return Pipeline([
        ("prev", PrevalenceThreshold(0.1)),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


# --------------------------------------------------------------------------- #
def test_prevalence_threshold_matches_filter_prevalence(rng):
    from cbrmb import filter_prevalence

    X = pd.DataFrame(rng.poisson(0.3, size=(30, 15)).astype(float))
    X.columns = [f"z{i}" for i in range(15)]
    sel = PrevalenceThreshold(0.2).fit(X)
    assert list(sel.get_feature_names_out()) == list(filter_prevalence(X, 0.2).columns)
    assert PrevalenceThreshold(3).fit(X).get_support().sum() == int(((X > 0).sum() >= 3).sum())


def test_relative_abundance_and_clr(rng):
    X = rng.poisson(5, size=(6, 4)).astype(float)
    np.testing.assert_allclose(RelativeAbundance().fit_transform(X).sum(1), 1)
    np.testing.assert_allclose(CLR().fit_transform(X).sum(1), 0, atol=1e-12)


def test_grouped_splitter_keeps_groups_together(rng):
    y = np.repeat([0, 1], 20)
    groups = np.repeat(np.arange(20), 2)
    cv = RepeatedStratifiedGroupKFold(4, 3, random_state=0)
    splits = list(cv.split(np.zeros(40), y, groups))
    assert len(splits) == 12 == cv.get_n_splits()
    for tr, te in splits:
        assert not set(groups[tr]) & set(groups[te])


# --------------------------------------------------------------------------- #
def test_nested_cv_tables(cohorts):
    X, y, cohort = cohorts
    a, b = cohort == "A", cohort == "B"
    res = nested_cv(_lgr(), {"clf__C": [0.1, 1.0]}, X[a], y[a], outer=(4, 3), inner=(3, 1),
                    external={"B": (X[b], y[b])}, metadata=pd.DataFrame({"m": np.arange(a.sum())}),
                    refit_final=True)
    assert len(res.scores) == 12 and set(res.scores["repeat"]) == {0, 1, 2}
    assert res.scores["roc_auc"].mean() > 0.8
    assert res.scores["average_precision"].notna().all()          # string labels work
    pred_test = res.predictions[res.predictions["dataset"] == "test"]
    assert len(pred_test) == 3 * a.sum()                          # each sample once per repeat
    assert set(pred_test["sample"]) == set(X.index[a])
    assert "m" in res.predictions
    assert set(res.external_scores["dataset"]) == {"B"}
    assert set(res.best_params["clf__C"]) <= {0.1, 1.0}
    assert res.weights["feature"].str.startswith("Z").all()
    assert len(res.cv_results) == 12 * 2
    assert res.final is not None and len(res.final["external_scores"]) == 3
    summ = res.summary()
    row = summ[(summ["dataset"] == "B") & (summ["metric"] == "roc_auc") & (summ["kind"] == "cv")]
    assert row["n"].item() == 12
    stab = res.feature_stability(top=5, min_freq=0.5, alpha=0.05)
    assert set(stab.loc[stab["stable"], "feature"]) & {"Z0", "Z1", "Z2"}


def test_nested_cv_reproducible_and_compare(cohorts):
    X, y, cohort = cohorts
    a = cohort == "A"
    kw = dict(outer=(4, 2), inner=(3, 1), random_state=7)
    r1 = nested_cv(_lgr(), {"clf__C": [0.1, 1.0]}, X[a], y[a], **kw)
    r2 = nested_cv(_lgr(), {"clf__C": [0.1, 1.0]}, X[a], y[a], n_jobs=2, **kw)
    pd.testing.assert_frame_equal(r1.scores, r2.scores)
    base = nested_cv(DummyClassifier(), None, X[a], y[a], **kw)
    cmp = r1.compare(base)
    assert cmp["mean_diff"] > 0.2 and cmp["n"] == 8
    other = nested_cv(DummyClassifier(), None, X[a], y[a], outer=(4, 2), random_state=8)
    with pytest.raises(ValueError):
        r1.compare(other)


def test_nested_cv_groups(cohorts):
    X, y, cohort = cohorts
    a = cohort == "A"
    groups = np.array([f"p{i // 2}" for i in range(a.sum())])
    res = nested_cv(KTSPClassifier(k=3), None, X[a], y[a], groups=groups, outer=(4, 2))
    assert len(res.scores) == 8
    for key in res._test_keys:
        te = np.array(key)
        tr = np.setdiff1d(np.arange(a.sum()), te)
        assert not set(groups[te]) & set(groups[tr])


def test_cross_cohort(cohorts):
    X, y, cohort = cohorts
    pipe = Pipeline([("prev", PrevalenceThreshold(0.1)), ("ktsp", KTSPClassifier(k=[1, 3, 5]))])
    cc = cross_cohort(pipe, None, X, y, cohort, scheme=["pairwise", "loco"], outer=(4, 1))
    m = cc.matrix()
    assert list(m.columns) == ["A", "B", "C"]
    assert list(m.index) == ["A", "B", "C", "LOCO:A", "LOCO:B", "LOCO:C"]
    assert m.loc[["A", "B", "C"], ["A", "B", "C"]].notna().all().all()
    assert m.loc["LOCO:A", "A"] > 0.7
    fin = cc.matrix(kind="final")
    assert np.allclose(np.diag(fin.loc[["A", "B", "C"]]), np.diag(m.loc[["A", "B", "C"]]))
    s = cc.summary("roc_auc")
    assert set(s["kind"]) == {"internal", "external", "final"}


def test_cross_cohort_rejects_groups_spanning_cohorts(cohorts):
    X, y, cohort = cohorts
    groups = np.zeros(len(y))
    with pytest.raises(ValueError):
        cross_cohort(KTSPClassifier(k=3), None, X, y, cohort, groups=groups, outer=(4, 1))


def test_permutation_test(cohorts):
    X, y, cohort = cohorts
    a = cohort == "A"
    pt = permutation_test(KTSPClassifier(k=3), None, X[a], y[a], n_permutations=9, outer=(4, 1))
    assert len(pt.null) == 9
    assert pt.observed > np.mean(pt.null)
    assert pt.pvalue == pytest.approx(0.1)
