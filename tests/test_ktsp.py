"""cbrmb.ml re-exports k-TSP from ktspy (the implementation and its switchBox tests live there)."""
import numpy as np
from sklearn.pipeline import Pipeline

import ktspy
from cbrmb.ml import KTSP, KTSPClassifier, PrevalenceThreshold, tsp_scores, wilcoxon_filter


def test_reexports_are_the_ktspy_objects():
    assert KTSP is ktspy.KTSP and KTSPClassifier is ktspy.KTSPClassifier
    assert tsp_scores is ktspy.tsp_scores and wilcoxon_filter is ktspy.wilcoxon_filter


def test_ktsp_in_a_cbrmb_pipeline(rng):
    y = np.array(["case"] * 20 + ["ctrl"] * 20)
    X = rng.poisson(5, size=(40, 12)).astype(float)
    X[y == "ctrl", 0] += 20                          # f0 > f1 in ctrl ...
    X[y == "case", 1] += 20                          # ... f1 > f0 in case
    pipe = Pipeline([("prev", PrevalenceThreshold(0.1)), ("ktsp", KTSPClassifier(k=1))]).fit(X, y)
    assert (pipe.predict(X) == y).mean() > 0.9
