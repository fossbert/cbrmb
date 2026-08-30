import numpy as np
from statsmodels.stats.multitest import multipletests

from cbrmb.pvalues import cut_p, fdr


def test_cut_p_boundaries():
    assert cut_p(0.0005) == "***"
    assert cut_p(0.005) == "**"
    assert cut_p(0.03) == "*"
    assert cut_p(0.07) == "0.07"
    assert cut_p(0.5) == "ns"
    # exact boundaries fall to the looser bucket
    assert cut_p(0.001) == "**"
    assert cut_p(0.05) == "0.05"


def test_fdr_matches_statsmodels_and_preserves_order():
    pvals = [0.001, 0.6, 0.04, 0.2, 0.009]
    got = fdr(pvals)
    expected = multipletests(pvals, method="fdr_bh")[1]
    np.testing.assert_allclose(got, expected)
    assert len(got) == len(pvals)
    assert np.all(got >= np.array(pvals) - 1e-12)
