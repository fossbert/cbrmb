import numpy as np
import pandas as pd
import pytest

from cbrmb.stats import (
    corr_test,
    fisher_test,
    kruskal_test,
    mwu_one_vs_rest,
    mwu_pairwise,
    mwu_test,
    wilcoxon_test,
)


def test_mwu_test_shape_and_ranking(two_group_df):
    df, ref = two_group_df
    res = mwu_test(df, ref)

    assert list(res.columns[:4]) == ["pval", "fdr", "u1", "u2"]
    assert {"A", "B"}.issubset(res.columns)
    # every feature evaluated
    assert set(res.index) == set(df.columns)
    # the shifted feature is the most significant
    assert res.index[0] == "hit"
    assert res.loc["hit", "pval"] < 0.001
    # U1 + U2 == n1 * n2
    n1 = int((ref == "A").sum())
    n2 = int((ref == "B").sum())
    np.testing.assert_allclose(res["u1"] + res["u2"], n1 * n2)
    # fdr >= pval, results sorted by pval
    assert np.all(res["fdr"].values >= res["pval"].values - 1e-12)
    assert res["pval"].is_monotonic_increasing


def test_mwu_test_requires_two_levels(three_group_df):
    df, ref = three_group_df
    with pytest.raises(ValueError, match="exactly two levels"):
        mwu_test(df, ref)


def test_kruskal_test_three_groups(three_group_df):
    df, ref = three_group_df
    res = kruskal_test(df, ref)

    assert list(res.columns[:3]) == ["pval", "fdr", "stat"]
    assert {"A", "B", "C"}.issubset(res.columns)
    assert res.index[0] == "hit"
    assert res.loc["hit", "pval"] < 0.01
    assert res["pval"].is_monotonic_increasing


def test_wilcoxon_test_runs_on_paired_data(rng):
    # regression guard: the original wilcoxon_test referenced undefined names
    n_pairs = 15
    ref = np.array(["pre"] * n_pairs + ["post"] * n_pairs)
    base = rng.normal(0, 1, n_pairs)
    df = pd.DataFrame(
        {
            "hit": np.r_[base, base + 3.0],          # consistent within-pair increase
            "null": np.r_[base, base + rng.normal(0, 0.1, n_pairs)],
        }
    )
    with pytest.warns(UserWarning, match="paired data"):
        res = wilcoxon_test(df, ref)

    assert list(res.columns[:3]) == ["stat", "pval", "fdr"]
    assert res.columns.get_loc("fdr") == res.columns.get_loc("pval") + 1
    assert {"pre", "post"}.issubset(res.columns)
    assert res.index[0] == "hit"
    assert res.loc["hit", "pval"] < 0.01


def test_corr_test(rng):
    n = 40
    ref = rng.normal(0, 1, n)
    df = pd.DataFrame({"lin": 3 * ref + rng.normal(0, 0.1, n), "noise": rng.normal(0, 1, n)})
    res = corr_test(df, ref)

    assert list(res.columns) == ["coef", "pval", "fdr"]
    assert res.loc["lin", "coef"] > 0.9
    assert res.loc["lin", "pval"] < 1e-3

    with pytest.raises(ValueError, match="method must be one of"):
        corr_test(df, ref, method="kendall")


def test_fisher_test_2x2_no_r_needed():
    # strong association in a 2x2 table
    ref = np.array(["case"] * 20 + ["ctrl"] * 20)
    feat = np.array([1] * 18 + [0] * 2 + [1] * 3 + [0] * 17)
    df = pd.DataFrame({"assoc": feat, "flat": np.r_[np.ones(20), np.ones(20)]})
    res = fisher_test(df, ref)

    assert list(res.columns) == ["pval", "fdr", "odds_ratio"]
    # 'flat' has a degenerate table (one column) -> dropped
    assert "flat" not in res.index
    assert res.loc["assoc", "pval"] < 0.01


def test_finalize_handles_empty_input(two_group_df):
    # nothing survived upstream filtering -> no columns -> no crash in fdr()
    _, ref = two_group_df
    df = pd.DataFrame(index=range(len(ref)))
    res = mwu_test(df, ref)
    assert res.empty
    assert list(res.columns[:2]) == ["pval", "fdr"]


def test_mwu_pairwise_and_one_vs_rest(three_group_df):
    df, ref = three_group_df

    pv, diff = mwu_pairwise(df, ref, verbose=False)
    assert set(pv.columns) == {"A_vs_B", "A_vs_C", "B_vs_C"}
    assert list(pv.index) == list(df.columns)
    assert diff.shape == pv.shape

    pv2, diff2 = mwu_one_vs_rest(df, ref, verbose=False)
    assert set(pv2.columns) == {"A_vs_rest", "B_vs_rest", "C_vs_rest"}
    assert pv2.loc["hit", "C_vs_rest"] < pv2.loc["null", "C_vs_rest"]
