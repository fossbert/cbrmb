"""Pure-Python Generalized UniFrac (cbrmb.unifrac)."""
import numpy as np
import pandas as pd
import pytest

from cbrmb.unifrac import (
    calc_gunifrac,
    generalized_unifrac,
    read_newick,
    root_at_midpoint,
)

# ((A,B),C) with A+B joined by a length-1 internal branch, C hanging off at 2.
HAND_TREE = "((A:1.0,B:1.0):1.0,C:2.0);"


# --------------------------------------------------------------------------- #
# Newick parsing
# --------------------------------------------------------------------------- #
def test_read_newick_tips_lengths_and_quotes():
    root = read_newick("(('Zotu 1':0.5,Zotu2:0.25)0.97:0.1,Zotu3:0.4);")
    tips = sorted(n.name for n in _tips(root))
    assert tips == ["Zotu 1", "Zotu2", "Zotu3"]          # surrounding quotes gone
    assert {round(n.length, 3) for n in _tips(root)} == {0.5, 0.25, 0.4}


def test_read_newick_handles_comments_and_whitespace():
    root = read_newick("(\n  A:1[a comment],\n  B:2\n)root;\n")
    assert sorted(n.name for n in _tips(root)) == ["A", "B"]


def _tips(root):
    from cbrmb.unifrac import _iter_postorder

    return [n for n in _iter_postorder(root) if n.is_tip()]


# --------------------------------------------------------------------------- #
# Midpoint rooting
# --------------------------------------------------------------------------- #
def test_midpoint_balances_longest_path():
    # longest path is A---C: 1 + 1 + 5 = 7  -> root sits 3.5 from each end,
    # i.e. 1.5 up the C branch.
    root = root_at_midpoint(read_newick("((A:1.0,B:1.0):1.0,C:5.0);"))
    d = _root_to_tip(root)
    assert d["A"] == pytest.approx(3.5)
    assert d["B"] == pytest.approx(3.5)
    assert d["C"] == pytest.approx(3.5)


def _root_to_tip(root):
    out, stack = {}, [(root, 0.0)]
    while stack:
        n, acc = stack.pop()
        if n.is_tip():
            out[n.name] = acc
        for c in n.children:
            stack.append((c, acc + (c.length or 0.0)))
    return out


# --------------------------------------------------------------------------- #
# Generalized UniFrac -- values computed by hand
# --------------------------------------------------------------------------- #
def test_generalized_unifrac_hand_values():
    counts = pd.DataFrame(
        {"A": [3.0, 0.0], "B": [1.0, 1.0], "C": [0.0, 3.0]}, index=["S1", "S2"]
    )
    res = generalized_unifrac(counts, HAND_TREE, alphas=(0.0, 0.5, 1.0), midpoint=False)
    assert res[0.0].loc["S1", "S2"] == pytest.approx(0.72)
    assert res[0.5].loc["S1", "S2"] == pytest.approx(0.7390314828)
    assert res[1.0].loc["S1", "S2"] == pytest.approx(0.75)
    for m in res.values():
        assert list(m.index) == ["S1", "S2"] and list(m.columns) == ["S1", "S2"]
        np.testing.assert_allclose(np.diag(m), 0.0)
        np.testing.assert_allclose(m.values, m.values.T)


def test_disjoint_communities_are_distance_one():
    counts = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 0.0], "C": [0.0, 1.0]}, index=["S1", "S2"]
    )
    res = generalized_unifrac(counts, HAND_TREE, alphas=(0.0, 1.0), midpoint=False)
    assert res[0.0].loc["S1", "S2"] == pytest.approx(1.0)
    assert res[1.0].loc["S1", "S2"] == pytest.approx(1.0)


def test_calc_gunifrac_scalar_returns_frame_iterable_returns_dict():
    counts = pd.DataFrame(
        {"A": [3.0, 1.0], "B": [1.0, 1.0], "C": [0.0, 3.0]}, index=["S1", "S2"]
    )
    one = calc_gunifrac(counts, HAND_TREE, alpha=0.5, midpoint=False)
    assert isinstance(one, pd.DataFrame) and one.shape == (2, 2)

    many = calc_gunifrac(counts, HAND_TREE, alpha=[0.0, 0.5], midpoint=False)
    assert set(many) == {0.0, 0.5}


def test_unknown_otu_validation():
    counts = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [1.0, 1.0], "C": [0.0, 3.0], "ZZZ": [5.0, 0.0]},
        index=["S1", "S2"],
    )
    with pytest.raises(ValueError, match="not tree tips"):
        generalized_unifrac(counts, HAND_TREE, midpoint=False)

    with pytest.warns(UserWarning, match="not tree tips"):
        res = generalized_unifrac(
            counts, HAND_TREE, alphas=(0.5,), midpoint=False, validate=False
        )[0.5]
    assert np.isfinite(res.values).all()
    np.testing.assert_allclose(np.diag(res), 0.0)
    np.testing.assert_allclose(res.values, res.values.T)

    # ZZZ's mass stays in each row's total (matches GUniFrac), so the result is
    # *not* the same as first dropping ZZZ and renormalising.
    dropped = generalized_unifrac(
        counts.drop(columns="ZZZ"), HAND_TREE, alphas=(0.5,), midpoint=False
    )[0.5]
    assert res.loc["S1", "S2"] != pytest.approx(dropped.loc["S1", "S2"])


def test_tree_pruned_to_observed_otus():
    # tips D, E, F are not in the table -> must be pruned, and the near-root
    # branch they share must not leak into the normalisation.
    tree = "(((A:1.0,B:1.0):1.0,(D:1.0,E:1.0):1.0):3.0,(C:1.0,F:1.0):2.0);"
    counts = pd.DataFrame(
        {"A": [3.0, 0.0], "B": [1.0, 1.0], "C": [0.0, 3.0]}, index=["S1", "S2"]
    )
    got = generalized_unifrac(counts, tree, alphas=(0.5,), midpoint=False)
    # equivalent hand tree with D/E/F removed and singleton nodes collapsed:
    # the (A,B) node absorbs its 1.0 branch + the 3.0 stalk -> 4.0; C absorbs
    # 1.0 + 2.0 -> 3.0; the old root is the MRCA of {A,B,C} so nothing above it.
    ref_tree = "((A:1.0,B:1.0):4.0,C:3.0);"
    ref = generalized_unifrac(counts, ref_tree, alphas=(0.5,), midpoint=False)
    assert got[0.5].loc["S1", "S2"] == pytest.approx(ref[0.5].loc["S1", "S2"])


# --------------------------------------------------------------------------- #
# Equivalence with R (GUniFrac + phangorn::midpoint), skipped without R
# --------------------------------------------------------------------------- #
def _r_ready():
    try:
        from cbrmb.rbackend import r_package

        for pkg in ("GUniFrac", "ape", "phangorn"):
            r_package(pkg)
        return True
    except Exception:
        return False


needs_r = pytest.mark.skipif(not _r_ready(), reason="needs rpy2 + R (GUniFrac, ape, phangorn)")


@needs_r
def test_matches_r_gunifrac_on_random_tree():
    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    rng = np.random.default_rng(0)
    ntip, nsamp = 30, 10
    tips = [f"Zotu{i}" for i in range(ntip)]

    subtrees = list(tips)
    while len(subtrees) > 1:
        i, j = sorted(rng.choice(len(subtrees), size=2, replace=False))
        la, lb = rng.uniform(0.05, 1.0, size=2).round(6)
        merged = f"({subtrees[i]}:{la},{subtrees[j]}:{lb})"
        subtrees = [s for k, s in enumerate(subtrees) if k not in (i, j)] + [merged]
    newick = subtrees[0] + ";"

    counts = pd.DataFrame(
        rng.poisson(4, size=(nsamp, ntip)) * (rng.random((nsamp, ntip)) > 0.3),
        index=[f"S{i}" for i in range(nsamp)],
        columns=tips,
        dtype=float,
    )
    counts = counts.loc[counts.sum(axis=1) > 0]

    alphas = (0.0, 0.5, 1.0)
    py = generalized_unifrac(counts, newick, alphas=alphas, midpoint=True)

    ape = importr("ape")
    phangorn = importr("phangorn")
    gu = importr("GUniFrac")
    tree = phangorn.midpoint(ape.read_tree(text=newick))
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["ct"] = counts.to_numpy()
    ro.globalenv["ct"].rownames = ro.StrVector(list(counts.index))
    ro.globalenv["ct"].colnames = ro.StrVector(list(counts.columns))
    arr = gu.GUniFrac(ro.globalenv["ct"], tree, alpha=ro.FloatVector(alphas))[0]
    r = np.asarray(arr)

    for k, a in enumerate(alphas):
        np.testing.assert_allclose(py[a].values, r[:, :, k], atol=1e-8)
