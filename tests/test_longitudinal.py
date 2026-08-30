import numpy as np
import pandas as pd
import pytest

from cbrmb.longitudinal import (
    calc_jsd,
    david_recipe,
    rel_abundance,
    top_otu_indices,
    weighted_median,
)


@pytest.fixture
def counts(rng):
    # 12 timepoints x 8 OTUs, a few dominant taxa
    base = rng.poisson(np.array([200, 150, 80, 40, 10, 5, 3, 1]), size=(12, 8)).astype(float)
    idx = [f"t{i}" for i in range(12)]
    cols = [f"Zotu{i}" for i in range(8)]
    return pd.DataFrame(base + 1.0, index=idx, columns=cols)


def test_rel_abundance_axes():
    m = np.array([[1.0, 3.0], [2.0, 2.0]])
    np.testing.assert_allclose(rel_abundance(m, axis=1).sum(1), 1.0)
    np.testing.assert_allclose(rel_abundance(m, axis=0).sum(0), 1.0)
    with pytest.raises(ValueError):
        rel_abundance(m, axis=2)


def test_calc_jsd_symmetric_zero_diagonal(counts):
    jsd = calc_jsd(rel_abundance(counts.values))
    assert jsd.shape == (12, 12)
    np.testing.assert_allclose(jsd, jsd.T)
    np.testing.assert_allclose(np.diag(jsd), 0.0, atol=1e-12)


def test_top_otu_indices_picks_dominant():
    # cumulative fractions: 0.5, 0.8, 0.95, 0.99, 1.0 -> pivot rank is index 2
    # (0.15); the strict ">" keeps only the OTUs above it.
    m = np.array([[100, 60, 30, 8, 2]] * 5, dtype=float)
    assert set(top_otu_indices(m, top_frac=0.9).tolist()) == {0, 1}


def test_weighted_median_matches_unweighted_with_equal_weights():
    data = [1, 2, 3, 4, 100]
    assert weighted_median(data, weights=[1, 1, 1, 1, 1]) == np.median(data)
    assert weighted_median(data) == np.median(data)


def test_david_recipe_shape_and_per_row_constant_scaling(counts):
    out = david_recipe(counts, random_state=0)

    assert list(out.index) == list(counts.index)
    assert list(out.columns) == list(counts.columns)
    assert np.isfinite(out.values).all()
    assert (out.values > 0).all()

    # the recipe scales each row by a single constant in linear space, so
    # within-row ratios must match those of (rel_abundance + offset)
    rel = rel_abundance(counts.values) + 1e-5
    ratio_in = rel / rel[:, [0]]
    ratio_out = out.values / out.values[:, [0]]
    np.testing.assert_allclose(ratio_out, ratio_in, rtol=1e-6)


def test_david_recipe_random_state_reproducible(counts):
    a = david_recipe(counts, random_state=42)
    b = david_recipe(counts, random_state=42)
    pd.testing.assert_frame_equal(a, b)
