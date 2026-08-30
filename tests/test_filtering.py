import numpy as np
import pandas as pd

from cbrmb.filtering import filter_features, filter_prevalence, filter_rel_abundance


def _df():
    # 10 samples x 4 features
    return pd.DataFrame(
        {
            "rare": [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],       # present in 1/10
            "common": [1, 0, 2, 3, 0, 1, 4, 2, 1, 5],     # present in 8/10
            "low": [0.01, 0.02, 0.0, 0.05, 0.01, 0.0, 0.02, 0.01, 0.03, 0.02],  # never > 0.25
            "high": [0.9, 0.0, 0.8, 0.1, 0.0, 0.7, 0.0, 0.6, 0.0, 0.0],         # sometimes high
        }
    )


def test_filter_prevalence_threshold():
    df = _df()
    kept = filter_prevalence(df, prevalence=0.2)
    assert list(kept.columns) == ["common", "low", "high"]  # 'rare' dropped
    kept_loose = filter_prevalence(df, prevalence=0.1)
    assert "rare" in kept_loose.columns


def test_filter_prevalence_mask_and_ndarray():
    df = _df()
    mask = filter_prevalence(df, prevalence=0.2, return_mask=True)
    assert mask.tolist() == [False, True, True, True]

    arr = df.values
    out = filter_prevalence(arr, prevalence=0.2)
    assert isinstance(out, np.ndarray)
    assert out.shape == (10, 3)


def test_filter_rel_abundance_drops_always_low():
    df = _df()
    kept = filter_rel_abundance(df, max_rel=0.25, frac=1)
    assert "low" not in kept.columns
    assert "high" in kept.columns


def test_filter_features_is_and_of_both():
    df = _df()
    kept = filter_features(df, prevalence=0.2, max_rel=0.25, frac=1)
    # 'rare' fails prevalence, 'low' fails rel-abundance -> only common + high survive
    assert list(kept.columns) == ["common", "high"]
