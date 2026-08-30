import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from cbrmb.ordination import calc_mds, calc_umap


@pytest.fixture
def dist(rng):
    pts = rng.normal(size=(15, 4))
    labels = [f"s{i}" for i in range(15)]
    return squareform(pdist(pts)), labels


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
