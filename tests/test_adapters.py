import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

from cbrmb import adapters


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"group": ["A"] * 6 + ["B"] * 6}, index=[f"s{i}" for i in range(12)])
    var = pd.DataFrame(index=[f"Zotu{i}" for i in range(5)])
    a = ad.AnnData(X=rng.poisson(3, (12, 5)).astype(float), obs=obs, var=var)
    a.layers["rel_abundance"] = a.X / a.X.sum(1, keepdims=True)
    a.obsm["tax_binning"] = pd.DataFrame(
        rng.random((12, 4)),
        index=a.obs_names,
        columns=["g__Bacteroides", "g__Prevotella", "p__Firmicutes", "p__Bacteroidota"],
    )
    a.obsm["alpha_diversity"] = pd.DataFrame(
        {"Shannon.Effective": rng.random(12)}, index=a.obs_names
    )
    return a


def test_zotus_roundtrip(adata):
    df = adapters.zotus(adata)
    assert df.shape == (12, 5)
    assert list(df.index) == list(adata.obs_names)
    assert list(df.columns) == list(adata.var_names)
    rel = adapters.zotus(adata, layer="rel_abundance")
    np.testing.assert_allclose(rel.sum(1), 1.0)


def test_taxa_level_prefix(adata):
    assert list(adapters.taxa(adata, level="g").columns) == ["g__Bacteroides", "g__Prevotella"]
    assert list(adapters.taxa(adata, level="p").columns) == ["p__Firmicutes", "p__Bacteroidota"]


def test_filter_zotu_mask_from_X_values_from_layer(adata):
    out = adapters.filter_zotu(adata, layer="rel_abundance", prevalence=0.1, frac=2, verbose=False)
    assert out.shape[1] == 5  # frac=2 keeps everything
    np.testing.assert_allclose(out.sum(1), 1.0)  # values came from the rel_abundance layer


def test_filter_tax_reports_all_ranks_as_denominator(adata, capsys):
    out = adapters.filter_tax(adata, level="g", prevalence=0.1, frac=2)
    assert list(out.columns) == ["g__Bacteroides", "g__Prevotella"]
    assert "of 4 taxa" in capsys.readouterr().out  # n_all counts every rank, not just genus


def test_top_taxa_lumps_remainder_and_strips_prefix(adata):
    out = adapters.top_taxa(adata, level="p", top=1)
    assert list(out.columns) in (["Firmicutes", "Other"], ["Bacteroidota", "Other"])
    assert out.shape == (12, 2)
    # nothing dropped: kept bin + Other == sum over all p__ columns
    all_p = adapters.taxa(adata, level="p").sum(axis=1)
    np.testing.assert_allclose(out.sum(axis=1), all_p)


def test_top_taxa_no_other_when_top_covers_all(adata):
    out = adapters.top_taxa(adata, level="p", top=5)
    assert "Other" not in out.columns and out.shape == (12, 2)
    out_raw = adapters.top_taxa(adata, level="p", top=5, strip_prefix=False)
    assert all(c.startswith("p__") for c in out_raw.columns)


def test_top_taxa_drop_remainder(adata):
    out = adapters.top_taxa(adata, level="p", top=1, other_label=None)
    assert out.shape == (12, 1)
