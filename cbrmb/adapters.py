"""AnnData <-> DataFrame convenience layer.

These helpers pull the wide tables that the rest of ``cbrmb`` operates on out of
an :class:`anndata.AnnData` assembled the usual way:

* ``adata.X`` / ``adata.layers[...]`` -- feature (zOTU) table, samples x features
* ``adata.obsm["tax_binning"]`` -- taxonomic bins, columns prefixed ``k__``/``p__``/``g__`` ...
* ``adata.obsm["alpha_diversity"]`` -- alpha-diversity metrics per sample

Needs the ``anndata`` extra, but only these functions import it.
"""

from __future__ import annotations

import pandas as pd

from .filtering import filter_features

__all__ = [
    "zotus",
    "taxa",
    "alpha_diversity",
    "filter_zotu",
    "filter_tax",
]


def zotus(adata, layer=None):
    """Feature table as a DataFrame (samples x features)."""
    X = adata.layers[layer] if layer else adata.X
    return pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)


def taxa(adata, level="g", key="tax_binning"):
    """Taxonomic-bin table restricted to one rank.

    ``level`` matches the column-name prefix in ``adata.obsm[key]`` (e.g. ``"g"``
    for genus columns like ``"g__Bacteroides"``, ``"p"`` for phylum).
    """
    df = adata.obsm[key]
    return df.loc[:, df.columns.str.startswith(level)].copy()


def alpha_diversity(adata, key="alpha_diversity"):
    """Alpha-diversity table as stored in ``adata.obsm[key]``."""
    return adata.obsm[key]


def filter_zotu(
    adata,
    *,
    layer=None,
    max_rel=0.25,
    frac=1,
    prevalence=0.1,
    verbose=True,
):
    """Prevalence + relative-abundance filter on the zOTU table.

    The kept/dropped decision is always made on ``adata.X``; when ``layer`` is
    given the returned values are taken from that layer for the surviving
    features. Returns a DataFrame (samples x kept features).
    """
    mask = filter_features(
        zotus(adata), prevalence=prevalence, max_rel=max_rel, frac=frac, return_mask=True
    )
    if verbose:
        print(f"Kept {int(mask.sum())} out of {adata.n_vars} zotus.")
    return zotus(adata, layer=layer).loc[:, mask.values]


def filter_tax(
    adata,
    *,
    level="g",
    key="tax_binning",
    max_rel=0.25,
    frac=1,
    prevalence=0.1,
    verbose=True,
):
    """Prevalence + relative-abundance filter on one taxonomic rank.

    ``n`` reported in the verbose message is the total number of bins across all
    ranks (before restricting to ``level``), matching the original helper.
    """
    all_bins = adata.obsm[key]
    n_all = all_bins.shape[1]
    df = all_bins.loc[:, all_bins.columns.str.startswith(level)]

    mask = filter_features(
        df, prevalence=prevalence, max_rel=max_rel, frac=frac, return_mask=True
    )
    df_sub = df.loc[:, mask]
    if verbose:
        print(f"Filtered to {df_sub.shape[1]} of {n_all} taxa.")
    return df_sub.copy()
