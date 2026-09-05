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
    "calc_gunifrac",
    "best_clusters",
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


def calc_gunifrac(adata, tree_path, *, layer=None, alpha=0.5, backend="python", **kwargs):
    """Generalized UniFrac on ``adata``.

    Thin wrapper over :func:`zotus(adata, layer=layer) <zotus>`.

    ``backend="python"`` (default) uses the pure-Python
    :func:`cbrmb.unifrac.calc_gunifrac` -- no R needed, and it reproduces
    ``GUniFrac`` + ``phangorn::midpoint`` to machine precision.  ``backend="r"``
    keeps the old :func:`cbrmb.rbackend.unifrac.calc_gunifrac` path (needs the
    ``r`` extra).
    """
    if backend == "r":
        from .rbackend.unifrac import calc_gunifrac as _calc_gunifrac
    elif backend == "python":
        from .unifrac import calc_gunifrac as _calc_gunifrac
    else:
        raise ValueError("backend must be 'python' or 'r'")

    return _calc_gunifrac(zotus(adata, layer=layer), tree_path, alpha=alpha, **kwargs)


def best_clusters(adata, name, *, gunifrac_key="gunifrac", **filter_kwargs):
    """NbClust partition for ``adata`` (needs the ``r`` extra).

    Filters the zOTU table with :func:`filter_zotu` (``**filter_kwargs``) and
    uses ``adata.obsp[gunifrac_key]`` as the dissimilarity.
    """
    from .rbackend.clustering import best_clusters as _best_clusters

    counts = filter_zotu(adata, verbose=False, **filter_kwargs)
    return _best_clusters(counts, adata.obsp[gunifrac_key], name)
