"""2D embeddings of a precomputed sample-distance matrix.

:func:`calc_umap` / :func:`calc_mds` are the low-level embedders.
:func:`ordination_report` ties an embedding together with a one-covariate
PERMANOVA + betadisper on the same distance matrix and returns an
:class:`OrdinationReport` whose ``.plot()`` draws the annotated scatter.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from textwrap import dedent

import numpy as np
import pandas as pd
from sklearn.manifold import MDS

__all__ = ["calc_mds", "calc_umap", "OrdinationReport", "ordination_report"]


def _as_distance_array(dist, index):
    """Return ``(ndarray, index)``; a square DataFrame contributes its index."""
    if isinstance(dist, pd.DataFrame):
        if index is None:
            index = dist.index
        dist = dist.values
    return np.asarray(dist), index


def _embedding_frame(emb, s1, index):
    out = pd.DataFrame(emb, columns=["Dim1", "Dim2"], index=index)
    if s1 is not None:
        out["s1"] = list(s1)
    return out


def calc_mds(dist, s1=None, *, metric=False, index=None, random_state=42, **mds_kwargs):
    """Multidimensional scaling of a precomputed distance matrix.

    Parameters
    ----------
    dist : ndarray or square DataFrame
        Precomputed pairwise distances. A DataFrame's index is used to label
        the output unless ``index`` is given.
    s1 : sequence, optional
        Per-sample grouping copied into a trailing ``s1`` column.
    metric : bool
        ``False`` (default) runs non-metric MDS, as in the original helper.
    index : sequence, optional
        Row labels for the returned frame (e.g. ``adata.obs_names``).
    random_state : int
    **mds_kwargs
        Forwarded to :class:`sklearn.manifold.MDS`.

    Returns
    -------
    DataFrame with columns ``Dim1``, ``Dim2`` and optionally ``s1``.
    """
    dist, index = _as_distance_array(dist, index)
    mds = MDS(
        n_components=2,
        metric=metric,
        dissimilarity="precomputed",
        random_state=random_state,
        **mds_kwargs,
    )
    return _embedding_frame(mds.fit_transform(dist), s1, index)


def calc_umap(dist, s1=None, *, index=None, random_state=42, **umap_kwargs):
    """UMAP embedding of a precomputed distance matrix.

    Same inputs/outputs as :func:`calc_mds`. ``**umap_kwargs`` go to
    :class:`umap.UMAP` (e.g. ``min_dist=0.5``). Needs the ``umap`` extra.
    """
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "calc_umap needs the 'umap' extra: pip install 'cbrmb[umap]'"
        ) from exc

    dist, index = _as_distance_array(dist, index)
    emb = UMAP(metric="precomputed", random_state=random_state, **umap_kwargs).fit_transform(dist)
    return _embedding_frame(emb, s1, index)


# --------------------------------------------------------------------------- #
# one covariate: embedding + PERMANOVA + betadisper
# --------------------------------------------------------------------------- #
_EMBEDDERS = {"umap": calc_umap, "mds": calc_mds}


def _unpack_inputs(data, dist_key, group, subject, index):
    """Return ``(D, groups, subject_arr, index, group_name)``.

    ``data`` is either an AnnData-like object (has ``.obsp``) or a square
    distance matrix (ndarray / DataFrame). ``group`` / ``subject`` are column
    names when ``data`` is AnnData-like, otherwise sequences aligned to the rows
    of the distance matrix.
    """
    if hasattr(data, "obsp"):
        D = data.obsp[dist_key]
        D = D.toarray() if hasattr(D, "toarray") else np.asarray(D)
        if isinstance(group, str):
            group_name, groups = group, data.obs[group]
        else:
            group_name, groups = "group", pd.Series(np.asarray(group))
        if isinstance(subject, str):
            subject = data.obs[subject]
        if index is None:
            index = data.obs_names
        return D, pd.Series(np.asarray(groups)), subject, index, group_name

    D, index = _as_distance_array(data, index)
    if group is None or np.isscalar(group):
        raise TypeError(
            "when `data` is a distance matrix, `group` must be a sequence of "
            "per-sample labels"
        )
    return D, pd.Series(np.asarray(group)), subject, index, "group"


@dataclass
class OrdinationReport:
    """Result of :func:`ordination_report`.

    Attributes
    ----------
    embedding : DataFrame
        ``Dim1``, ``Dim2`` and a third column named after the covariate.
    permanova : Series or None
        ``n``, ``n_groups``, ``scheme``, ``R2``, ``pval``, ``betadisper_F``,
        ``betadisper_pval``. ``None`` when ``test=False`` or the R extra is
        missing. ``.attrs["group_dist_to_centroid"]`` holds the per-group mean
        distance to centroid from ``betadisper``.
    group : str
        Name of the covariate.
    method : str
        ``"umap"`` or ``"mds"``.
    """

    embedding: pd.DataFrame
    permanova: pd.Series | None
    group: str
    method: str

    # copy-pasteable example, printed by usage(); left unannotated on purpose so
    # the dataclass does not treat it as a field
    _EXAMPLE = """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist, squareform

    import cbrmb as mb

    # three groups pulled apart along one axis -> a real ordination signal
    rng = np.random.default_rng(0)
    groups = np.repeat(["healthy", "mild", "severe"], 20)
    X = rng.normal(size=(60, 8))
    X[:, 0] += np.repeat([0.0, 2.0, 4.0], 20)
    dist = squareform(pdist(X))                 # any precomputed sample distance

    # embedding + PERMANOVA + betadisper on the *same* matrix, one covariate
    rep = mb.ordination_report(dist, group=groups, method="mds", metric=True)
    print(rep)                                  # OrdinationReport(mds, n=60, group: R2=..., p=...)
    print(rep.permanova)                        # n, n_groups, scheme, R2, pval, betadisper_F/_pval

    fig, ax = plt.subplots(figsize=(4, 4))
    rep.plot(ax=ax, colors=["#4C72B0", "#DD8452", "#C44E52"], title="cohort")

    # real use: from an AnnData, looping the facet yourself
    #   sub = adata[adata.obs["sample_type"] == "feces"]
    #   rep = mb.ordination_report(sub, dist_key="gunifrac", group="group",
    #                              subject="patient_id",     # -> restricted permutation
    #                              method="umap", min_dist=0.5)
    #   rep.embedding    # Dim1, Dim2, <covariate>   (index = obs_names)
    """

    @classmethod
    def usage(cls) -> None:
        """Print a minimal, runnable example for :func:`ordination_report`."""
        print(f"# Typical use of ordination_report\n\n{dedent(cls._EXAMPLE).strip()}")

    def __repr__(self) -> str:
        n = len(self.embedding)
        if self.permanova is None:
            stat = "no test"
        else:
            stat = (
                f"R2={self.permanova['R2']:.3f}, p={self.permanova['pval']:.3g}, "
                f"scheme={self.permanova['scheme']}"
            )
        return f"OrdinationReport({self.method}, n={n}, {self.group}: {stat})"

    def annotation(self) -> str:
        """Two-line PERMANOVA / betadisper summary (``""`` when no test)."""
        p = self.permanova
        if p is None:
            return ""
        return (
            f"PERMANOVA R$^2$={p['R2']:.2f}, p={p['pval']:.3f}\n"
            f"betadisper p={p['betadisper_pval']:.2f}"
        )

    def plot(self, ax=None, *, colors=None, order=None, densities=True,
             legend=True, annotate=True, title=None, figsize=(4, 4),
             annotate_kwargs=None, **plot_embedding_kwargs):
        """Draw the embedding via :func:`cbrviz.plot_embedding`.

        Adds the PERMANOVA / betadisper summary as a corner annotation when
        ``annotate`` and a test was run. Creates an Axes when ``ax`` is None.
        Returns the Axes. Needs the ``plotting`` extra (matplotlib + cbrviz).
        """
        try:
            import matplotlib.pyplot as plt

            from cbrviz import plot_embedding
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "OrdinationReport.plot needs the 'plotting' extra: "
                "pip install 'cbrmb[plotting]'"
            ) from exc

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        plot_embedding(
            ax, self.embedding, colors=colors, order=order,
            densities=densities, legend=legend, title=title,
            **plot_embedding_kwargs,
        )

        if annotate and self.permanova is not None:
            akw = dict(
                x=0.02, y=0.98, s=self.annotation(), transform=ax.transAxes,
                va="top", ha="left", fontsize="x-small",
            )
            akw.update(annotate_kwargs or {})
            ax.text(**akw)

        return ax


def ordination_report(data, *, dist_key="gunifrac", group="group", subject=None,
                      method="umap", test=True, n_perm=999, seed=42,
                      index=None, **embed_kwargs):
    """UMAP/MDS embedding plus PERMANOVA + betadisper for one covariate.

    Parameters
    ----------
    data : AnnData-like or square distance matrix
        AnnData-like objects (anything with ``.obsp`` / ``.obs`` / ``.obs_names``)
        contribute the distance matrix ``data.obsp[dist_key]``; ``group`` and
        ``subject`` are then column names in ``data.obs``. A bare ndarray /
        DataFrame is used directly as the distance matrix, and ``group`` /
        ``subject`` must be passed as sequences.
    dist_key : str
        Key into ``data.obsp`` (ignored for a bare distance matrix).
    group : str or sequence
        The covariate to colour by and to test.
    subject : str, sequence or None
        Repeated-measures blocking factor. When given, the PERMANOVA and
        betadisper permutations switch to the restricted scheme classified by
        :func:`cbrmb.subject_variation` (``scheme="auto"``).
    method : {"umap", "mds"}
    test : bool
        Run PERMANOVA + betadisper (needs the ``r`` extra). On a missing extra
        a warning is emitted and ``report.permanova`` is ``None``.
    n_perm, seed : int
        Passed through to the permutation tests.
    index : sequence, optional
        Row labels for the embedding (defaults to ``data.obs_names`` when
        available, else the distance-matrix DataFrame index).
    **embed_kwargs
        Forwarded to :func:`calc_umap` / :func:`calc_mds` (e.g. ``min_dist``,
        ``n_neighbors``, ``metric=True`` for metric MDS).

    Returns
    -------
    OrdinationReport

    Examples
    --------
    ``OrdinationReport.usage()`` prints a runnable snippet.
    """
    try:
        embed = _EMBEDDERS[method]
    except KeyError:
        raise ValueError(f"method must be one of {sorted(_EMBEDDERS)}, got {method!r}")

    D, groups, subj, index, group_name = _unpack_inputs(
        data, dist_key, group, subject, index
    )

    emb = embed(D, s1=groups, index=index, **embed_kwargs)
    emb = emb.rename(columns={"s1": group_name})

    permanova = None
    if test:
        try:
            from .rbackend.permanova import betadisper, test_confounder
        except ImportError as exc:  # pragma: no cover - environment dependent
            warnings.warn(
                f"ordination_report(test=True) needs the 'r' extra; "
                f"skipping PERMANOVA ({exc})",
                stacklevel=2,
            )
        else:
            perm = test_confounder(
                D, groups, seed=seed, subject=subj, scheme="auto", n_perm=n_perm
            )
            bd = betadisper(
                D, groups, subject=subj, scheme="auto", n_perm=n_perm, seed=seed
            )
            permanova = pd.Series(
                {
                    "n": int(D.shape[0]),
                    "n_groups": int(groups.nunique()),
                    "scheme": perm.attrs.get("scheme", ""),
                    "R2": float(perm["R2"]),
                    "pval": float(perm["Pr(>F)"]),
                    "betadisper_F": float(bd["F"]),
                    "betadisper_pval": float(bd["pval"]),
                }
            )
            permanova.attrs["group_dist_to_centroid"] = bd.attrs.get(
                "group_dist_to_centroid"
            )

    return OrdinationReport(
        embedding=emb, permanova=permanova, group=group_name, method=method
    )
