"""Labelled distance matrices: subsetting, PCoA, Mantel tests.

Every function here takes and returns square :class:`pandas.DataFrame`
distance matrices whose index/columns carry the sample (or subject) labels,
so subsetting and aligning is done by label, never by position. Inputs can
also be an ``AnnData`` (``key=`` names the ``obsp`` slot) or a bare ndarray
with ``labels=``.

    d = mb.as_distance_frame(adata, "gunifrac")
    mb.mantel_test(shift_dist, qol_dist)          # aligns on the shared labels

The per-patient side of a Mantel analysis usually comes from
:class:`cbrmb.paired.Pairs` (``shift_distance``, ``delta_distance``); the
other side from :func:`distance_matrix` on any per-patient table:

>>> import numpy as np, pandas as pd
>>> import cbrmb as mb
>>> rng = np.random.default_rng(0)
>>> pts = [f"pt{i:02d}" for i in range(25)]
>>> qol_delta = pd.DataFrame(rng.normal(size=(25, 3)), index=pts,
...                          columns=["Fatigue", "Pain", "Diarrhoea"])
>>> mb_shift = pd.DataFrame(qol_delta.to_numpy() @ rng.normal(size=(3, 4))
...                         + rng.normal(size=(25, 4)), index=pts)
>>> shift_dist = mb.distance_matrix(mb_shift)
>>> qol_dist = mb.distance_matrix(qol_delta, standardize=True)
>>> print(mb.mantel_test(shift_dist, qol_dist, random_state=1))
Mantel r = 0.583, p = 0.001, n = 25

Which domains carry it? One Mantel test per column, with FDR:

>>> mb.mantel_screen(shift_dist, qol_delta, random_state=1).round(3)
               r   pval    fdr   n
Diarrhoea  0.457  0.001  0.003  25
Fatigue    0.239  0.008  0.012  25
Pain       0.245  0.014  0.014  25
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
from scipy.stats import rankdata

from .pvalues import fdr

__all__ = [
    "as_distance_frame",
    "subset_distance",
    "distance_matrix",
    "pcoa",
    "MantelResult",
    "mantel_test",
    "mantel_screen",
    "within_group_distances",
]


def as_distance_frame(dist, key=None, *, labels=None) -> pd.DataFrame:
    """Return ``dist`` as a labelled square DataFrame.

    Parameters
    ----------
    dist : DataFrame, AnnData or array-like
        A square DataFrame is returned as is. An ``AnnData`` contributes
        ``adata.obsp[key]`` labelled by ``obs_names`` (``key`` may be omitted
        when there is exactly one ``obsp`` entry). An ndarray needs ``labels``.
    key : str, optional
        ``obsp`` key for AnnData input.
    labels : sequence, optional
        Row/column labels for ndarray input.

    Examples
    --------
    >>> import numpy as np
    >>> import cbrmb as mb
    >>> mb.as_distance_frame(np.array([[0, 1], [1, 0]]), labels=["a", "b"])
       a  b
    a  0  1
    b  1  0

    From AnnData (the ``obsp`` key can be omitted if there is only one)::

        d = mb.as_distance_frame(adata, "gunifrac")
    """
    if isinstance(dist, pd.DataFrame):
        if dist.shape[0] != dist.shape[1]:
            raise ValueError(f"distance matrix must be square, got {dist.shape}")
        return dist
    if hasattr(dist, "obsp") and hasattr(dist, "obs_names"):
        if key is None:
            keys = list(dist.obsp.keys())
            if len(keys) != 1:
                raise ValueError(f"pass key= to pick one of adata.obsp: {keys}")
            key = keys[0]
        mat = dist.obsp[key]
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        return pd.DataFrame(np.asarray(mat), index=dist.obs_names, columns=dist.obs_names)
    mat = np.asarray(dist)
    if labels is None:
        raise ValueError("pass labels= for a bare distance array")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] != len(labels):
        raise ValueError(f"distance array of shape {mat.shape} does not match {len(labels)} labels")
    return pd.DataFrame(mat, index=labels, columns=labels)


def subset_distance(dist, samples, key=None, *, labels=None) -> pd.DataFrame:
    """Sub-matrix of ``dist`` for ``samples`` (in that order), by label.

    Raises ``KeyError`` naming the samples that are not in ``dist``.

    Replaces the ``np.where(adata.obs_names == s)`` / ``np.ix_`` dance::

        sub = mb.subset_distance(adata, stool_samples, "gunifrac")

    >>> import cbrmb as mb
    >>> meta, alpha, counts = mb.paired.example_data()
    >>> D = mb.distance_matrix(alpha, standardize=True)
    >>> mb.subset_distance(D, ["s4", "s1"]).round(2)
    sample_id    s4    s1
    sample_id
    s4         0.00  2.17
    s1         2.17  0.00
    """
    d = as_distance_frame(dist, key, labels=labels)
    samples = pd.Index(samples, name=d.index.name)
    pos = d.index.get_indexer(samples)
    if (pos < 0).any():
        missing = list(samples[pos < 0])
        raise KeyError(f"{len(missing)} sample(s) not in the distance matrix: {missing[:10]}")
    return pd.DataFrame(d.values[np.ix_(pos, pos)], index=samples, columns=samples)


def distance_matrix(df, metric="euclidean", *, standardize=False) -> pd.DataFrame:
    """Pairwise distances between the rows of ``df``.

    Rows with any NaN are dropped first. ``standardize=True`` z-scores the
    columns (ddof=0) beforehand -- use it when several features on different
    scales (e.g. QoL domains) go in together. ``metric`` is anything
    :func:`scipy.spatial.distance.pdist` accepts (``"euclidean"`` on CLR data
    is the Aitchison distance; ``"cityblock"`` on a single 0/±1 delta gives
    ``|delta_i - delta_j|``).

    Examples
    --------
    >>> import pandas as pd
    >>> import cbrmb as mb
    >>> delta = pd.DataFrame({"Pain": [10.0, -5.0, 0.0], "Fatigue": [20.0, 0.0, None]},
    ...                      index=["p1", "p2", "p3"])
    >>> mb.distance_matrix(delta)            # p3 has a NaN -> dropped
          p1    p2
    p1   0.0  25.0
    p2  25.0   0.0
    >>> mb.distance_matrix(delta["Pain"], metric="cityblock")
          p1    p2   p3
    p1   0.0  15.0  10.0
    p2  15.0   0.0   5.0
    p3  10.0   5.0   0.0
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    sub = df.dropna(how="any")
    X = sub.to_numpy(dtype=float)
    if standardize:
        sd = X.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        X = (X - X.mean(axis=0)) / sd
    dmat = ssd.squareform(ssd.pdist(X, metric=metric))
    return pd.DataFrame(dmat, index=sub.index, columns=sub.index)


def pcoa(dist, n_axes=None, key=None, *, labels=None, min_eig_frac=1e-8) -> pd.DataFrame:
    """Classical (Torgerson/Gower) principal coordinates of a distance matrix.

    Eigenvalues below ``min_eig_frac * largest`` count as zero and are dropped
    (non-Euclidean distances such as UniFrac produce slightly negative ones).

    Returns one column per kept axis (``PCo1``, ``PCo2``, ...), same index as
    ``dist``; ``.attrs["explained_variance_ratio"]`` holds each axis' share of
    the kept positive eigenvalues (as a list, so pandas can compare attrs).

    Examples
    --------
    A Euclidean distance matrix is reproduced exactly by its coordinates:

    >>> import numpy as np
    >>> import cbrmb as mb
    >>> meta, alpha, counts = mb.paired.example_data()
    >>> D = mb.distance_matrix(alpha, standardize=True)
    >>> coords = mb.pcoa(D)
    >>> list(coords.columns), [round(v, 3) for v in coords.attrs["explained_variance_ratio"]]
    (['PCo1', 'PCo2'], [0.99, 0.01])
    >>> bool(np.allclose(mb.distance_matrix(coords), D))
    True

    UniFrac from AnnData, first two axes::

        coords = mb.pcoa(adata, n_axes=2, key="gunifrac")
    """
    d = as_distance_frame(dist, key, labels=labels)
    D = d.to_numpy(dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh((B + B.T) / 2)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    keep = eigvals > min_eig_frac * eigvals[0]
    eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]
    ratio = eigvals / eigvals.sum()
    if n_axes is not None:
        eigvals, eigvecs, ratio = eigvals[:n_axes], eigvecs[:, :n_axes], ratio[:n_axes]

    out = pd.DataFrame(
        eigvecs * np.sqrt(eigvals),
        index=d.index,
        columns=[f"PCo{i + 1}" for i in range(len(eigvals))],
    )
    out.attrs["explained_variance_ratio"] = ratio.tolist()
    return out


class MantelResult(NamedTuple):
    """``(r, p, n)`` -- unpacks like a tuple."""

    r: float
    p: float
    n: int

    def __str__(self):
        return f"Mantel r = {self.r:.3f}, p = {self.p:.3g}, n = {self.n}"


def _upper(mat):
    return mat[np.triu_indices(mat.shape[0], k=1)]


def _ranked_square(D):
    """Square matrix whose upper triangle holds the ranks of ``D``'s upper triangle."""
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    R = np.zeros_like(D, dtype=float)
    R[iu] = rankdata(D[iu])
    return R + R.T


def mantel_test(dist1, dist2, method="spearman", n_perm=999, *, random_state=None,
                alternative="two-sided") -> MantelResult:
    """Mantel test between two distance matrices over the same units.

    Both matrices are reduced to their shared labels (order irrelevant); at
    least 3 are needed. Significance comes from permuting the unit labels of
    ``dist2`` (rows and columns together).

    Parameters
    ----------
    dist1, dist2 : DataFrame
        Labelled square distance matrices (e.g. from :func:`distance_matrix`
        or :meth:`cbrmb.paired.Pairs.shift_distance`).
    method : {"spearman", "pearson"}
    n_perm : int
    random_state : int or Generator, optional
    alternative : {"two-sided", "greater", "less"}

    Returns
    -------
    MantelResult
        ``(r, p, n)``; p is ``(#{perm as extreme} + 1) / (n_perm + 1)``.

    Notes
    -----
    Do not adjust for a confounder with the classical partial Mantel test --
    it is anti-conservative under pure confounding. Residualize the input
    variables against the confounder before computing the distances instead,
    then run a plain Mantel test.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import cbrmb as mb
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(20, 2)), index=[f"pt{i:02d}" for i in range(20)])
    >>> d1 = mb.distance_matrix(X)
    >>> d2 = mb.distance_matrix(X + rng.normal(scale=0.5, size=X.shape))
    >>> res = mb.mantel_test(d1, d2.iloc[::-1, ::-1], random_state=0)   # order does not matter
    >>> print(res)
    Mantel r = 0.597, p = 0.001, n = 20
    >>> r, p, n = res                                                   # tuple unpacking

    Only the shared labels are used, so matrices from different patient
    subsets can go in as they are:

    >>> mb.mantel_test(d1.iloc[:15, :15], d2.iloc[5:, 5:], random_state=0).n
    10

    Typical paired-sample use::

        shift = stool.shift_distance(adata, "gunifrac")
        mb.mantel_test(shift, stool.delta_distance(cov, "antibiotics", metric="cityblock"))
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less', got {alternative!r}")

    common = dist1.index.intersection(dist2.index)
    n = len(common)
    if n < 3:
        raise ValueError(f"only {n} shared units between dist1 and dist2 -- need at least 3")
    D1 = dist1.loc[common, common].to_numpy(dtype=float)
    D2 = dist2.loc[common, common].to_numpy(dtype=float)
    if method == "spearman":
        D1, D2 = _ranked_square(D1), _ranked_square(D2)

    # Pearson on the upper triangles. Permuting D2 only reorders its entries,
    # so the mean and norm of y stay fixed and r_perm = x_c . y_perm / const.
    x = _upper(D1)
    xc = x - x.mean()
    y = _upper(D2)
    y_mean = y.mean()
    denom = np.sqrt((xc ** 2).sum() * ((y - y_mean) ** 2).sum())
    if denom == 0:
        raise ValueError("one of the distance matrices is constant")
    r_obs = float(xc @ (y - y_mean) / denom)

    rng = np.random.default_rng(random_state)
    iu0, iu1 = np.triu_indices(n, k=1)
    perm_r = np.empty(n_perm)
    chunk = max(1, 2_000_000 // max(len(x), 1))
    for start in range(0, n_perm, chunk):
        k = min(chunk, n_perm - start)
        P = np.argsort(rng.random((k, n)), axis=1)
        Y = D2[P[:, iu0], P[:, iu1]]
        perm_r[start:start + k] = (Y - y_mean) @ xc / denom

    if alternative == "two-sided":
        hits = np.sum(np.abs(perm_r) >= abs(r_obs) - 1e-12)
    elif alternative == "greater":
        hits = np.sum(perm_r >= r_obs - 1e-12)
    else:
        hits = np.sum(perm_r <= r_obs + 1e-12)
    return MantelResult(r_obs, float((hits + 1) / (n_perm + 1)), n)


def mantel_screen(dist, table, *, metric="euclidean", standardize=False, method="spearman",
                  n_perm=999, random_state=None, alternative="two-sided") -> pd.DataFrame:
    """One Mantel test of ``dist`` against each column of ``table``.

    Each column is turned into a unit x unit distance matrix with
    :func:`distance_matrix` (NaN rows dropped per column), then tested against
    ``dist`` on the shared units. Returns ``r``, ``pval``, ``fdr``, ``n`` per
    column, sorted by ``pval``.

    Replaces the loop over QoL domains::

        mb.mantel_screen(stool.shift_distance(adata, "gunifrac"), qol_delta)

    See the module docstring for a runnable example.
    """
    if isinstance(table, pd.Series):
        table = table.to_frame()
    rows = {}
    for col in table.columns:
        d = distance_matrix(table[[col]], metric=metric, standardize=standardize)
        rows[col] = mantel_test(dist, d, method=method, n_perm=n_perm,
                                random_state=random_state, alternative=alternative)
    out = pd.DataFrame(rows, index=["r", "pval", "n"]).T
    out["n"] = out["n"].astype(int)
    out.insert(2, "fdr", fdr(out["pval"].to_numpy(dtype=float)))
    return out.sort_values("pval")


def within_group_distances(dist, groups, key=None, *, labels=None) -> pd.DataFrame:
    """All pairwise distances between samples of the same group, long format.

    ``groups`` is a Series mapping sample label -> group (e.g. patient);
    samples missing from ``dist`` are dropped. Returns columns ``group``,
    ``sample_a``, ``sample_b``, ``distance`` -- one row per unordered pair.

    For units with many samples (dense time series) rather than pre/post pairs,
    e.g. within-patient UniFrac spread:

    >>> import cbrmb as mb
    >>> meta, alpha, counts = mb.paired.example_data()
    >>> D = mb.distance_matrix(alpha, standardize=True)
    >>> stool = meta[meta["sample_type"] == "feces"]
    >>> out = mb.within_group_distances(D, stool["case_id"])
    >>> out.round(2)
      group sample_a sample_b  distance
    0    p1       s1       s2      0.88
    1    p2       s3       s4      0.57
    >>> out.groupby("group")["distance"].median().round(2).to_dict()
    {'p1': 0.88, 'p2': 0.57}

    With AnnData: ``mb.within_group_distances(adata, adata.obs["case_id"], "gunifrac")``.
    """
    d = as_distance_frame(dist, key, labels=labels)
    groups = pd.Series(groups).dropna()
    groups = groups[groups.index.isin(d.index)]
    pos = pd.Series(d.index.get_indexer(groups.index), index=groups.index)
    D = d.to_numpy()
    rows = []
    for g, members in groups.groupby(groups, sort=False):
        idx = pos.loc[members.index].to_numpy()
        a, b = np.triu_indices(len(idx), k=1)
        rows.append(pd.DataFrame({
            "group": g,
            "sample_a": members.index[a],
            "sample_b": members.index[b],
            "distance": D[idx[a], idx[b]],
        }))
    if not rows:
        return pd.DataFrame(columns=["group", "sample_a", "sample_b", "distance"])
    return pd.concat(rows, ignore_index=True)
