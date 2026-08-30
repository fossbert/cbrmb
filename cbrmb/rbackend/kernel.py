"""Kernel-machine association tests for clustered / repeated-measures data (MiRKAT).

Test whether the microbiome -- as a kernel built from a sample distance matrix --
is associated with an outcome ``y`` while adjusting for covariates ``X`` and a
random intercept per ``subject``:

* :func:`glmm_mirkat` -- ``MiRKAT::GLMMMiRKAT``; Gaussian, binomial or Poisson
  outcome, permutation p-value.
* :func:`cskat` -- the Gaussian + Davies (analytic) special case; MiRKAT folds
  the former CSKAT into ``GLMMMiRKAT(model="gaussian", method="davies")``.

For confounder screening, loop candidate covariates as ``y`` (each adjusted for
whatever else you pass as ``covariates``).

Needs the ``r`` extra plus the R package ``MiRKAT`` (>= 1.2; pulls
``CompQuadForm``, ``GLMMadaptive``, ``PearsonDS``). It is a CRAN package, not on
conda-forge::

    Rscript -e 'install.packages("MiRKAT")'
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import require_rpy2, r_package
from ._bridge import numpy_to_rpy2, pandas_to_rpy2, square_array

__all__ = ["glmm_mirkat", "cskat"]

_INSTALL_HINT = (
    "kernel tests need the R package 'MiRKAT' (CRAN, not conda-forge). "
    "Install it, e.g.  Rscript -e 'install.packages(\"MiRKAT\")'."
)


def _require_mirkat():
    try:
        return r_package("MiRKAT")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _design_matrix(covariates):
    """R numeric model matrix for ``covariates`` (dummy-coded), or R NULL."""
    ro, *_ = require_rpy2()
    if covariates is None:
        return ro.NULL, None
    X = pd.get_dummies(
        pd.DataFrame(covariates).reset_index(drop=True), drop_first=True
    ).astype(float)
    if X.shape[1] == 0:
        return ro.NULL, None
    # standardize non-binary columns (kernel test is invariant to this; it just
    # keeps the adjustment GLMM well-conditioned and quiet)
    for c in X.columns:
        col = X[c]
        if col.nunique() > 2:
            sd = col.std(ddof=0)
            if sd > 0:
                X[c] = (col - col.mean()) / sd
    return ro.r["as.matrix"](pandas_to_rpy2(X)), X


def _align_drop_na(dist, y, covariates, subject):
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    n = len(y)
    dist = square_array(dist)
    if dist.shape[0] != n:
        raise ValueError(f"dist is {dist.shape}, y has length {n}")
    if subject is None:
        raise ValueError("subject= (the cluster/subject id per sample) is required")
    subject = pd.Series(np.asarray(subject)).reset_index(drop=True)

    keep = y.notna().to_numpy() & subject.notna().to_numpy()
    cov_df = None
    if covariates is not None:
        cov_df = pd.DataFrame(covariates).reset_index(drop=True)
        keep &= cov_df.notna().all(axis=1).to_numpy()

    idx = np.flatnonzero(keep)
    dist = dist[np.ix_(idx, idx)]
    y = y.iloc[idx].reset_index(drop=True)
    subject = subject.iloc[idx].reset_index(drop=True)
    if cov_df is not None:
        cov_df = cov_df.iloc[idx].reset_index(drop=True)
    return dist, y, cov_df, subject


def glmm_mirkat(
    dist,
    y,
    covariates=None,
    subject=None,
    *,
    model: str = "gaussian",
    method: str = "perm",
    nperm: int = 999,
    seed: int = 42,
):
    """``MiRKAT::GLMMMiRKAT``: kernel association of the microbiome with ``y``.

    Parameters
    ----------
    dist : ndarray or square DataFrame
        Sample distance matrix; converted to a kernel via ``MiRKAT::D2K``.
    y : array-like
        Outcome. Numeric for ``model="gaussian"``, 0/1 for ``"binomial"``,
        counts for ``"poisson"``.
    covariates : DataFrame, optional
        Adjustment covariates (categoricals are dummy-coded).
    subject : array-like
        Cluster / subject id per sample (the random intercept). Required.
    model : {"gaussian", "binomial", "poisson"}
    method : {"perm", "davies"}
        ``"davies"`` (analytic) is Gaussian-only.
    nperm, seed : int

    Returns
    -------
    Series ``{"pval", "omnibus_p"}`` (equal for a single kernel).
    """
    ro, *_ = require_rpy2()
    mirkat = _require_mirkat()

    dist, y, cov_df, subject = _align_drop_na(dist, y, covariates, subject)
    K = mirkat.D2K(numpy_to_rpy2(dist))
    Xr, _ = _design_matrix(cov_df)
    y_r = numpy_to_rpy2(y.to_numpy().astype(float))
    id_r = ro.StrVector([str(s) for s in subject])

    ro.r("set.seed")(seed)
    res = mirkat.GLMMMiRKAT(
        y=y_r, X=Xr, Ks=ro.r["list"](K), id=id_r,
        model=model, method=method, nperm=int(nperm),
    )
    names = list(res.names)
    pv = float(np.asarray(res.rx2("p_values"))[0])
    omni = float(np.asarray(res.rx2("omnibus_p"))[0]) if "omnibus_p" in names else pv
    return pd.Series({"pval": pv, "omnibus_p": omni})


def cskat(dist, y, covariates=None, subject=None, *, seed: int = 42):
    """CSKAT: Gaussian outcome with the Davies (analytic) p-value.

    Equivalent to :func:`glmm_mirkat` with ``model="gaussian",
    method="davies"`` (how MiRKAT >= 1.2 exposes the former ``CSKAT``).
    """
    return glmm_mirkat(
        dist, y, covariates, subject, model="gaussian", method="davies", seed=seed
    )
