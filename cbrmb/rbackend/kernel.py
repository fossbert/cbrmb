"""Kernel-machine (MiRKAT) association tests for clustered / repeated-measures data.

For an outcome ``y`` (a phenotype, or a candidate covariate you are screening),
test its association with the microbiome -- represented as a kernel built from a
sample distance matrix -- while adjusting for other covariates ``X`` and a
random intercept per ``subject``:

* :func:`cskat` -- CSKAT, continuous (Gaussian) outcome, linear mixed model H0.
* :func:`glmm_mirkat` -- GLMM-MiRKAT, generalized (e.g. binary/count) outcome.

Needs the ``r`` extra plus the R package ``MiRKAT`` (and its deps
``CompQuadForm``, ``GLMMadaptive``, ``lme4``). Install into the target env, e.g.::

    mamba install -n microbiome -c conda-forge r-mirkat
    # or, inside R:  install.packages("MiRKAT")

The exact ``MiRKAT`` call is pinned to the API of MiRKAT >= 1.2; if your
installed version differs and a call fails, that is the first thing to check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import require_rpy2, r_package
from ._bridge import numpy_to_rpy2, pandas_to_rpy2, square_array

__all__ = ["cskat", "glmm_mirkat"]

_INSTALL_HINT = (
    "kernel tests need the R package 'MiRKAT'. Install it into the target env, "
    "e.g.  mamba install -n microbiome -c conda-forge r-mirkat  "
    "(or, in R:  install.packages('MiRKAT'))."
)


def _require_mirkat():
    try:
        return r_package("MiRKAT")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _distance_to_kernel(distmat):
    """MiRKAT's centered Gower kernel ``D2K`` from a square distance matrix."""
    mirkat = _require_mirkat()
    return mirkat.D2K(numpy_to_rpy2(square_array(distmat)))


def _design_matrix(covariates, n):
    """Return an R numeric model matrix for ``covariates`` (or R NULL)."""
    ro, *_ = require_rpy2()
    if covariates is None:
        return ro.NULL
    X = pd.get_dummies(pd.DataFrame(covariates).reset_index(drop=True),
                       drop_first=True).astype(float)
    if X.shape[1] == 0:
        return ro.NULL
    with_names = pandas_to_rpy2(X)
    return ro.r["as.matrix"](with_names)


def cskat(dist, y, covariates=None, subject=None, *, seed: int = 42):
    """CSKAT: association of the microbiome kernel with a continuous ``y``.

    H0 is the linear mixed model ``y ~ covariates + (1 | subject)``; the kernel
    is ``MiRKAT::D2K(dist)``. Returns a Series ``{"stat", "pval"}``.
    """
    ro, *_ = require_rpy2()
    from rpy2.robjects import Formula

    mirkat = _require_mirkat()
    dist = square_array(dist)
    y = pd.Series(np.asarray(y), dtype=float).reset_index(drop=True)
    if subject is None:
        raise ValueError("cskat needs subject= (the clustering variable)")

    K = _distance_to_kernel(dist)
    env = ro.globalenv
    env["._y"] = numpy_to_rpy2(y.to_numpy())
    env["._id"] = ro.StrVector([str(s) for s in subject])
    X = _design_matrix(covariates, len(y))
    env["._X"] = X
    fmla = Formula("._y ~ 1" if X is ro.NULL else "._y ~ ._X")

    ro.r("set.seed")(seed)
    res = mirkat.CSKAT(formula_H0=fmla, data=ro.NULL, Ks=ro.r["list"](K), id=env["._id"])
    return pd.Series({
        "stat": float(np.asarray(res.rx2("Q.adj"))[0]) if "Q.adj" in list(res.names) else np.nan,
        "pval": float(np.asarray(res.rx2("p.value"))[0]),
    })


def glmm_mirkat(dist, y, covariates=None, subject=None, *, family="gaussian", seed: int = 42):
    """GLMM-MiRKAT: association of the microbiome kernel with ``y`` under a GLMM.

    ``family`` is ``"gaussian"``, ``"binomial"`` or ``"poisson"``. Adjusts for
    ``covariates`` and a random intercept per ``subject``. Returns a Series
    ``{"pval"}`` (the omnibus p-value).
    """
    ro, *_ = require_rpy2()
    mirkat = _require_mirkat()
    dist = square_array(dist)
    if subject is None:
        raise ValueError("glmm_mirkat needs subject= (the clustering variable)")

    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    y_r = numpy_to_rpy2(y.to_numpy().astype(float))
    id_r = ro.StrVector([str(s) for s in subject])
    K = _distance_to_kernel(dist)
    X = _design_matrix(covariates, len(y))

    ro.r("set.seed")(seed)
    res = mirkat.GLMM_MiRKAT(y=y_r, X=X, Ks=ro.r["list"](K), id=id_r, family=family)
    pval = res.rx2("p_values") if "p_values" in list(res.names) else res.rx2("omnibus_p")
    return pd.Series({"pval": float(np.asarray(pval)[0])})
