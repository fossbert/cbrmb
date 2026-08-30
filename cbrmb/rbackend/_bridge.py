"""Thin numpy/pandas <-> rpy2 conversion helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import require_rpy2


def square_array(distmat):
    """Return a square distance matrix as a plain 2d ``ndarray``."""
    if isinstance(distmat, pd.DataFrame):
        distmat = distmat.values
    arr = np.asarray(distmat)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected a square distance matrix, got shape {arr.shape}")
    return arr


def numpy_to_rpy2(p_in):
    ro, numpy2ri, _, localconverter, _ = require_rpy2()
    with localconverter(ro.default_converter + numpy2ri.converter):
        return ro.conversion.py2rpy(p_in)


def rpy2_to_numpy(r_in):
    ro, numpy2ri, _, localconverter, _ = require_rpy2()
    with localconverter(ro.default_converter + numpy2ri.converter):
        return ro.conversion.rpy2py(r_in)


def pandas_to_rpy2(p_in):
    ro, _, pandas2ri, localconverter, _ = require_rpy2()
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(p_in)


def rpy2_to_pandas(r_in):
    ro, _, pandas2ri, localconverter, _ = require_rpy2()
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_in)
