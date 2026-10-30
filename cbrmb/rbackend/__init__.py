"""Optional R-backed functionality (UniFrac, PERMANOVA/adonis, NbClust, r x c Fisher).

Everything here needs the ``r`` extra (``pip install cbrmb[r]``) plus the
relevant R packages. Imports are deferred to call time so that ``import cbrmb``
works without R installed.
"""

from __future__ import annotations

import functools

_R_EXTRA_HINT = (
    "This feature needs the 'r' extra: pip install 'cbrmb[r]'  "
    "(and the R packages: {pkgs})."
)


def _start_r_without_workspace():
    """Make the embedded R skip restoring ``./.RData`` when it starts.

    rpy2 starts R with ``--no-save`` only, so R restores a ``.RData`` found in the
    working directory. Objects from an old session (and their "namespace not
    available" warnings, a stale ``.Random.seed``) would then leak into every
    call. Has no effect if R was already started elsewhere in the process.
    """
    try:
        from rpy2.rinterface_lib import embedded

        if not embedded.isinitialized():
            embedded.set_initoptions(("rpy2", "--quiet", "--no-save", "--no-restore"))
    except Exception:  # pragma: no cover - rpy2 missing or R already running
        pass


def require_rpy2():
    """Import and return the rpy2 pieces used across the R backend."""
    _start_r_without_workspace()
    try:
        from rpy2 import robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_R_EXTRA_HINT.format(pkgs="ape, phangorn, GUniFrac, NbClust")) from exc
    return ro, numpy2ri, pandas2ri, localconverter, importr


@functools.lru_cache(maxsize=None)
def r_package(name: str):
    """Import an R package by name, with a helpful error if it is missing."""
    _, _, _, _, importr = require_rpy2()
    try:
        return importr(name)
    except Exception as exc:  # rpy2 raises PackageNotInstalledError (subclass of Exception)
        raise ImportError(_R_EXTRA_HINT.format(pkgs=name)) from exc
