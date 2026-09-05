"""cbrmb -- microbiome analysis helpers.

Flat re-exports of the most-used helpers so notebooks can do::

    import cbrmb as mb

    res = mb.mwu_test(mb.filter_zotu(adata), adata.obs["group"])

Submodules: :mod:`cbrmb.stats`, :mod:`cbrmb.filtering`, :mod:`cbrmb.adapters`,
:mod:`cbrmb.clinical`, :mod:`cbrmb.pvalues`, :mod:`cbrmb.ordination`,
:mod:`cbrmb.longitudinal`, :mod:`cbrmb.unifrac` (pure-Python Generalized
UniFrac), and :mod:`cbrmb.rbackend` (needs the ``r`` extra).
"""

from __future__ import annotations

from .adapters import (
    alpha_diversity,
    best_clusters,
    calc_gunifrac,
    filter_tax,
    filter_zotu,
    taxa,
    zotus,
)
from .clinical import bernoulli_var, filter_bernoulli
from .filtering import filter_features, filter_prevalence, filter_rel_abundance
from .longitudinal import david_recipe
from .ordination import calc_mds, calc_umap
from .pvalues import cut_p, fdr
from .rbackend.kernel import cskat, glmm_mirkat
from .rbackend.permanova import (
    betadisper,
    screen_confounder,
    screen_effect_modifiers,
    subject_variation,
    test_confounder,
)
from .unifrac import generalized_unifrac
from .stats import (
    corr_test,
    fisher_test,
    kruskal_test,
    mwu_one_vs_rest,
    mwu_pairwise,
    mwu_test,
    wilcoxon_test,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # pvalues
    "fdr",
    "cut_p",
    # stats
    "kruskal_test",
    "mwu_test",
    "wilcoxon_test",
    "corr_test",
    "fisher_test",
    "mwu_one_vs_rest",
    "mwu_pairwise",
    # filtering
    "filter_prevalence",
    "filter_rel_abundance",
    "filter_features",
    # adapters
    "zotus",
    "taxa",
    "alpha_diversity",
    "filter_zotu",
    "filter_tax",
    "calc_gunifrac",
    "generalized_unifrac",
    "best_clusters",
    # ordination
    "calc_mds",
    "calc_umap",
    # longitudinal
    "david_recipe",
    # confounder / effect-modifier screening (R backend, lazy)
    "subject_variation",
    "screen_confounder",
    "test_confounder",
    "screen_effect_modifiers",
    "betadisper",
    "cskat",
    "glmm_mirkat",
    # clinical
    "bernoulli_var",
    "filter_bernoulli",
]
