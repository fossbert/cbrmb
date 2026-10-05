"""cbrmb -- microbiome analysis helpers.

Flat re-exports of the most-used helpers so notebooks can do::

    import cbrmb as mb

    res = mb.mwu_test(mb.filter_zotu(adata), adata.obs["group"])

Submodules: :mod:`cbrmb.stats`, :mod:`cbrmb.filtering`, :mod:`cbrmb.adapters`,
:mod:`cbrmb.clinical`, :mod:`cbrmb.pvalues`, :mod:`cbrmb.ordination`,
:mod:`cbrmb.longitudinal`, :mod:`cbrmb.survival` (added value of microbiome
measures over a clinical Cox model), :mod:`cbrmb.paired` (find paired samples, deltas,
shift distances), :mod:`cbrmb.distance` (PCoA, Mantel), :mod:`cbrmb.unifrac`
(pure-Python Generalized UniFrac), :mod:`cbrmb.palettes` (phylum colour book), :mod:`cbrmb.plotting`
(needs the ``plotting`` extra), :mod:`cbrmb.ml` (nested CV, cross-cohort
validation, k-TSP; use as ``mb.ml.nested_cv``), and :mod:`cbrmb.rbackend`
(needs the ``r`` extra).
"""

from __future__ import annotations

from .adapters import (
    alpha_diversity,
    best_clusters,
    calc_gunifrac,
    filter_tax,
    filter_zotu,
    taxa,
    top_taxa,
    zotus,
)
from .clinical import bernoulli_var, filter_bernoulli
from .distance import (
    MantelResult,
    as_distance_frame,
    distance_matrix,
    mantel_screen,
    mantel_test,
    pcoa,
    subset_distance,
    within_group_distances,
)
from .filtering import filter_features, filter_prevalence, filter_rel_abundance
from .longitudinal import david_recipe
from .ordination import OrdinationReport, calc_mds, calc_umap, ordination_report
from .paired import Pairs, align_samples, find_pairs
from .palettes import PHYLUM_COLORS, phylum_colors, resolve_phylum
from .plotting import phylum_handles, plot_read_depth
from .pvalues import cut_p, fdr
from .rbackend.kernel import cskat, glmm_mirkat
from .rbackend.mixed import AlphaMixedFit, alpha_mixed, alpha_mixed_screen
from .rbackend.permanova import (
    betadisper,
    bootstrap_mediation,
    mediation_decompose,
    screen_confounder,
    screen_effect_modifiers,
    subject_variation,
    test_confounder,
    test_confounder_adjusted,
)
from .survival import concordance_index, cox_added_value, cox_screen, outcome_free_score
from .unifrac import generalized_unifrac
from . import ml
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
    # survival
    "cox_added_value",
    "cox_screen",
    "outcome_free_score",
    "concordance_index",
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
    "top_taxa",
    "alpha_diversity",
    "filter_zotu",
    "filter_tax",
    "calc_gunifrac",
    "generalized_unifrac",
    "best_clusters",
    # ordination
    "calc_mds",
    "calc_umap",
    "ordination_report",
    "OrdinationReport",
    # paired samples
    "find_pairs",
    "Pairs",
    "align_samples",
    # distance matrices
    "as_distance_frame",
    "subset_distance",
    "distance_matrix",
    "pcoa",
    "mantel_test",
    "mantel_screen",
    "MantelResult",
    "within_group_distances",
    # palettes
    "PHYLUM_COLORS",
    "phylum_colors",
    "resolve_phylum",
    # plotting
    "plot_read_depth",
    "phylum_handles",
    # longitudinal
    "david_recipe",
    # confounder / effect-modifier screening (R backend, lazy)
    "subject_variation",
    "screen_confounder",
    "test_confounder",
    "test_confounder_adjusted",
    "mediation_decompose",
    "bootstrap_mediation",
    "screen_effect_modifiers",
    "betadisper",
    "cskat",
    "glmm_mirkat",
    # alpha-diversity mixed models (R backend, lazy)
    "alpha_mixed",
    "alpha_mixed_screen",
    "AlphaMixedFit",
    # clinical
    "bernoulli_var",
    "filter_bernoulli",
]
