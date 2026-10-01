"""Machine learning: nested cross-validation, cross-cohort validation, k-TSP.

* :mod:`cbrmb.ml.ncv` -- :func:`nested_cv` for any sklearn estimator/pipeline
  (Hermida et al. 2022 design), model comparison, feature stability,
  :func:`permutation_test`.
* :mod:`cbrmb.ml.cohorts` -- :func:`cross_cohort`: train in each cohort (or all
  but one), validate on the others.
* :mod:`cbrmb.ml.ktsp` -- k top scoring pairs, a pure-NumPy port of
  ``switchBox``: :class:`KTSP` (pair features for any classifier) and
  :class:`KTSPClassifier` (majority vote).
* :mod:`cbrmb.ml.transformers` -- pipeline steps for count tables
  (:class:`PrevalenceThreshold`, :class:`RelativeAbundance`, :class:`CLR`).
* :mod:`cbrmb.ml.splitters` -- :class:`RepeatedStratifiedGroupKFold`.
"""

from .cohorts import CrossCohortResult, cross_cohort
from .ktsp import KTSP, KTSPClassifier, tsp_scores, wilcoxon_filter
from .ncv import NestedCVResult, PermutationResult, nested_cv, permutation_test
from .splitters import RepeatedStratifiedGroupKFold, make_cv
from .transformers import CLR, PrevalenceThreshold, RelativeAbundance

__all__ = [
    "nested_cv",
    "NestedCVResult",
    "permutation_test",
    "PermutationResult",
    "cross_cohort",
    "CrossCohortResult",
    "KTSP",
    "KTSPClassifier",
    "tsp_scores",
    "wilcoxon_filter",
    "PrevalenceThreshold",
    "RelativeAbundance",
    "CLR",
    "RepeatedStratifiedGroupKFold",
    "make_cv",
]
