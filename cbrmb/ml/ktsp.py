"""k-TSP now lives in the standalone package ``ktspy``; re-exported here for compatibility.

``from cbrmb.ml import KTSP, KTSPClassifier`` keeps working. For more than two classes use
``ktspy.MultiKTSPClassifier``.
"""

from ktspy.core import KTSP, KTSPClassifier, tsp_scores, wilcoxon_filter  # noqa: F401

__all__ = ["KTSP", "KTSPClassifier", "tsp_scores", "wilcoxon_filter"]
