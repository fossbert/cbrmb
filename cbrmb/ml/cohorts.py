"""Cross-cohort validation now lives in the standalone package ``cvkit``; re-exported here for compatibility."""

from cvkit.cohorts import CrossCohortResult, cross_cohort  # noqa: F401

__all__ = ["cross_cohort", "CrossCohortResult"]
