"""Nested cross-validation now lives in the standalone package ``cvkit``; re-exported here for compatibility."""

from cvkit.nested import NestedCVResult, PermutationResult, nested_cv, permutation_test  # noqa: F401

__all__ = ["nested_cv", "NestedCVResult", "permutation_test", "PermutationResult"]
