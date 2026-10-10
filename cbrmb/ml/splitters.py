"""Splitters now live in the standalone package ``cvkit``; re-exported here for compatibility."""

from cvkit.splitters import RepeatedStratifiedGroupKFold, make_cv  # noqa: F401

__all__ = ["RepeatedStratifiedGroupKFold", "make_cv"]
