"""Robustness evaluation for paired ECG time-domain feature tables."""

from .analysis import BootstrapInputs, aggregate_pairs, bootstrap_inputs, bootstrap_metrics, compute_metrics, matching_report, pair_condition
from .constants import FEATURE_COLUMNS, KEY_COLUMNS
from .io import DuplicateKeyError, discover_feature_files, load_data_root, load_feature_tables

__all__ = ["FEATURE_COLUMNS", "KEY_COLUMNS", "DuplicateKeyError", "discover_feature_files", "load_feature_tables", "load_data_root", "pair_condition", "matching_report", "aggregate_pairs", "compute_metrics", "BootstrapInputs", "bootstrap_inputs", "bootstrap_metrics"]
