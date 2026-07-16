"""CLI for clean/noisy/denoised time-domain robustness evaluation."""

import argparse

import pandas as pd

from time_domain_robustness.analysis import aggregate_pairs, bootstrap_metrics, compute_metrics, confidence_intervals, matching_report, pair_condition, sample_errors, snr_sort_key
from time_domain_robustness.constants import FEATURE_COLUMNS
from time_domain_robustness.io import load_data_root
from time_domain_robustness.reporting import write_outputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate clean, noisy, and denoised ECG time-domain features under one data root.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features", nargs=13, metavar="FEATURE", default=FEATURE_COLUMNS, help="Exactly 13 feature columns; defaults to the registered time-domain archive schema.")
    parser.add_argument("--evaluation-level", choices=("beat", "record", "both"), default="both")
    parser.add_argument("--aggregation", choices=("mean", "median", "both"), default="both")
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-sample-errors", action="store_true")
    args = parser.parse_args()
    features = tuple(args.features)
    data, quality = load_data_root(args.data_root, features)
    if not (data.Condition == "clean").any():
        parser.error("--data-root must contain clean tables")
    levels = ("beat", "record") if args.evaluation_level == "both" else (args.evaluation_level,)
    aggregations = ("mean", "median") if args.aggregation == "both" else (args.aggregation,)
    metrics, samples, reports, errors = [], [], [], []
    candidates = data[data.Condition.isin(["noisy", "denoised"])][["Condition", "SNR"]].drop_duplicates().to_dict("records")
    for candidate in sorted(candidates, key=lambda row: (row["Condition"], snr_sort_key(row["SNR"]))):
        comparison, snr = candidate["Condition"], candidate["SNR"]
        reports.append(matching_report(data, comparison, snr))
        pairs = pair_condition(data, comparison, snr, features)
        if pairs.empty:
            continue
        for level in levels:
            for aggregation in aggregations:
                evaluated = aggregate_pairs(pairs, level, aggregation, features)
                point = compute_metrics(evaluated, features, level, aggregation)
                draws = bootstrap_metrics(evaluated, features, level, aggregation, args.bootstraps, args.seed)
                metrics.append(confidence_intervals(point, draws))
                samples.append(draws)
                errors.append(sample_errors(evaluated, features, None if args.all_sample_errors else 100))
    if not metrics:
        parser.error("No clean/comparison composite keys matched")
    write_outputs(args.output_dir, quality, pd.concat(reports, ignore_index=True), pd.concat(metrics, ignore_index=True), pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(), pd.concat(errors, ignore_index=True), args.all_sample_errors)


if __name__ == "__main__":
    main()
