"""CLI for version 2 time-domain feature robustness evaluation."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from time_domain_robustness.constants import FEATURE_COLUMNS
from time_domain_robustness.io import load_data_root
from time_domain_robustness_v2 import clean_scale, compute_metrics, confidence_intervals, input_manifest, overlap_audit, pair_condition, bootstrap_metrics


def _write_plots(output, metrics):
    data = metrics[(metrics.feature != "__macro_13d__") & (metrics.evaluation_level == "beat")]
    for comparison, subset in data.groupby("comparison"):
        pivot = subset.pivot_table(index="feature", columns="snr_db", values="nmae", aggfunc="first")
        figure, axis = plt.subplots(figsize=(9, 6))
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns)
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels(pivot.index)
        axis.set_title("{} NMAE by SNR (v2)".format(comparison))
        figure.colorbar(image, ax=axis, label="NMAE")
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(output / "v2_heatmap_{}_nmae.{}".format(comparison, extension), dpi=150)
        plt.close(figure)
    macro = metrics[metrics.feature.eq("__macro_13d__")]
    figure, axis = plt.subplots(figsize=(8, 5))
    for comparison, subset in macro.groupby("comparison"):
        axis.plot(subset.snr_db, subset.nmae, marker="o", label=comparison)
    axis.set(xlabel="SNR (dB)", ylabel="Strict macro NMAE", title="13-feature robustness (v2)")
    axis.legend()
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / "v2_macro_nmae_by_snr.{}".format(extension), dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Evaluate time-domain robustness using the v2 metric contract.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features", nargs=13, metavar="FEATURE", default=FEATURE_COLUMNS)
    parser.add_argument("--evaluation-level", choices=("beat", "record", "both"), default="both")
    parser.add_argument("--aggregation", choices=("mean", "median", "both"), default="both")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-bootstrap", action="store_true")
    args = parser.parse_args()
    features = tuple(args.features)
    if len(set(features)) != 13:
        parser.error("--features must contain exactly 13 unique columns")
    data, quality = load_data_root(args.data_root, features)
    if not data.Condition.eq("clean").any():
        parser.error("--data-root must contain clean tables")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_manifest(args.data_root).to_csv(output / "v2_input_manifest.csv", index=False)
    quality.to_csv(output / "v2_quality_report.csv", index=False)
    overlap_audit(data).to_csv(output / "v2_overlap_audit.csv", index=False)
    levels = ("beat", "record") if args.evaluation_level == "both" else (args.evaluation_level,)
    aggregations = ("mean", "median") if args.aggregation == "both" else (args.aggregation,)
    metrics, draws = [], []
    for candidate in data[data.Condition.ne("clean")][["Condition", "SNR"]].drop_duplicates().to_dict("records"):
        paired = pair_condition(data, candidate["Condition"], candidate["SNR"], features)
        if paired.empty:
            continue
        scales = {feature: clean_scale(paired[feature + "_clean"]) for feature in features}
        for level in levels:
            for aggregation in aggregations:
                evaluated = paired if level == "beat" else paired.groupby(["RecordNumber", "comparison", "snr_db"], as_index=False)[[feature + suffix for feature in features for suffix in ("_clean", "_comparison")]].agg(aggregation)
                point = compute_metrics(evaluated, features, scales, level, aggregation)
                sample = pd.DataFrame() if args.disable_bootstrap else bootstrap_metrics(evaluated, features, scales, level, aggregation, args.bootstrap_iterations, args.seed)
                metrics.append(confidence_intervals(point, sample))
                draws.append(sample)
    if not metrics:
        parser.error("No clean/comparison composite keys matched")
    metric_table = pd.concat(metrics, ignore_index=True)
    draw_table = pd.concat(draws, ignore_index=True) if draws else pd.DataFrame()
    metric_table.to_csv(output / "v2_feature_metrics.csv", index=False)
    metric_table[metric_table.feature.eq("__macro_13d__")].to_csv(output / "v2_macro_metrics.csv", index=False)
    draw_table.to_csv(output / "v2_bootstrap_samples.csv", index=False)
    _write_plots(output, metric_table)


if __name__ == "__main__":
    main()
