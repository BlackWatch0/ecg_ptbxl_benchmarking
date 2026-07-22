"""Compare standardized experiment metrics across seeds."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .report import build_manifest
except ImportError:  # Direct execution: python evaluation/compare_experiments.py
    from report import build_manifest


STANDARD_METRICS = (
    "macro_roc_auc",
    "micro_roc_auc",
    "macro_pr_auc",
    "micro_pr_auc",
    "macro_f1",
    "micro_f1",
    "samples_f1",
    "label_accuracy",
    "exact_match_accuracy",
    "predicted_positive_rate",
    "mean_predicted_labels",
    "all_zero_prediction_rate",
    "parameter_count",
    "trainable_parameter_count",
    "training_time_seconds",
    "best_epoch",
    "best_valid_loss",
    "inference_time_per_sample_ms",
    "actual_batch_size",
)
DEFAULT_PLOT_METRICS = (
    "macro_roc_auc",
    "macro_pr_auc",
    "macro_f1",
    "micro_f1",
    "exact_match_accuracy",
)
ALIASES = {
    "experiment": "experiment_name",
    "model": "experiment_name",
    "scenario": "ecg_scenario",
    "thresholding_strategy": "threshold_strategy",
    "macro_auc": "macro_roc_auc",
    "macro_auroc": "macro_roc_auc",
    "macro_auprc": "macro_pr_auc",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_").lower()


def _candidate_files(root: Path, metrics_filename: Optional[str]) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    preferred = []
    if metrics_filename:
        preferred.extend([root / metrics_filename, root / "metrics" / metrics_filename])
    preferred.extend(
        [
            root / "overall_metrics.csv",
            root / "metrics" / "overall_metrics.csv",
            root / "metrics" / "runner_metrics.csv",
            root / "benchmark_summary.csv",
            root / "final_report" / "benchmark_summary.csv",
        ]
    )
    for path in preferred:
        if path.is_file():
            return [path]
    files = sorted(
        path
        for path in (root / "metrics").rglob("*.csv")
        if not any(token in path.name.lower() for token in ("per_class", "threshold", "complexity"))
    ) if (root / "metrics").is_dir() else []
    if not files:
        raise FileNotFoundError("no standardized overall metrics CSV found under {}".format(root))
    return files


def load_experiment_metrics(
    source: Union[str, Path], metrics_filename: Optional[str] = None
) -> Tuple[pd.DataFrame, List[Path]]:
    """Load one standardized CSV or a directory of per-seed metric CSVs."""
    paths = _candidate_files(Path(source).expanduser(), metrics_filename)
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame.columns = [column.strip().lower() for column in frame.columns]
        for old, new in ALIASES.items():
            if old in frame and new not in frame:
                frame = frame.rename(columns={old: new})
        frame["source_csv"] = str(path.resolve())
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False)
    if "experiment_name" not in result and "model_name" in result:
        result["experiment_name"] = result["model_name"]
    required = ["experiment_name", "seed"]
    missing = [column for column in required if column not in result]
    if missing:
        raise ValueError("{} is missing columns {}".format(source, missing))
    return result, paths


def aggregate_seed_metrics(
    metrics: pd.DataFrame,
    metric_columns: Optional[Sequence[str]] = None,
    group_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return wide and long mean/std/min/max/n summaries over distinct seeds."""
    if "experiment_name" not in metrics or "seed" not in metrics:
        raise ValueError("metrics must contain experiment_name and seed")
    selected_metrics = list(metric_columns or [column for column in STANDARD_METRICS if column in metrics])
    selected_metrics = [
        column
        for column in selected_metrics
        if column in metrics and pd.to_numeric(metrics[column], errors="coerce").notna().any()
    ]
    if not selected_metrics:
        raise ValueError("no measured metric columns are available for comparison")
    groups = list(group_columns or [
        column
        for column in ("experiment_name", "ecg_scenario", "target_snr_db", "threshold_strategy")
        if column in metrics
    ])
    if "experiment_name" not in groups:
        groups.insert(0, "experiment_name")

    long_parts = []
    for metric in selected_metrics:
        values = metrics[groups + ["seed", metric]].copy()
        values[metric] = pd.to_numeric(values[metric], errors="coerce")
        values = values.dropna(subset=[metric])
        duplicate_keys = groups + ["seed"]
        duplicates = values.duplicated(duplicate_keys, keep=False)
        if duplicates.any():
            bad = values.loc[duplicates, duplicate_keys].drop_duplicates().head(10).to_dict("records")
            raise ValueError("multiple {} rows for the same seed/group: {}".format(metric, bad))
        grouped = values.groupby(groups, dropna=False)[metric]
        part = grouped.agg(["mean", "min", "max", "count"]).reset_index()
        part["std"] = grouped.std(ddof=0).to_numpy()
        part = part.rename(columns={"count": "n"})
        part.insert(len(groups), "metric", metric)
        part = part[groups + ["metric", "mean", "std", "min", "max", "n"]]
        long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True, sort=False)

    wide = long[groups].drop_duplicates().reset_index(drop=True)
    for metric in selected_metrics:
        rows = long[long["metric"] == metric].drop(columns="metric")
        rows = rows.rename(
            columns={stat: "{}_{}".format(metric, stat) for stat in ("mean", "std", "min", "max", "n")}
        )
        wide = wide.merge(rows, on=groups, how="left")
    return wide, long


def plot_comparison(
    comparison_long: pd.DataFrame,
    output_dir: Union[str, Path],
    metrics: Optional[Sequence[str]] = None,
    dpi: int = 200,
) -> List[Dict[str, Path]]:
    """Create standard comparison bars and an exact plotting-data CSV per image."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    available = comparison_long["metric"].dropna().astype(str).unique().tolist()
    selected = [metric for metric in (metrics or DEFAULT_PLOT_METRICS) if metric in available]
    artifacts: List[Dict[str, Path]] = []
    identity = [
        column
        for column in ("experiment_name", "ecg_scenario", "target_snr_db", "threshold_strategy")
        if column in comparison_long
    ]
    for metric in selected:
        data = comparison_long[comparison_long["metric"] == metric].copy()
        data["series"] = data[identity].apply(
            lambda row: " / ".join(str(value) for value in row if not pd.isna(value)), axis=1
        )
        figure, axis = plt.subplots(figsize=(max(7.0, len(data) * 0.65), 4.8))
        x = np.arange(len(data))
        axis.bar(x, data["mean"], yerr=data["std"].fillna(0), capsize=3)
        axis.set_xticks(x)
        axis.set_xticklabels(data["series"], rotation=30, ha="right")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title("{} comparison across seeds".format(metric.replace("_", " ").title()))
        figure.tight_layout()
        stem = "comparison_" + _slug(metric)
        png = destination / (stem + ".png")
        pdf = destination / (stem + ".pdf")
        data_path = destination / (stem + "_plot_data.csv")
        figure.savefig(str(png), dpi=max(200, int(dpi)), bbox_inches="tight")
        figure.savefig(str(pdf), bbox_inches="tight")
        plt.close(figure)
        data.to_csv(data_path, index=False)
        artifacts.append({"png": png, "pdf": pdf, "data": data_path})
    return artifacts


def compare_experiments(
    experiment_paths: Sequence[Union[str, Path]],
    output_dir: Union[str, Path],
    metrics: Optional[Sequence[str]] = None,
    threshold_strategy: Optional[str] = "fixed_global",
    group_columns: Optional[Sequence[str]] = None,
    metrics_filename: Optional[str] = None,
    dpi: int = 200,
) -> Dict[str, Any]:
    """Load experiments, aggregate independent seeds, and write CSV/plot artifacts."""
    if len(experiment_paths) < 2:
        raise ValueError("at least two experiment paths are required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frames, input_rows = [], []
    for source in experiment_paths:
        frame, paths = load_experiment_metrics(source, metrics_filename)
        frames.append(frame)
        input_rows.extend(
            {"input": str(source), "source_csv": str(path.resolve()), "rows": len(pd.read_csv(path)),
             "status": "complete", "reason": None}
            for path in paths
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if threshold_strategy is not None and "threshold_strategy" in combined:
        combined = combined[combined["threshold_strategy"] == threshold_strategy].copy()
    if combined.empty:
        raise ValueError("no rows match threshold strategy {}".format(threshold_strategy))
    wide, long = aggregate_seed_metrics(combined, metrics, group_columns)
    summary_path = destination / "comparison_summary.csv"
    long_path = destination / "comparison_long.csv"
    inputs_path = destination / "comparison_inputs.csv"
    wide.to_csv(summary_path, index=False)
    long.to_csv(long_path, index=False)
    pd.DataFrame(input_rows).to_csv(inputs_path, index=False)
    model_comparison = destination / "model_comparison.csv"
    per_class_comparison = destination / "per_class_model_comparison.csv"
    robustness_comparison = destination / "robustness_comparison.csv"
    efficiency_comparison = destination / "efficiency_comparison.csv"
    wide.to_csv(model_comparison, index=False)
    per_class_frames, robustness_frames, efficiency_frames = [], [], []
    for source in experiment_paths:
        root = Path(source)
        for relative, target in (("metrics/per_class_metrics.csv", per_class_frames),
                                 ("metrics/robustness_summary.csv", robustness_frames),
                                 ("metrics/efficiency_metrics.csv", efficiency_frames)):
            path = root / relative
            if path.is_file():
                target.append(pd.read_csv(path))
    def grouped(frames: List[pd.DataFrame], output: Path) -> None:
        if not frames:
            pd.DataFrame(columns=["status", "reason"]).assign(
                status=["not_generated"], reason=["source artifact unavailable"]).to_csv(output, index=False)
            return
        frame = pd.concat(frames, ignore_index=True, sort=False)
        numeric = [column for column in frame.select_dtypes(include=[np.number]).columns if column != "seed"]
        groups = [column for column in ("model_name", "experiment_name", "condition",
                                        "target_snr_db", "class_name", "metric") if column in frame]
        if "seed" in frame and groups and numeric:
            result = frame.groupby(groups, dropna=False)[numeric].agg(["mean", "std", "min", "max", "count"])
            result.columns = ["{}_{}".format(metric, statistic) for metric, statistic in result.columns]
            result.reset_index().to_csv(output, index=False)
        else:
            frame.to_csv(output, index=False)
    grouped(per_class_frames, per_class_comparison)
    grouped(robustness_frames, robustness_comparison)
    grouped(efficiency_frames, efficiency_comparison)
    figures = plot_comparison(long, destination / "figures", metrics=metrics, dpi=dpi)
    figure_dir = destination / "figures"
    if "target_snr_db" in combined:
        snr = combined[pd.to_numeric(combined.target_snr_db, errors="coerce").notna()].copy()
        snr["target_snr_db"] = pd.to_numeric(snr.target_snr_db)
        for metric, stem in (("macro_roc_auc", "models_snr_auc"),
                             ("macro_f1", "models_snr_f1")):
            if metric not in snr or snr.empty:
                continue
            data = snr.groupby(["experiment_name", "target_snr_db"], as_index=False)[metric].agg(
                ["mean", "std", "min", "max", "count"]).reset_index()
            figure, axis = plt.subplots(figsize=(8, 5))
            for name, rows in data.groupby("experiment_name"):
                rows = rows.sort_values("target_snr_db")
                axis.plot(rows.target_snr_db, rows["mean"], marker="o", label=name)
            axis.set_xlabel("SNR (dB)"); axis.set_ylabel(metric.replace("_", " ").title())
            axis.set_ylim(0, 1); axis.legend(); axis.set_title("Model robustness vs SNR")
            figure_dir.mkdir(parents=True, exist_ok=True)
            png = figure_dir / (stem + ".png"); figure.savefig(png, dpi=max(200, dpi), bbox_inches="tight")
            plt.close(figure); data.to_csv(figure_dir / (stem + "_plot_data.csv"), index=False)
    metadata_path = destination / "comparison_summary.json"
    metadata = {
        "status": "complete",
        "experiments": sorted(combined["experiment_name"].dropna().astype(str).unique().tolist()),
        "seeds": sorted(str(value) for value in combined["seed"].dropna().unique()),
        "threshold_strategy": threshold_strategy,
        "input_rows": int(len(combined)),
        "groups": int(len(wide)),
        "metrics": sorted(long["metric"].unique().tolist()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = build_manifest(destination)
    return {
        "summary_csv": summary_path,
        "model_comparison": model_comparison,
        "per_class_comparison": per_class_comparison,
        "robustness_comparison": robustness_comparison,
        "efficiency_comparison": efficiency_comparison,
        "long_csv": long_path,
        "inputs_csv": inputs_path,
        "summary_json": metadata_path,
        "figures": figures,
        "manifest": manifest,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", type=Path, help="Metric CSVs or experiment roots")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--experiments", dest="experiment_names", nargs="+")
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--metric", action="append", dest="metrics", help="Metric to compare; repeatable")
    parser.add_argument("--threshold-strategy", default="fixed_global")
    parser.add_argument("--all-threshold-strategies", action="store_true")
    parser.add_argument("--metrics-filename", help="Preferred CSV filename inside each experiment")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    experiments = list(args.experiments)
    if args.experiment_names:
        if args.results_root is None:
            raise ValueError("--experiments requires --results-root")
        experiments.extend(args.results_root / name for name in args.experiment_names)
    result = compare_experiments(
        experiments,
        args.output_dir,
        metrics=args.metrics,
        threshold_strategy=None if args.all_threshold_strategies else args.threshold_strategy,
        metrics_filename=args.metrics_filename,
        dpi=args.dpi,
    )
    print(result["summary_csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STANDARD_METRICS",
    "aggregate_seed_metrics",
    "compare_experiments",
    "load_experiment_metrics",
    "main",
    "parse_args",
    "plot_comparison",
]
