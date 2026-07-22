"""Plots generated exclusively from standardized evaluation CSV files."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "macro_roc_auc",
    "micro_roc_auc",
    "macro_pr_auc",
    "micro_pr_auc",
    "macro_f1",
    "micro_f1",
    "samples_f1",
    "label_accuracy",
    "exact_match_accuracy",
)
METRIC_LABELS = {
    "macro_roc_auc": "Macro ROC-AUC",
    "micro_roc_auc": "Micro ROC-AUC",
    "macro_pr_auc": "Macro PR-AUC",
    "micro_pr_auc": "Micro PR-AUC",
    "macro_f1": "Macro F1",
    "micro_f1": "Micro F1",
    "samples_f1": "Samples F1",
    "label_accuracy": "Label accuracy",
    "exact_match_accuracy": "Exact-match accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_").lower()


def _experiment_column(frame: pd.DataFrame) -> str:
    for column in ("experiment_name", "model_name"):
        if column in frame:
            return column
    raise ValueError("metrics CSV must contain experiment_name or model_name")


def _read_csv(path: Union[str, Path], required: Sequence[str]) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    frame = pd.read_csv(source)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError("{} is missing columns {}".format(source, missing))
    return frame


def _save(
    figure: plt.Figure,
    data: pd.DataFrame,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / (stem + ".png")
    pdf = output_dir / (stem + ".pdf")
    data_path = output_dir / (stem + "_plot_data.csv")
    figure.tight_layout()
    figure.savefig(str(png), dpi=max(200, int(dpi)), bbox_inches="tight")
    figure.savefig(str(pdf), bbox_inches="tight")
    plt.close(figure)
    data.to_csv(data_path, index=False)
    return {"png": png, "pdf": pdf, "data": data_path}


def _scenario_label(frame: pd.DataFrame) -> pd.Series:
    if "ecg_scenario" in frame:
        return frame["ecg_scenario"].fillna("unspecified").astype(str)
    return pd.Series(["all"] * len(frame), index=frame.index)


def plot_overall_metrics(
    metrics_csv: Union[str, Path],
    output_dir: Union[str, Path],
    metrics: Optional[Sequence[str]] = None,
    threshold_strategy: Optional[str] = "per_class_thresholds",
    dpi: int = 200,
) -> List[Dict[str, Path]]:
    """Plot per-experiment means and seed standard deviations by scenario."""
    frame = _read_csv(metrics_csv, [])
    experiment = _experiment_column(frame)
    if threshold_strategy is not None and "threshold_strategy" in frame:
        frame = frame[frame["threshold_strategy"] == threshold_strategy].copy()
    frame["scenario"] = _scenario_label(frame)
    selected = list(metrics or DEFAULT_METRICS)
    artifacts: List[Dict[str, Path]] = []
    for metric in selected:
        if metric not in frame or not pd.to_numeric(frame[metric], errors="coerce").notna().any():
            continue
        values = frame[[experiment, "scenario", metric]].copy()
        values[metric] = pd.to_numeric(values[metric], errors="coerce")
        values = values.dropna(subset=[metric])
        grouped = values.groupby([experiment, "scenario"], dropna=False)[metric]
        plot_data = grouped.agg(["mean", "min", "max", "count"]).reset_index()
        plot_data["std"] = grouped.std(ddof=0).to_numpy()
        plot_data = plot_data.rename(columns={"count": "n"})

        scenarios = list(dict.fromkeys(plot_data["scenario"].astype(str)))
        experiments = list(dict.fromkeys(plot_data[experiment].astype(str)))
        figure, axis = plt.subplots(figsize=(max(7.0, len(scenarios) * 1.1), 4.8))
        x = np.arange(len(scenarios), dtype=float)
        width = 0.8 / max(1, len(experiments))
        for index, name in enumerate(experiments):
            rows = plot_data[plot_data[experiment].astype(str) == name].set_index("scenario")
            means = rows.reindex(scenarios)["mean"].to_numpy(dtype=float)
            errors = rows.reindex(scenarios)["std"].fillna(0).to_numpy(dtype=float)
            axis.bar(
                x - 0.4 + width / 2 + index * width,
                means,
                width,
                yerr=errors,
                capsize=2,
                label=name,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(scenarios, rotation=25, ha="right")
        axis.set_xlabel("ECG scenario")
        axis.set_ylabel(METRIC_LABELS.get(metric, metric.replace("_", " ").title()))
        axis.set_title("{} by scenario".format(METRIC_LABELS.get(metric, metric)))
        axis.legend(fontsize="small")
        artifacts.append(_save(figure, plot_data, Path(output_dir), "overall_" + _slug(metric), dpi))
    return artifacts


def plot_per_class_metrics(
    metrics_csv: Union[str, Path],
    output_dir: Union[str, Path],
    metrics: Sequence[str] = ("roc_auc", "pr_auc", "f1"),
    threshold_strategy: Optional[str] = "per_class_thresholds",
    dpi: int = 200,
) -> List[Dict[str, Path]]:
    """Plot class-level metric means; each image has a matching data CSV."""
    frame = _read_csv(metrics_csv, ["class_name"])
    experiment = _experiment_column(frame)
    if threshold_strategy is not None and "threshold_strategy" in frame:
        frame = frame[frame["threshold_strategy"] == threshold_strategy].copy()
    frame["scenario"] = _scenario_label(frame)
    artifacts: List[Dict[str, Path]] = []
    for metric in metrics:
        if metric not in frame or not pd.to_numeric(frame[metric], errors="coerce").notna().any():
            continue
        values = frame[[experiment, "scenario", "class_name", metric]].copy()
        values[metric] = pd.to_numeric(values[metric], errors="coerce")
        plot_data = (
            values.dropna(subset=[metric])
            .groupby([experiment, "scenario", "class_name"], dropna=False)[metric]
            .agg(["mean", "min", "max", "count"])
            .reset_index()
            .rename(columns={"count": "n"})
        )
        plot_data["std"] = (
            values.dropna(subset=[metric])
            .groupby([experiment, "scenario", "class_name"], dropna=False)[metric]
            .std(ddof=0)
            .to_numpy()
        )
        labels = plot_data.apply(
            lambda row: "{} / {}".format(row[experiment], row["scenario"]), axis=1
        )
        plot_data["series"] = labels
        pivot = plot_data.pivot(index="class_name", columns="series", values="mean")
        figure, axis = plt.subplots(figsize=(max(7.5, len(pivot) * 1.3), 4.8))
        pivot.plot(kind="bar", ax=axis)
        axis.set_xlabel("Diagnostic superclass")
        axis.set_ylabel(METRIC_LABELS.get(metric, metric.replace("_", " ").title()))
        axis.set_title("Per-class {}".format(METRIC_LABELS.get(metric, metric)))
        axis.tick_params(axis="x", rotation=0)
        axis.legend(fontsize="small")
        artifacts.append(_save(figure, plot_data, Path(output_dir), "per_class_" + _slug(metric), dpi))
    return artifacts


def plot_training_history(
    history_csv: Union[str, Path],
    output_dir: Union[str, Path],
    dpi: int = 200,
) -> List[Dict[str, Path]]:
    """Plot measured train/validation losses from a standardized history CSV."""
    frame = _read_csv(history_csv, ["epoch"])
    loss_columns = [column for column in ("train_loss", "valid_loss") if column in frame]
    if not loss_columns:
        raise ValueError("history CSV must contain train_loss or valid_loss")
    series_columns = [column for column in ("experiment_name", "model_name", "seed") if column in frame]
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for keys, rows in frame.groupby(series_columns, dropna=False) if series_columns else [("run", frame)]:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        label = "/".join(str(value) for value in key_values)
        rows = rows.sort_values("epoch")
        for column in loss_columns:
            axis.plot(rows["epoch"], rows[column], label="{} {}".format(label, column))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training history")
    axis.legend(fontsize="small")
    return [_save(figure, frame, Path(output_dir), "training_history", dpi)]


def generate_standard_plots(
    overall_metrics_csv: Union[str, Path],
    output_dir: Union[str, Path],
    per_class_metrics_csv: Optional[Union[str, Path]] = None,
    history_csv: Optional[Union[str, Path]] = None,
    metrics: Optional[Sequence[str]] = None,
    threshold_strategy: Optional[str] = "per_class_thresholds",
    dpi: int = 200,
) -> List[Dict[str, Path]]:
    """Generate all available standard plots and their plotting-data CSVs."""
    artifacts = plot_overall_metrics(
        overall_metrics_csv, output_dir, metrics, threshold_strategy, dpi
    )
    if per_class_metrics_csv is not None:
        artifacts.extend(
            plot_per_class_metrics(
                per_class_metrics_csv, output_dir, threshold_strategy=threshold_strategy, dpi=dpi
            )
        )
    if history_csv is not None:
        artifacts.extend(plot_training_history(history_csv, output_dir, dpi=dpi))
    return artifacts


def generate_evaluation_plots(output_dir: Union[str, Path], dpi: int = 220) -> List[Dict[str, Path]]:
    """Generate the standard single-evaluation figure set from persisted artifacts.

    Missing optional inputs (for example training history) simply omit the
    corresponding figure; the run manifest records that absence.
    """
    root = Path(output_dir)
    figures = root / "figures"
    overall_path = root / "metrics" / "overall_metrics.csv"
    per_class_path = root / "metrics" / "per_class_metrics.csv"
    calibration_bins_path = root / "metrics" / "calibration_bins.csv"
    predictions_path = root / "predictions" / "sample_probabilities.npz"
    history_path = root / "history" / "training_history.csv"
    efficiency_path = root / "metrics" / "efficiency_metrics.csv"
    artifacts: List[Dict[str, Path]] = []
    overall = pd.read_csv(overall_path)
    noisy = overall[pd.to_numeric(overall.get("target_snr_db"), errors="coerce").notna()].copy()
    noisy["target_snr_db"] = pd.to_numeric(noisy["target_snr_db"])
    for metric, stem in (("macro_roc_auc", "snr_macro_auc"),
                         ("macro_pr_auc", "snr_macro_pr_auc"),
                         ("macro_f1", "snr_macro_f1")):
        if metric not in noisy or noisy.empty:
            continue
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        for condition, rows in noisy.groupby("condition", sort=False):
            rows = rows.sort_values("target_snr_db")
            axis.plot(rows.target_snr_db, rows[metric], marker="o", label=str(condition))
        axis.set_xlabel("SNR (dB)"); axis.set_ylabel(METRIC_LABELS.get(metric, metric))
        axis.set_ylim(0, 1); axis.set_title("{} vs SNR".format(METRIC_LABELS.get(metric, metric)))
        axis.legend(); artifacts.append(_save(figure, noisy, figures, stem, dpi))
    robustness_path = root / "metrics" / "robustness_summary.csv"
    if robustness_path.is_file():
        robust = pd.read_csv(robustness_path)
        if len(robust):
            figure, axis = plt.subplots(figsize=(8, 4.8))
            plot = robust[robust.metric.isin(["macro_roc_auc", "macro_f1"])].copy()
            if "mean_metric" in plot and "clean_value" in plot:
                plot["retention"] = plot.mean_metric / plot.clean_value.replace(0, np.nan)
                for domain, rows in plot.groupby("domain"):
                    axis.bar(rows.metric.astype(str) + "/" + str(domain), rows.retention)
                axis.set_ylim(0, 1.05); axis.set_ylabel("Performance retention")
                axis.tick_params(axis="x", rotation=25); axis.set_title("Robustness retention")
                artifacts.append(_save(figure, plot, figures, "robustness_retention", dpi))
    if per_class_path.is_file():
        per_class = pd.read_csv(per_class_path)
        clean = per_class[per_class.condition == "clean"] if "condition" in per_class else per_class
        for metric, stem in (("roc_auc", "per_class_auc"), ("f1", "per_class_f1"),
                             ("pr_auc", "per_class_pr_auc"),
                             ("prevalence", "class_prevalence")):
            if metric not in clean or clean.empty:
                continue
            plot = clean.groupby("class_name", as_index=False)[metric].mean()
            figure, axis = plt.subplots(figsize=(8, 4.8)); axis.bar(plot.class_name, plot[metric])
            axis.set_ylim(0, 1); axis.set_ylabel(METRIC_LABELS.get(metric, metric.title()))
            axis.set_title(stem.replace("_", " ").title())
            artifacts.append(_save(figure, plot, figures, stem, dpi))
        confusion_columns = [column for column in ("tp", "fp", "fn", "tn") if column in clean]
        if confusion_columns:
            plot = clean.groupby("class_name", as_index=False)[confusion_columns].sum()
            figure, axis = plt.subplots(figsize=(8, 4.8)); bottom = np.zeros(len(plot))
            for column in confusion_columns:
                axis.bar(plot.class_name, plot[column], bottom=bottom, label=column.upper())
                bottom += plot[column].to_numpy()
            axis.set_ylabel("Count"); axis.set_title("Per-class binary confusion summary"); axis.legend()
            artifacts.append(_save(figure, plot, figures, "confusion_summary", dpi))
    if predictions_path.is_file():
        with np.load(predictions_path, allow_pickle=False) as archive:
            probabilities = np.asarray(archive["probabilities"])
            labels = np.asarray(archive["labels"])
            predictions = np.asarray(archive["predictions"])
        plot = pd.DataFrame({"true_label_count": labels.sum(1),
                             "predicted_label_count": predictions.sum(1)})
        figure, axis = plt.subplots(figsize=(7, 4.8)); axis.scatter(
            plot.true_label_count, plot.predicted_label_count, alpha=.25)
        axis.set_xlabel("True labels per sample"); axis.set_ylabel("Predicted labels per sample")
        axis.set_title("Predicted vs true label count")
        artifacts.append(_save(figure, plot, figures, "predicted_vs_true_label_count", dpi))
        histogram = pd.DataFrame({"confidence": probabilities.ravel()})
        figure, axis = plt.subplots(figsize=(7, 4.8)); axis.hist(histogram.confidence, bins=20)
        axis.set_xlabel("Predicted probability"); axis.set_ylabel("Count")
        axis.set_title("Confidence histogram")
        artifacts.append(_save(figure, histogram, figures, "confidence_histogram", dpi))
    if calibration_bins_path.is_file():
        bins = pd.read_csv(calibration_bins_path)
        plot = bins[bins.scope == "flattened"].copy()
        if len(plot):
            figure, axis = plt.subplots(figsize=(6, 6)); axis.plot([0, 1], [0, 1], "--", color="gray")
            axis.plot(plot.mean_probability, plot.observed_frequency, marker="o")
            axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.set_xlabel("Mean confidence")
            axis.set_ylabel("Observed frequency"); axis.set_title("Reliability diagram")
            artifacts.append(_save(figure, plot, figures, "reliability_diagram", dpi))
    if history_path.is_file():
        history = pd.read_csv(history_path)
        artifacts.extend(plot_training_history(history_path, figures, dpi))
        if "learning_rate" in history:
            figure, axis = plt.subplots(figsize=(8, 4.8)); axis.plot(history.epoch, history.learning_rate)
            axis.set_xlabel("Epoch"); axis.set_ylabel("Learning rate"); axis.set_title("Learning rate")
            artifacts.append(_save(figure, history, figures, "learning_rate_curve", dpi))
            if "valid_loss" in history:
                figure, loss_axis = plt.subplots(figsize=(8, 4.8))
                rate_axis = loss_axis.twinx()
                loss_axis.plot(history.epoch, history.valid_loss, color="tab:blue", label="Validation loss")
                rate_axis.plot(history.epoch, history.learning_rate, color="tab:orange", label="Learning rate")
                loss_axis.set_xlabel("Epoch"); loss_axis.set_ylabel("Validation loss", color="tab:blue")
                rate_axis.set_ylabel("Learning rate", color="tab:orange")
                loss_axis.set_title("Validation loss and learning rate")
                artifacts.append(_save(figure, history, figures,
                                       "validation_loss_learning_rate", dpi))
        if "epoch_time" in history:
            figure, axis = plt.subplots(figsize=(8, 4.8)); axis.plot(history.epoch, history.epoch_time)
            axis.set_xlabel("Epoch"); axis.set_ylabel("Epoch time (s)"); axis.set_title("Training time")
            artifacts.append(_save(figure, history, figures, "training_time_curve", dpi))
        if "train_loss" in history or "valid_loss" in history:
            source = figures / "training_history.png"
            if source.exists():
                shutil_target = figures / "loss_curve.png"
                shutil_target.write_bytes(source.read_bytes())
                history.to_csv(figures / "loss_curve_plot_data.csv", index=False)
    if efficiency_path.is_file():
        efficiency = pd.read_csv(efficiency_path)
        clean_metrics = overall[overall.condition == "clean"] if "condition" in overall else overall
        if len(efficiency) and len(clean_metrics):
            plot = efficiency.copy(); plot["macro_roc_auc"] = float(clean_metrics.macro_roc_auc.mean())
            figure, axis = plt.subplots(figsize=(7, 4.8)); axis.scatter(
                plot.average_sample_time_ms, plot.macro_roc_auc)
            axis.set_xlabel("Model inference time (ms/sample)"); axis.set_ylabel("Clean Macro ROC-AUC")
            axis.set_ylim(0, 1); axis.set_title("Performance and efficiency")
            artifacts.append(_save(figure, plot, figures, "performance_efficiency", dpi))
    return artifacts


__all__ = [
    "DEFAULT_METRICS",
    "generate_standard_plots",
    "plot_overall_metrics",
    "plot_per_class_metrics",
    "plot_training_history",
    "generate_evaluation_plots",
]
