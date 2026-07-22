"""End-to-end artifact orchestration for the independent evaluator."""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calibration import compute_calibration
from .config import EvaluationConfig
from .evaluator import EvaluationRecord, Evaluator
from .metrics import (ThresholdManager, apply_thresholds,
                      bootstrap_confidence_intervals, compute_metrics)
from .plotting import generate_evaluation_plots
from .prediction_export import export_sample_predictions
from .report import build_manifest, sha256_file
from .robustness import denoising_recovery, summarize_robustness


PRIMARY_METRICS = (
    "macro_roc_auc", "micro_roc_auc", "macro_pr_auc", "micro_pr_auc",
    "macro_f1", "micro_f1", "exact_match_accuracy",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(str(temporary), str(path))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(str(temporary), str(path))


def _git_revision(root: Path) -> Dict[str, Any]:
    def command(*arguments: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *arguments],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {"commit": command("rev-parse", "HEAD"),
                "dirty": bool(command("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("ecg_evaluation.{}".format(path.parent.name))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _prepare_output(config: EvaluationConfig) -> Path:
    output = Path(config.run.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not config.run.overwrite:
            raise FileExistsError("Output directory is not empty; use --overwrite: {}".format(output))
        for candidate in list(logging.Logger.manager.loggerDict.values()):
            if not isinstance(candidate, logging.Logger):
                continue
            for handler in list(candidate.handlers):
                filename = getattr(handler, "baseFilename", None)
                if filename and (Path(filename).resolve() == output or
                                 output in Path(filename).resolve().parents):
                    handler.close()
                    candidate.removeHandler(handler)
        shutil.rmtree(output)
    for name in ("config", "metrics", "predictions", "history", "figures", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    return output


def _thresholds(config: EvaluationConfig, class_names: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    threshold = config.analysis.threshold
    if config.analysis.thresholds:
        # Compatibility with the concise initial config field.
        values = np.asarray(config.analysis.thresholds, dtype=float)
        manager = ThresholdManager("fixed_per_class" if len(values) > 1 else "fixed_global",
                                   global_threshold=float(values[0]),
                                   per_class_thresholds=values,
                                   source_split=threshold.source_split)
    else:
        manager = ThresholdManager(threshold.mode, threshold.global_threshold,
                                   threshold.per_class or None, threshold.file,
                                   threshold.source_split)
    values = manager.resolve(len(class_names), class_names)
    payload = {"mode": manager.strategy, "source_split": manager.source_split,
               "comparison": ">=", "class_names": list(class_names),
               "thresholds": dict(zip(class_names, values.tolist()))}
    if manager.path:
        payload["source_file"] = str(Path(manager.path).resolve())
    return values, payload


def _partitions(record: EvaluationRecord) -> Iterable[Tuple[str, float, np.ndarray]]:
    frame = pd.DataFrame({"condition": record.condition.astype(str),
                          "snr": pd.to_numeric(record.snr, errors="coerce")})
    for (condition, snr), indices in frame.groupby(["condition", "snr"],
                                                   dropna=False, sort=False).indices.items():
        yield str(condition), float(snr) if pd.notna(snr) else float("nan"), np.asarray(indices)


def _condition_metrics(records: Sequence[EvaluationRecord], config: EvaluationConfig,
                       class_names: Sequence[str], thresholds: np.ndarray,
                       metrics_dir: Path, threshold_strategy: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                    pd.DataFrame, pd.DataFrame]:
    overall_rows: List[Dict[str, Any]] = []
    class_frames: List[pd.DataFrame] = []
    calibration_frames: List[pd.DataFrame] = []
    calibration_bins: List[pd.DataFrame] = []
    bootstrap_frames: List[pd.DataFrame] = []
    for record in records:
        for condition, snr, indices in _partitions(record):
            batch_losses = record.batch_losses if len(indices) == len(record.labels) else None
            result = compute_metrics(record.labels[indices], record.probabilities[indices],
                                     class_names, thresholds, batch_losses)
            identity = {"experiment_name": config.run.experiment_name,
                        "model_name": config.model.name, "seed": config.run.seed,
                        "dataset_split": config.run.dataset_split,
                        "ecg_scenario": record.scenario_name, "condition": condition,
                        "target_snr_db": snr, "sample_count": len(indices),
                        "threshold_strategy": threshold_strategy,
                        "metric_input": "probabilities",
                        "loss_input": "probabilities",
                        "nan_policy": "undefined per-class AUC is NaN and excluded from macro mean"}
            overall_rows.append(dict(identity, **result["overall"]))
            per_class = result["per_class"].copy()
            for key, value in identity.items():
                per_class[key] = value
            class_frames.append(per_class)
            if config.analysis.calibration:
                calibration = compute_calibration(record.labels[indices],
                                                  record.probabilities[indices], class_names,
                                                  config.analysis.calibration_bins)
                for table, destination in ((calibration["summary"], calibration_frames),
                                           (calibration["bins"], calibration_bins)):
                    table = table.copy()
                    for key, value in identity.items():
                        table[key] = value
                    destination.append(table)
            if config.analysis.bootstrap:
                frame = bootstrap_confidence_intervals(
                    record.labels[indices], record.probabilities[indices], thresholds,
                    config.analysis.bootstrap, random_state=config.run.seed,
                    metric_names=PRIMARY_METRICS)
                for key, value in identity.items():
                    frame[key] = value
                bootstrap_frames.append(frame)
    overall = pd.DataFrame(overall_rows)
    per_class = pd.concat(class_frames, ignore_index=True) if class_frames else pd.DataFrame()
    calibration = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    bins = pd.concat(calibration_bins, ignore_index=True) if calibration_bins else pd.DataFrame()
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True) if bootstrap_frames else pd.DataFrame()
    _write_csv(metrics_dir / "overall_metrics.csv", overall)
    _write_csv(metrics_dir / "per_class_metrics.csv", per_class)
    snr_metrics = overall.copy()
    clean_rows = snr_metrics[snr_metrics.condition == "clean"] if "condition" in snr_metrics else pd.DataFrame()
    if len(clean_rows):
        clean = clean_rows.iloc[0]
        for metric, prefix in (("macro_roc_auc", "macro_auc"),
                               ("micro_roc_auc", "micro_auc"),
                               ("macro_pr_auc", "macro_pr_auc"),
                               ("macro_f1", "macro_f1")):
            if metric in snr_metrics:
                snr_metrics[prefix + "_drop"] = float(clean[metric]) - snr_metrics[metric]
                snr_metrics[prefix + "_performance_retention"] = (
                    snr_metrics[metric] / float(clean[metric]) if float(clean[metric]) != 0 else np.nan)
                snr_metrics[prefix + "_relative_drop_percent"] = (
                    snr_metrics[prefix + "_drop"] / abs(float(clean[metric])) * 100
                    if float(clean[metric]) != 0 else np.nan)
    _write_csv(metrics_dir / "snr_metrics.csv", snr_metrics)
    if len(calibration):
        _write_csv(metrics_dir / "calibration_metrics.csv", calibration)
        _write_csv(metrics_dir / "per_class_calibration.csv", calibration[calibration.scope == "per_class"])
        _write_csv(metrics_dir / "calibration_bins.csv", bins)
    if len(bootstrap):
        _write_csv(metrics_dir / "bootstrap_ci.csv", bootstrap)
    return overall, per_class, calibration, bins, bootstrap


def _robustness(overall: pd.DataFrame, metrics_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    source = overall.rename(columns={"condition": "domain", "target_snr_db": "snr_db"})
    metric_names = [name for name in ("macro_roc_auc", "micro_roc_auc", "macro_pr_auc",
                                      "micro_pr_auc", "macro_f1", "micro_f1") if name in source]
    summary = pd.DataFrame()
    recovery = pd.DataFrame()
    domains = set(source.domain.astype(str)) if len(source) else set()
    if "clean" in domains and domains.difference({"clean"}):
        summary = summarize_robustness(source, metric_names,
                                       group_columns=("experiment_name", "model_name", "seed"))
        summary = summary.rename(columns={"mean_value": "mean_metric",
                                          "worst_value": "worst_snr_metric",
                                          "mean_relative_drop": "relative_drop_percent"})
        if "relative_drop_percent" in summary:
            summary["relative_drop_percent"] *= 100.0
        _write_csv(metrics_dir / "robustness_summary.csv", summary)
    if {"clean", "noisy", "denoised"}.issubset(domains):
        recovery = denoising_recovery(source, metric_names,
                                      group_columns=("experiment_name", "model_name", "seed"))
        recovery = recovery.rename(columns={"degradation": "noise_drop",
                                            "denoising_improvement": "denoising_gain",
                                            "recovery_fraction": "recovery_rate",
                                            "clean_value": "clean_metric",
                                            "noisy_value": "noisy_metric",
                                            "denoised_value": "denoised_metric"})
        recovery["denoising_recovery"] = recovery["denoising_gain"]
        _write_csv(metrics_dir / "denoising_recovery.csv", recovery)
    return summary, recovery


def _efficiency(records: Sequence[EvaluationRecord], config: EvaluationConfig,
                metrics_dir: Path) -> pd.DataFrame:
    checkpoint = Path(config.model.checkpoint).expanduser() if config.model.checkpoint else None
    rows = []
    for record in records:
        total = record.model_seconds
        metadata = dict(record.efficiency)
        rows.append({"experiment_name": config.run.experiment_name,
                     "model_name": config.model.name, "scenario": record.scenario_name,
                     "total_parameters": metadata.get("total_parameters", np.nan),
                     "trainable_parameters": metadata.get("trainable_parameters", np.nan),
                     "checkpoint_size_mb": (checkpoint.stat().st_size / 1024 ** 2
                                            if checkpoint and checkpoint.is_file() else np.nan),
                     "model_size_mb": metadata.get("model_size_mb", np.nan),
                     "total_inference_time": total,
                     "end_to_end_time": record.end_to_end_seconds,
                     "average_batch_time": total / record.batch_count,
                     "average_sample_time_ms": record.model_ms_per_sample,
                     "throughput_samples_per_second": record.sample_count / total if total > 0 else np.nan,
                     "peak_gpu_memory_mb": metadata.get("peak_gpu_memory_mb", np.nan),
                     "average_gpu_memory_mb": metadata.get("average_gpu_memory_mb", np.nan),
                     "batch_size": config.data.batch_size,
                     "device_name": metadata.get("device_name", config.inference.device),
                     "dtype": config.inference.dtype, "number_of_samples": record.sample_count,
                     "flops": np.nan, "macs": np.nan})
    frame = pd.DataFrame(rows)
    _write_csv(metrics_dir / "efficiency_metrics.csv", frame)
    return frame


def _integrity(records: Sequence[EvaluationRecord], config: EvaluationConfig,
               class_names: Sequence[str]) -> Dict[str, Any]:
    checks = []
    reference = records[0] if records else None
    for record in records:
        unique = len(np.unique(record.sample_id)) == len(record.sample_id)
        checks.append({"scenario": record.scenario_name, "sample_count": record.sample_count,
                       "label_count": record.labels.shape[1], "sample_id_unique": unique,
                       "finite_labels": bool(np.isfinite(record.labels).all()),
                       "finite_probabilities": bool(np.isfinite(record.probabilities).all()),
                       "aligned_with_first": bool(reference is None or
                                                   np.array_equal(record.sample_id, reference.sample_id))})
        if not unique:
            raise ValueError("sample_id is not unique in {}".format(record.scenario_name))
    return {"status": "passed", "dataset_split": config.run.dataset_split,
            "class_names": list(class_names), "scenarios": checks,
            "source_paths": [scenario.path for scenario in config.data.scenarios]}


def _report(output: Path, config: EvaluationConfig, overall: pd.DataFrame,
            per_class: pd.DataFrame, robustness: pd.DataFrame,
            recovery: pd.DataFrame, calibration: pd.DataFrame,
            efficiency: pd.DataFrame, warnings: Sequence[str],
            class_names: Sequence[str]) -> Dict[str, Any]:
    clean = overall[overall.condition == "clean"] if "condition" in overall else pd.DataFrame()
    noisy = overall[overall.condition == "noisy"] if "condition" in overall else pd.DataFrame()
    best_class = worst_class = None
    if len(per_class) and "f1" in per_class:
        means = per_class.groupby("class_name").f1.mean()
        best_class, worst_class = str(means.idxmax()), str(means.idxmin())
    summary = {"experiment_name": config.run.experiment_name, "model_name": config.model.name,
               "seed": config.run.seed, "checkpoint": config.model.checkpoint,
               "checkpoint_sha256": (sha256_file(config.model.checkpoint)
                                     if config.model.checkpoint and
                                     Path(config.model.checkpoint).is_file() else None),
               "dataset": {"name": config.run.dataset_name, "version": config.run.dataset_version,
                           "split": config.run.dataset_split,
                           "paths": [scenario.path for scenario in config.data.scenarios],
                           "conditions": overall[["condition", "target_snr_db", "sample_count"]].to_dict(
                               "records") if len(overall) else []},
               "class_names": list(class_names),
               "threshold_mode": config.analysis.threshold.mode,
               "clean_metrics": clean.iloc[0].to_dict() if len(clean) else None,
               "worst_snr_metrics": (noisy.loc[pd.to_numeric(noisy.macro_roc_auc,
                                                               errors="coerce").idxmin()].to_dict()
                                     if len(noisy) and pd.to_numeric(
                                         noisy.macro_roc_auc, errors="coerce").notna().any() else None),
               "robustness_summary": robustness.to_dict("records"),
               "efficiency_summary": efficiency.to_dict("records"),
               "calibration_summary": calibration.to_dict("records"),
               "best_class": best_class, "worst_class": worst_class,
               "warnings": list(warnings)}
    _write_json(output / "summary.json", summary)
    def markdown(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "Not generated."
        columns = list(frame.columns)
        rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
        return "\n".join(["| " + " | ".join(columns) + " |",
                          "| " + " | ".join("---" for _ in columns) + " |"] +
                         ["| " + " | ".join(row) + " |" for row in rows])
    lines = ["# {} Evaluation Report".format(config.run.experiment_name), "",
             "- Model: `{}`".format(config.model.name),
             "- Checkpoint: `{}`".format(config.model.checkpoint or "precomputed predictions"),
             "- Dataset: `{}` version `{}` split `{}`".format(
                 config.run.dataset_name, config.run.dataset_version, config.run.dataset_split),
             "- Threshold mode: `{}`".format(config.analysis.threshold.mode), "",
             "## Conditions", "", markdown(overall), "",
             "## Denoising Recovery", "",
             markdown(recovery) if len(recovery) else "Not generated: denoised evaluation unavailable.",
             "", "## Class Summary", "",
             "Best mean F1: `{}`; worst mean F1: `{}`.".format(best_class, worst_class),
             "", "## Calibration", "",
             markdown(calibration) if len(calibration) else "Not generated.",
             "", "## Efficiency", "", markdown(efficiency),
             "", "## Warnings", ""]
    lines.extend(["- " + item for item in warnings] or ["- None"])
    lines.extend(["", "## Figures", ""])
    lines.extend("- [{}](figures/{})".format(path.name, path.name)
                 for path in sorted((output / "figures").glob("*.png")))
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_standard_evaluation(config: EvaluationConfig) -> Path:
    """Run inference once and generate the complete standardized artifact tree."""
    output = _prepare_output(config)
    logger = _logger(output / "logs" / "evaluation.log")
    repository_root = Path(__file__).resolve().parents[1]
    class_names = config.analysis.class_names or tuple(
        "class_{}".format(index) for index in range(config.model.num_classes))
    thresholds, threshold_payload = _thresholds(config, class_names)
    _write_json(output / "config" / "thresholds.json", threshold_payload)
    try:
        import yaml
        (output / "config" / "evaluation_config.yaml").write_text(
            yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
        (output / "config" / "model_config.yaml").write_text(
            yaml.safe_dump(asdict(config.model), sort_keys=False), encoding="utf-8")
    except ImportError:
        _write_json(output / "config" / "evaluation_config.json", asdict(config))
        _write_json(output / "config" / "model_config.json", asdict(config.model))
    git = _git_revision(repository_root)
    checkpoint_hash = (sha256_file(config.model.checkpoint)
                       if config.model.checkpoint and Path(config.model.checkpoint).is_file() else None)
    environment = {"generated_at": datetime.now(timezone.utc).isoformat(),
                   "python": sys.version, "platform": platform.platform(), "git": git,
                   "checkpoint": config.model.checkpoint, "checkpoint_sha256": checkpoint_hash}
    _write_json(output / "config" / "environment.json", environment)
    (output / "config" / "environment.txt").write_text(
        "\n".join("{}={}".format(key, value) for key, value in environment.items()) + "\n",
        encoding="utf-8")
    logger.info("Collecting predictions for %s", config.model.name)
    evaluator = Evaluator(config)
    records = evaluator.collect()
    integrity = _integrity(records, config, class_names)
    _write_json(output / "data_integrity_report.json", integrity)
    overall, per_class, calibration, bins, bootstrap = _condition_metrics(
        records, config, class_names, thresholds, output / "metrics", threshold_payload["mode"])
    robustness, recovery = (_robustness(overall, output / "metrics")
                            if config.analysis.robustness else (pd.DataFrame(), pd.DataFrame()))
    efficiency = (_efficiency(records, config, output / "metrics")
                  if config.output.measure_efficiency else pd.DataFrame())
    missing: Dict[str, str] = {}
    if not len(robustness):
        missing["metrics/robustness_summary.csv"] = (
            "robustness disabled" if not config.analysis.robustness else
            "clean plus noisy/denoised conditions were not available")
    if not len(recovery):
        missing["metrics/denoising_recovery.csv"] = "matched clean/noisy/denoised conditions were not available"
    if not len(calibration):
        missing["metrics/calibration_metrics.csv"] = "calibration disabled"
        missing["metrics/per_class_calibration.csv"] = "calibration disabled"
    if not len(bootstrap):
        missing["metrics/bootstrap_ci.csv"] = "bootstrap disabled"
    if not len(efficiency):
        missing["metrics/efficiency_metrics.csv"] = "efficiency measurement disabled"
    if config.output.save_predictions:
        labels = np.concatenate([record.labels for record in records])
        probabilities = np.concatenate([record.probabilities for record in records])
        logits = (np.concatenate([record.logits for record in records])
                  if config.output.save_logits and
                  all(record.logits is not None for record in records) else None)
        exported = export_sample_predictions(
            output / "predictions", np.concatenate([record.sample_id for record in records]),
            labels, probabilities, class_names, thresholds=thresholds, logits=logits,
            extra_columns={"dataset_split": config.run.dataset_split,
                           "condition": np.concatenate([record.condition for record in records]),
                           "snr": np.concatenate([record.snr for record in records])},
            basename="sample_predictions")
        os.replace(str(exported["npz"]),
                   str(output / "predictions" / "sample_probabilities.npz"))
    else:
        missing["predictions/sample_predictions.csv"] = "save_predictions disabled"
        missing["predictions/sample_probabilities.npz"] = "save_predictions disabled"
    if config.run.history_file:
        source = Path(config.run.history_file)
        if source.is_file():
            history = pd.read_csv(source)
            _write_csv(output / "history" / "training_history.csv", history)
        else:
            missing["history/training_history.csv"] = "configured history file is absent"
    else:
        missing["history/training_history.csv"] = "no history file configured"
    if config.output.save_plots:
        generate_evaluation_plots(output)
        expected_figures = ("snr_macro_auc.png", "snr_macro_pr_auc.png", "snr_macro_f1.png",
                            "robustness_retention.png", "per_class_auc.png", "per_class_f1.png",
                            "per_class_pr_auc.png", "class_prevalence.png", "confusion_summary.png",
                            "predicted_vs_true_label_count.png", "reliability_diagram.png",
                            "confidence_histogram.png", "loss_curve.png", "learning_rate_curve.png",
                            "performance_efficiency.png")
        for name in expected_figures:
            if not (output / "figures" / name).is_file():
                missing["figures/" + name] = "required source condition or history was unavailable"
    else:
        missing["figures/"] = "save_plots disabled"
    warnings_list: List[str] = []
    if config.output.save_logits and config.output.save_predictions and logits is None:
        warnings_list.append("Raw logits were unavailable; probabilities were evaluated directly.")
    _report(output, config, overall, per_class, robustness, recovery,
            calibration, efficiency, warnings_list, class_names)
    build_manifest(output, missing=missing)
    # Refresh summary after manifest so callers have an explicit generated file inventory.
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["generated_files"] = [entry["path"] for entry in
                                  json.loads((output / "manifest.json").read_text())["artifacts"]
                                  if entry["status"] == "generated"]
    _write_json(summary_path, summary)
    build_manifest(output, missing=missing)
    logger.info("Evaluation complete: %s", output)
    return output


__all__ = ["run_standard_evaluation"]
