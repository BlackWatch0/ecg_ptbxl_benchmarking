"""Shared result-contract helpers for complete benchmark experiments."""

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TEST_CONDITIONS = (
    "clean", "noisy_snr24", "noisy_snr12", "noisy_snr6", "noisy_snr0",
    "noisy_snrm6", "denoised_snr24", "denoised_snr12", "denoised_snr6",
    "denoised_snr0", "denoised_snrm6",
)
THRESHOLD_FILES = ("0.5", "best_global_threshold", "per_class_thresholds")
REQUIRED_REPORT_FILES = (
    "ORIGINAL_MODELS_BENCHMARK_RESULTS.md", "benchmark_summary.csv",
    "best_model_summary.json", "clean_comparison.csv", "noisy_snr_comparison.csv",
    "denoised_snr_comparison.csv", "denoising_contributions.csv",
    "mean_domain_metrics.csv", "model_complexity.csv", "per_class_metrics.csv",
    "robustness_metrics.csv",
)
REQUIRED_FIGURES = (
    "clean_per_class_f1", "minus6db_per_class_f1", "noisy_macro_f1_vs_snr",
    "noisy_macro_roc_auc_vs_snr", "denoised_macro_f1_vs_snr",
    "denoised_macro_roc_auc_vs_snr", "noisy_vs_denoised_macro_f1",
    "noisy_vs_denoised_macro_roc_auc", "macro_f1_drops_from_clean",
    "macro_roc_auc_drops_from_clean", "parameters_tradeoff", "inference_tradeoff",
)
HISTORY_COLUMNS = (
    "epoch", "train_loss", "valid_loss", "train_accuracy", "valid_accuracy",
    "learning_rate", "epoch_duration_seconds", "best_epoch_so_far",
)


class ArtifactValidationError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda item: item.item()
                               if hasattr(item, "item") else str(item)) + "\n",
                    encoding="utf-8")


def prepare_experiment_root(root):
    root = Path(root)
    for name in ("config", "checkpoints", "metrics", "predictions", "training_logs",
                 "final_report/figures", "runtime_logs", "manifest", "errors"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def write_runtime_log(root, model_name, seed, message):
    path = Path(root) / "runtime_logs" / "{}_seed_{}.log".format(model_name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{} {}\n".format(utc_now(), message))
    return path


def save_predictions(path, ids, y_true, probabilities, prediction, thresholds):
    """Write one portable prediction table for every threshold strategy."""
    ids = np.asarray(ids)
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    prediction = np.asarray(prediction)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), probabilities.shape[1])
    frame = pd.DataFrame({"sample_id": ids, "record_id": ids, "ecg_id": ids})
    for index, class_name in enumerate(("NORM", "MI", "STTC", "CD", "HYP")):
        frame["true_" + class_name] = y_true[:, index].astype(int)
        frame["prob_" + class_name] = probabilities[:, index]
        frame["pred_" + class_name] = prediction[:, index].astype(int)
        frame["threshold_" + class_name] = thresholds[index]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_checkpoint_metadata(checkpoint_dir, model_name, seed, best_path, last_path,
                              config, best_epoch, best_loss, validation_metrics,
                              early_stopping="not_configured"):
    checkpoint_dir = Path(checkpoint_dir)
    suffix = Path(best_path).suffix
    best_standard = checkpoint_dir / ("best_model" + suffix)
    last_standard = checkpoint_dir / ("last_model" + Path(last_path).suffix)
    for source, target in ((Path(best_path), best_standard), (Path(last_path), last_standard)):
        if source.exists() and source != target:
            shutil.copy2(source, target)
    json_dump(checkpoint_dir / "checkpoint_metadata.json", {
        "model_name": model_name, "seed": seed, "best_checkpoint": best_standard.name,
        "last_checkpoint": last_standard.name, "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "early_stopping": early_stopping,
        "optimizer_state": "stored in last checkpoint when supported by the model runtime",
        "scheduler_state": "stored in last checkpoint when supported by the model runtime",
        "model_config": config, "validation_metrics": validation_metrics,
    })


def _expected_paths(models, seeds):
    files = ["config/{}".format(name) for name in (
        "resolved_config.yaml", "resolved_config.json", "data_integrity.json",
        "dataset_manifest.json", "split_manifest.json", "mlb.pkl",
        "standard_scaler.pkl", "artifact_status.json",
    )]
    files.extend("final_report/{}".format(name) for name in REQUIRED_REPORT_FILES)
    for figure in REQUIRED_FIGURES:
        files.extend("final_report/figures/{}.{}".format(figure, extension)
                     for extension in ("png", "pdf"))
    for model in models:
        figure_model = "wavelet_nn" if model == "wavelet_nn" else model
        for seed in seeds:
            prefix = "seed_{}".format(seed)
            files.extend((
                "training_logs/{}/{}.csv".format(model, prefix),
                "runtime_logs/{}_{}.log".format(model, prefix),
                "metrics/{}/{}_per_class.csv".format(model, prefix),
                "metrics/{}/{}_complexity.csv".format(model, prefix),
                "metrics/{}/{}_threshold_search.csv".format(model, prefix),
                "metrics/{}/{}_validation_metrics.json".format(model, prefix),
                "predictions/{}/{}/validation_predictions.csv".format(model, prefix),
                "predictions/{}/{}/validation_predictions_threshold_0_5.csv".format(model, prefix),
                "checkpoints/{}/{}/checkpoint_metadata.json".format(model, prefix),
            ))
            for condition in TEST_CONDITIONS:
                files.append("metrics/{}/{}_{}_integrity.json".format(model, prefix, condition))
                files.append("predictions/{}/{}/test_predictions_{}.csv".format(
                    model, prefix, condition))
                for strategy in THRESHOLD_FILES:
                    files.append("predictions/{}/{}/test_predictions_{}_{}.csv".format(
                        model, prefix, condition, strategy))
                if condition == "clean":
                    files.append("metrics/{}/{}.csv".format(model, prefix))
            files.extend("final_report/figures/{}.{}".format(
                "training_loss_{}".format(figure_model), extension) for extension in ("png", "pdf"))
            files.extend("final_report/figures/{}.{}".format(
                "training_validation_accuracy_{}_{}".format(figure_model, prefix), extension)
                         for extension in ("png", "pdf"))
    return files


def _check_prediction(path, expected_count):
    frame = pd.read_csv(path)
    required = {"sample_id"} | {"true_{}".format(name) for name in ("NORM", "MI", "STTC", "CD", "HYP")} | {
        "prob_{}".format(name) for name in ("NORM", "MI", "STTC", "CD", "HYP")}
    missing = required - set(frame.columns)
    if missing:
        return "missing columns {}".format(sorted(missing))
    if len(frame) != expected_count:
        return "expected {} samples, found {}".format(expected_count, len(frame))
    if frame.sample_id.duplicated().any():
        return "duplicate sample_id values"
    probabilities = frame[[column for column in frame if column.startswith("prob_")]].to_numpy()
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        return "probabilities must be finite values in [0, 1]"
    return None


def validate_experiment(root, models, seeds, strict=True):
    """Create manifests and raise when a completed benchmark violates the contract."""
    root = Path(root).resolve()
    manifest = root / "manifest"
    manifest.mkdir(parents=True, exist_ok=True)
    expected = _expected_paths(models, seeds)
    missing, invalid = [], []
    for relative in expected:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
            continue
        try:
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".csv":
                frame = pd.read_csv(path)
                if relative.startswith("training_logs/"):
                    columns = set(frame.columns)
                    absent = set(HISTORY_COLUMNS) - columns
                    if absent:
                        invalid.append((relative, "missing history columns {}".format(sorted(absent))))
                if "/predictions/" in relative:
                    count = None
                    split_path = root / "config" / "split_manifest.json"
                    if split_path.exists():
                        split = json.loads(split_path.read_text(encoding="utf-8"))
                        count = split["validation_records"] if "validation_" in relative else split["test_records"]
                    if count is not None:
                        problem = _check_prediction(path, count)
                        if problem:
                            invalid.append((relative, problem))
        except Exception as error:
            invalid.append((relative, str(error)))
    for model in models:
        for seed in seeds:
            directory = root / "checkpoints" / model / "seed_{}".format(seed)
            if not any(directory.glob("best_model.*")):
                missing.append(str(directory.relative_to(root) / "best_model.*"))
            if not any(directory.glob("last_model.*")):
                missing.append(str(directory.relative_to(root) / "last_model.*"))
    figures = root / "final_report" / "figures"
    if figures.exists():
        for png in figures.glob("*.png"):
            if not png.with_suffix(".pdf").is_file():
                invalid.append((str(png.relative_to(root)), "missing paired PDF"))
        for pdf in figures.glob("*.pdf"):
            if not pdf.with_suffix(".png").is_file():
                invalid.append((str(pdf.relative_to(root)), "missing paired PNG"))
    actual = [str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file()]
    json_dump(manifest / "expected_artifacts.json", {"models": models, "seeds": seeds, "files": expected})
    json_dump(manifest / "actual_artifacts.json", {"files": sorted(actual)})
    json_dump(manifest / "missing_artifacts.json", {"missing": sorted(set(missing)), "invalid": invalid})
    tree = "\n".join(sorted(actual)) + "\n"
    (manifest / "directory_tree.txt").write_text(tree, encoding="utf-8")
    checksums = {}
    for relative in actual:
        path = root / relative
        if path.name != "artifact_checksums.sha256":
            checksums[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (manifest / "artifact_checksums.sha256").write_text(
        "".join("{}  {}\n".format(value, key) for key, value in sorted(checksums.items())),
        encoding="utf-8")
    report = {"artifact_validation_passed": not missing and not invalid,
              "expected_file_count": len(expected), "actual_file_count": len(actual),
              "missing_required_files": sorted(set(missing)), "invalid_files": invalid}
    json_dump(manifest / "artifact_validation_report.json", report)
    if strict and not report["artifact_validation_passed"]:
        lines = ["[ERROR] {}".format(item) for item in report["missing_required_files"]]
        lines.extend("[ERROR] {}: {}".format(path, reason) for path, reason in invalid)
        raise ArtifactValidationError("\n".join(lines))
    return report


def write_experiment_status(root, status, experiment_name, git_commit, models, seeds,
                            validation_report, failed_models=None):
    json_dump(Path(root) / "manifest" / "experiment_status.json", {
        "status": status, "experiment_name": experiment_name, "git_commit": git_commit,
        "started_at_utc": None, "completed_at_utc": utc_now(), "models": models,
        "seeds": seeds, "expected_file_count": validation_report.get("expected_file_count", 0),
        "actual_file_count": validation_report.get("actual_file_count", 0),
        "missing_required_files": validation_report.get("missing_required_files", []),
        "failed_models": failed_models or [],
        "artifact_validation_passed": validation_report.get("artifact_validation_passed", False),
    })


def verify_archive(archive_path, root, required_files):
    root = Path(root).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    missing = [name for name in required_files if name not in names]
    if missing:
        raise ArtifactValidationError("Archive is missing required files: {}".format(
            ", ".join(missing)))
    return names
