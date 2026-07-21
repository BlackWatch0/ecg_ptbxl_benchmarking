"""Measured-result reports, reproducibility copies, and artifact manifests."""

import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


SUMMARY_METRICS = (
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


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def csv_row_count(path: Union[str, Path]) -> Optional[int]:
    """Count CSV data rows, excluding the header; return None for non-CSV files."""
    source = Path(path)
    if source.suffix.lower() != ".csv":
        return None
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def manifest_entry(
    path: Union[str, Path],
    root: Union[str, Path],
    status: str = "generated",
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Describe one present artifact with integrity and row-count metadata."""
    source, base = Path(path), Path(root)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    try:
        name = source.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        name = source.name
    return {
        "path": name,
        "sha256": sha256_file(source),
        "size": source.stat().st_size,
        "rows": csv_row_count(source),
        "status": status,
        "reason": reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def missing_manifest_entry(path: str, reason: str) -> Dict[str, Any]:
    """Describe an unavailable artifact without inventing content or metadata."""
    return {
        "path": Path(path).as_posix(),
        "sha256": None,
        "size": None,
        "rows": None,
        "status": "not_generated",
        "reason": reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_manifest(
    output_dir: Union[str, Path],
    missing: Optional[Mapping[str, str]] = None,
    manifest_name: str = "manifest.json",
) -> Path:
    """Hash every output file and append explicit records for expected missing files."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / manifest_name
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != manifest_path
    )
    entries = [manifest_entry(path, root) for path in files]
    present = {entry["path"] for entry in entries}
    for name, reason in sorted((missing or {}).items()):
        normalized = Path(name).as_posix()
        if normalized not in present:
            entries.append(missing_manifest_entry(normalized, reason))
    payload = {
        "manifest_version": 1,
        "root": ".",
        "artifact_count": len(entries),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _copy_group(
    sources: Iterable[Union[str, Path]],
    destination: Path,
    category: str,
) -> Tuple[List[Path], Dict[str, str]]:
    copied: List[Path] = []
    missing: Dict[str, str] = {}
    destination.mkdir(parents=True, exist_ok=True)
    names = set()
    for value in sources:
        source = Path(value).expanduser()
        target_name = source.name or "missing"
        relative = "{}/{}".format(category, target_name)
        if not source.is_file():
            missing[relative] = "source file does not exist: {}".format(source)
            continue
        if target_name in names:
            raise ValueError("duplicate {} artifact filename: {}".format(category, target_name))
        names.add(target_name)
        target = destination / target_name
        if source.resolve() != target.resolve():
            shutil.copy2(str(source), str(target))
        copied.append(target)
    return copied, missing


def copy_reproducibility_artifacts(
    output_dir: Union[str, Path],
    config_files: Sequence[Union[str, Path]] = (),
    environment_files: Sequence[Union[str, Path]] = (),
    history_files: Sequence[Union[str, Path]] = (),
) -> Dict[str, Any]:
    """Copy supplied config, environment, and history files into fixed subdirectories."""
    root = Path(output_dir)
    result: Dict[str, Any] = {"config": [], "environment": [], "history": [], "missing": {}}
    for category, sources in (
        ("config", config_files),
        ("environment", environment_files),
        ("history", history_files),
    ):
        copied, missing = _copy_group(sources, root / category, category)
        result[category] = copied
        result["missing"].update(missing)
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def summarize_metrics(
    overall_metrics_csv: Union[str, Path],
    threshold_strategy: Optional[str] = "per_class_thresholds",
) -> Dict[str, Any]:
    """Build a JSON-safe summary from measured standardized overall metrics."""
    source = Path(overall_metrics_csv)
    frame = pd.read_csv(source)
    if threshold_strategy is not None and "threshold_strategy" in frame:
        selected = frame[frame["threshold_strategy"] == threshold_strategy].copy()
    else:
        selected = frame.copy()
    experiment_column = next(
        (column for column in ("experiment_name", "model_name") if column in selected), None
    )
    summary: Dict[str, Any] = {
        "status": "complete" if len(selected) else "missing",
        "reason": None if len(selected) else "no rows matched the requested threshold strategy",
        "source": str(source),
        "threshold_strategy": threshold_strategy,
        "rows": int(len(selected)),
        "experiments": sorted(selected[experiment_column].dropna().astype(str).unique().tolist())
        if experiment_column
        else [],
        "seeds": sorted(_json_value(value) for value in selected["seed"].dropna().unique())
        if "seed" in selected
        else [],
        "scenarios": sorted(selected["ecg_scenario"].dropna().astype(str).unique().tolist())
        if "ecg_scenario" in selected
        else [],
        "metrics": {},
        "best": {},
    }
    for metric in SUMMARY_METRICS:
        if metric not in selected:
            continue
        values = pd.to_numeric(selected[metric], errors="coerce")
        measured = values.dropna()
        if not len(measured):
            continue
        summary["metrics"][metric] = {
            "mean": float(measured.mean()),
            "std": float(measured.std(ddof=0)),
            "min": float(measured.min()),
            "max": float(measured.max()),
            "n": int(measured.count()),
        }
        if experiment_column:
            means = (
                pd.DataFrame({experiment_column: selected[experiment_column], metric: values})
                .dropna(subset=[experiment_column, metric])
                .groupby(experiment_column)[metric]
                .mean()
            )
            if len(means):
                winner = means.idxmax()
                summary["best"][metric] = {
                    "experiment_name": str(winner),
                    "mean": float(means.loc[winner]),
                }
    return _json_value(summary)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No measured rows are available."
    values: List[List[str]] = [[str(column) for column in frame.columns]]
    for row in frame.itertuples(index=False, name=None):
        values.append(
            [
                "{:.4f}".format(value)
                if isinstance(value, (float, np.floating)) and math.isfinite(float(value))
                else "missing"
                if pd.isna(value)
                else str(value)
                for value in row
            ]
        )
    widths = [max(len(row[index]) for row in values) for index in range(len(values[0]))]
    lines = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in values
    ]
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([lines[0], separator] + lines[1:])


def write_markdown_report(
    path: Union[str, Path],
    overall_metrics_csv: Union[str, Path],
    title: str = "Evaluation Report",
    threshold_strategy: Optional[str] = "per_class_thresholds",
    summary: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a compact Markdown report whose values are aggregated from the CSV."""
    source = Path(overall_metrics_csv)
    frame = pd.read_csv(source)
    if threshold_strategy is not None and "threshold_strategy" in frame:
        frame = frame[frame["threshold_strategy"] == threshold_strategy].copy()
    experiment = next((column for column in ("experiment_name", "model_name") if column in frame), None)
    metrics = [metric for metric in SUMMARY_METRICS if metric in frame]
    if experiment and metrics and len(frame):
        table = frame.groupby(experiment, as_index=False)[metrics].mean(numeric_only=True)
    else:
        table = pd.DataFrame()
    details = dict(summary or summarize_metrics(source, threshold_strategy))
    reason = details.get("reason")
    lines = [
        "# {}".format(title),
        "",
        "All values below are calculated from `{}`; unavailable values are marked missing and are not estimated.".format(
            source.name
        ),
        "",
        "- Status: `{}`".format(details.get("status", "unknown")),
        "- Rows: `{}`".format(details.get("rows", 0)),
        "- Threshold strategy: `{}`".format(threshold_strategy or "all"),
    ]
    if reason:
        lines.append("- Reason: {}".format(reason))
    lines.extend(["", "## Mean Results", "", _markdown_table(table), ""])
    destination = Path(path)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def generate_report(
    output_dir: Union[str, Path],
    overall_metrics_csv: Union[str, Path],
    title: str = "Evaluation Report",
    threshold_strategy: Optional[str] = "per_class_thresholds",
    config_files: Sequence[Union[str, Path]] = (),
    environment_files: Sequence[Union[str, Path]] = (),
    history_files: Sequence[Union[str, Path]] = (),
    expected_artifacts: Optional[Mapping[str, str]] = None,
) -> Dict[str, Path]:
    """Generate ``REPORT.md``, ``summary.json``, copied inputs, and ``manifest.json``."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    copied = copy_reproducibility_artifacts(
        root, config_files=config_files, environment_files=environment_files, history_files=history_files
    )
    source = Path(overall_metrics_csv)
    summary_path = root / "summary.json"
    report_path = root / "REPORT.md"
    missing: Dict[str, str] = dict(expected_artifacts or {})
    missing.update(copied["missing"])
    if source.is_file():
        summary = summarize_metrics(source, threshold_strategy)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_markdown_report(report_path, source, title, threshold_strategy, summary)
    else:
        reason = "overall metrics CSV does not exist: {}".format(source)
        summary = {"status": "missing", "reason": reason, "rows": 0, "metrics": {}, "best": {}}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report_path.write_text(
            "# {}\n\n- Status: `missing`\n- Reason: {}\n".format(title, reason), encoding="utf-8"
        )
        missing["overall_metrics.csv"] = reason
    manifest_path = build_manifest(root, missing=missing)
    return {"markdown": report_path, "summary": summary_path, "manifest": manifest_path}


build_report = generate_report
create_manifest = build_manifest


__all__ = [
    "build_manifest",
    "build_report",
    "copy_reproducibility_artifacts",
    "create_manifest",
    "csv_row_count",
    "generate_report",
    "manifest_entry",
    "missing_manifest_entry",
    "sha256_file",
    "summarize_metrics",
    "write_markdown_report",
]
