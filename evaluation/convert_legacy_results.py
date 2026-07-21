"""Convert measured legacy CSV values to the standardized evaluation schema."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

try:
    from .report import build_manifest
except ImportError:  # Direct execution: python evaluation/convert_legacy_results.py
    from report import build_manifest


METADATA_COLUMNS = (
    "experiment_name",
    "model_name",
    "seed",
    "ecg_scenario",
    "target_snr_db",
    "threshold_strategy",
)
OVERALL_METRIC_COLUMNS = (
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
)
COMPLEXITY_COLUMNS = (
    "parameter_count",
    "trainable_parameter_count",
    "training_time_seconds",
    "best_epoch",
    "best_valid_loss",
    "inference_time_per_sample_ms",
    "actual_batch_size",
)
PER_CLASS_COLUMNS = (
    "class_index",
    "class_name",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "specificity",
    "f1",
    "support_positive",
    "support_negative",
    "predicted_positive_count",
    "threshold",
)
PROVENANCE_COLUMNS = (
    "status",
    "reason",
    "missing_fields",
    "source_file",
    "source_row",
    "source_statistic",
)
ALIASES = {
    "experiment": "experiment_name",
    "model": "model_name",
    "scenario": "ecg_scenario",
    "snr": "target_snr_db",
    "thresholding_strategy": "threshold_strategy",
    "macro_auc": "macro_roc_auc",
    "macro_auroc": "macro_roc_auc",
    "micro_auroc": "micro_roc_auc",
    "macro_auprc": "macro_pr_auc",
    "micro_auprc": "micro_pr_auc",
    "class": "class_name",
    "label": "class_name",
    "auroc": "roc_auc",
    "auc": "roc_auc",
    "auprc": "pr_auc",
    "parameters": "parameter_count",
    "num_parameters": "parameter_count",
    "trainable_parameters": "trainable_parameter_count",
}
LEGACY_PRESERVED = {
    "fmax": "legacy_fmax",
    "f_beta_macro": "legacy_f_beta_macro",
    "g_beta_macro": "legacy_g_beta_macro",
}


def _normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    for old, new in ALIASES.items():
        if old in result and new not in result:
            result = result.rename(columns={old: new})
    for old, new in LEGACY_PRESERVED.items():
        if old in result and new not in result:
            result = result.rename(columns={old: new})
    return result


def _legacy_rows(path: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    frame = pd.read_csv(path)
    first = str(frame.columns[0]).strip().lower() if len(frame.columns) else ""
    statistics = {"point", "mean", "lower", "upper"}
    if first.startswith("unnamed") or first in ("index", "statistic"):
        values = frame.iloc[:, 0].astype(str).str.lower()
        if set(values).intersection(statistics):
            frame = frame.copy()
            frame["source_statistic"] = values
            point = frame[values == "point"]
            if len(point):
                return point.drop(columns=frame.columns[0]).reset_index(drop=True), "point"
            mean = frame[values == "mean"]
            if len(mean):
                return mean.drop(columns=frame.columns[0]).reset_index(drop=True), "mean"
            return frame.iloc[0:0].drop(columns=frame.columns[0]), None
    return frame, None


def _kind(frame: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    names = {ALIASES.get(str(column).strip().lower(), str(column).strip().lower()) for column in frame.columns}
    return "per-class" if "class_name" in names else "overall"


def _provided_metadata(
    experiment_name: Optional[str],
    model_name: Optional[str],
    seed: Optional[str],
    ecg_scenario: Optional[str],
    target_snr_db: Optional[float],
    threshold_strategy: Optional[str],
) -> Dict[str, Any]:
    return {
        key: value
        for key, value in {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "seed": seed,
            "ecg_scenario": ecg_scenario,
            "target_snr_db": target_snr_db,
            "threshold_strategy": threshold_strategy,
        }.items()
        if value is not None
    }


def _convert_frame(
    frame: pd.DataFrame,
    source: Path,
    kind: str,
    metadata: Mapping[str, Any],
    selected_statistic: Optional[str],
) -> pd.DataFrame:
    normalized = _normalized_columns(frame)
    metric_columns = list(PER_CLASS_COLUMNS if kind == "per-class" else OVERALL_METRIC_COLUMNS + COMPLEXITY_COLUMNS)
    output_columns = list(METADATA_COLUMNS) + metric_columns
    preserved = [column for column in LEGACY_PRESERVED.values() if column in normalized]
    records = []
    for position, (_, source_row) in enumerate(normalized.iterrows()):
        record: Dict[str, Any] = {column: pd.NA for column in output_columns}
        for column in output_columns:
            if column in normalized and not pd.isna(source_row[column]):
                record[column] = source_row[column]
        for column, value in metadata.items():
            if column in record and pd.isna(record[column]):
                record[column] = value
        if pd.isna(record.get("experiment_name")) and not pd.isna(record.get("model_name")):
            record["experiment_name"] = record["model_name"]
        if pd.isna(record.get("model_name")) and not pd.isna(record.get("experiment_name")):
            record["model_name"] = record["experiment_name"]

        required_metadata = ("experiment_name", "seed", "ecg_scenario", "threshold_strategy")
        required_metrics = list(PER_CLASS_COLUMNS if kind == "per-class" else OVERALL_METRIC_COLUMNS)
        missing = [
            column
            for column in list(required_metadata) + required_metrics
            if column not in record or pd.isna(record[column])
        ]
        record["status"] = "partial" if missing else "complete"
        record["reason"] = (
            "legacy source lacks standardized fields: {}".format(", ".join(missing))
            if missing
            else None
        )
        record["missing_fields"] = json.dumps(missing)
        record["source_file"] = str(source.resolve())
        record["source_row"] = int(position)
        record["source_statistic"] = (
            source_row.get("source_statistic", selected_statistic)
            if "source_statistic" in normalized
            else selected_statistic
        )
        for column in preserved:
            record[column] = source_row[column]
        records.append(record)
    return pd.DataFrame(records, columns=output_columns + preserved + list(PROVENANCE_COLUMNS))


def _source_files(sources: Sequence[Union[str, Path]], output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    destination = output_dir.resolve()
    for value in sources:
        source = Path(value).expanduser()
        if source.is_file():
            paths.append(source)
        elif source.is_dir():
            paths.extend(
                path
                for path in sorted(source.rglob("*.csv"))
                if destination not in path.resolve().parents
            )
        else:
            raise FileNotFoundError(str(source))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise FileNotFoundError("no legacy CSV files found")
    return unique


def convert_legacy_results(
    sources: Sequence[Union[str, Path]],
    output_dir: Union[str, Path],
    kind: str = "auto",
    experiment_name: Optional[str] = None,
    model_name: Optional[str] = None,
    seed: Optional[str] = None,
    ecg_scenario: Optional[str] = None,
    target_snr_db: Optional[float] = None,
    threshold_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert CSVs while leaving unavailable standardized fields empty and marked."""
    if kind not in ("auto", "overall", "per-class"):
        raise ValueError("kind must be auto, overall, or per-class")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = _provided_metadata(
        experiment_name, model_name, seed, ecg_scenario, target_snr_db, threshold_strategy
    )
    overall_frames, class_frames, statuses = [], [], []
    for source in _source_files(sources, destination):
        try:
            frame, statistic = _legacy_rows(source)
            detected = _kind(frame, kind)
            converted = _convert_frame(frame, source, detected, metadata, statistic)
            if detected == "overall":
                overall_frames.append(converted)
            else:
                class_frames.append(converted)
            partial = bool(len(converted) and (converted["status"] != "complete").any())
            statuses.append(
                {
                    "source_file": str(source.resolve()),
                    "kind": detected,
                    "rows_written": int(len(converted)),
                    "status": "partial" if partial else "complete",
                    "reason": "; ".join(converted["reason"].dropna().astype(str).unique()) if partial else None,
                }
            )
        except Exception as error:
            statuses.append(
                {
                    "source_file": str(source.resolve()),
                    "kind": kind,
                    "rows_written": 0,
                    "status": "missing",
                    "reason": "conversion failed: {}".format(error),
                }
            )

    outputs: Dict[str, Any] = {}
    if overall_frames:
        overall = pd.concat(overall_frames, ignore_index=True, sort=False)
        overall_path = destination / "overall_metrics.csv"
        overall.to_csv(overall_path, index=False)
        outputs["overall_metrics"] = overall_path
    if class_frames:
        per_class = pd.concat(class_frames, ignore_index=True, sort=False)
        class_path = destination / "per_class_metrics.csv"
        per_class.to_csv(class_path, index=False)
        outputs["per_class_metrics"] = class_path
    status_path = destination / "conversion_status.csv"
    pd.DataFrame(statuses).to_csv(status_path, index=False)
    outputs["status"] = status_path
    outputs["manifest"] = build_manifest(destination)
    return outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", type=Path, help="Legacy CSV files or directories")
    parser.add_argument("--input", dest="input_sources", action="append", type=Path)
    parser.add_argument("-o", "--output", "--output-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("auto", "overall", "per-class"), default="auto")
    parser.add_argument("--experiment-name")
    parser.add_argument("--model-name")
    parser.add_argument("--seed")
    parser.add_argument("--ecg-scenario")
    parser.add_argument("--target-snr-db", type=float)
    parser.add_argument("--threshold-strategy")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = convert_legacy_results(
        list(args.sources) + list(args.input_sources or []),
        args.output_dir,
        kind=args.kind,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        seed=args.seed,
        ecg_scenario=args.ecg_scenario,
        target_snr_db=args.target_snr_db,
        threshold_strategy=args.threshold_strategy,
    )
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPLEXITY_COLUMNS",
    "METADATA_COLUMNS",
    "OVERALL_METRIC_COLUMNS",
    "PER_CLASS_COLUMNS",
    "convert_legacy_results",
    "main",
    "parse_args",
]
