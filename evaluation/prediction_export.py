"""Framework-independent, ordered prediction exports."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, Sequence[Any]]


def _matrix(name: str, values: ArrayLike, rows: int, columns: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (rows, columns):
        raise ValueError(
            "{} must have shape ({}, {}), got {}".format(name, rows, columns, array.shape)
        )
    return array


def _binary_predictions(
    probabilities: np.ndarray,
    predictions: Optional[ArrayLike],
    thresholds: Optional[ArrayLike],
) -> np.ndarray:
    if predictions is not None:
        result = np.asarray(predictions)
        if result.shape != probabilities.shape:
            raise ValueError(
                "predictions must have shape {}, got {}".format(
                    probabilities.shape, result.shape
                )
            )
        return result.astype(np.int8)
    if thresholds is None:
        raise ValueError("predictions or thresholds must be provided")
    threshold_array = np.asarray(thresholds, dtype=float)
    if threshold_array.ndim == 0:
        threshold_array = np.repeat(threshold_array, probabilities.shape[1])
    if threshold_array.shape != (probabilities.shape[1],):
        raise ValueError(
            "thresholds must be scalar or have one value per class, got {}".format(
                threshold_array.shape
            )
        )
    return (probabilities >= threshold_array).astype(np.int8)


def _extra_column(values: Any, rows: int) -> np.ndarray:
    if isinstance(values, (str, bytes)) or np.isscalar(values):
        return np.repeat(values, rows)
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != rows:
        raise ValueError("extra columns must be scalar or have one value per sample")
    return array


def export_sample_predictions(
    output_dir: Union[str, Path],
    sample_ids: ArrayLike,
    y_true: ArrayLike,
    y_prob: ArrayLike,
    class_names: Sequence[str],
    y_pred: Optional[ArrayLike] = None,
    thresholds: Optional[ArrayLike] = None,
    extra_columns: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    logits: Optional[ArrayLike] = None,
    basename: str = "predictions",
    id_column: str = "sample_id",
) -> Dict[str, Path]:
    """Write matching CSV and NPZ files without changing sample or class order.

    The CSV is intentionally compact and contains sample-level summaries. Full
    class arrays, optional logits, thresholds and class names are stored in NPZ.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not basename or Path(basename).name != basename:
        raise ValueError("basename must be a non-empty filename stem")

    ids = np.asarray(sample_ids)
    if ids.ndim != 1:
        raise ValueError("sample_ids must be one-dimensional")
    names = [str(name) for name in class_names]
    if not names or len(set(names)) != len(names):
        raise ValueError("class_names must be non-empty and unique")
    if any(not name for name in names):
        raise ValueError("class names must not be empty")

    rows, columns = len(ids), len(names)
    truth = _matrix("y_true", y_true, rows, columns).astype(np.int8)
    probabilities = _matrix("y_prob", y_prob, rows, columns).astype(float)
    predictions = _binary_predictions(probabilities, y_pred, thresholds)

    if not id_column:
        raise ValueError("id_column must not be empty")
    frame = pd.DataFrame({id_column: ids})
    for name, values in (extra_columns or {}).items():
        if name == id_column or name in frame:
            raise ValueError("duplicate prediction CSV column: {}".format(name))
        frame[str(name)] = _extra_column(values, rows)
    true_sets = [[names[index] for index in np.flatnonzero(row)] for row in truth]
    predicted_sets = [[names[index] for index in np.flatnonzero(row)] for row in predictions]
    tp = np.sum((truth == 1) & (predictions == 1), axis=1)
    true_count = truth.sum(axis=1)
    predicted_count = predictions.sum(axis=1)
    frame["true_labels"] = [json.dumps(values) for values in true_sets]
    frame["predicted_labels"] = [json.dumps(values) for values in predicted_sets]
    frame["true_label_count"] = true_count
    frame["predicted_label_count"] = predicted_count
    frame["exact_match"] = np.all(truth == predictions, axis=1)
    precision = np.divide(tp, predicted_count, out=np.zeros(rows, dtype=float), where=predicted_count != 0)
    recall = np.divide(tp, true_count, out=np.zeros(rows, dtype=float), where=true_count != 0)
    frame["sample_precision"] = precision
    frame["sample_recall"] = recall
    frame["sample_f1"] = np.divide(2 * precision * recall, precision + recall,
                                    out=np.zeros(rows, dtype=float), where=(precision + recall) != 0)
    clipped = np.clip(probabilities, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    frame["loss"] = -np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped), axis=1)

    csv_path = destination / (basename + ".csv")
    npz_path = destination / (basename + ".npz")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=str(destination), delete=False, newline=""
    ) as temporary:
        temporary_csv = Path(temporary.name)
        frame.to_csv(temporary, index=False)
    os.replace(str(temporary_csv), str(csv_path))

    archive: Dict[str, np.ndarray] = {
        "sample_id": ids.astype(str) if ids.dtype == object else ids,
        "sample_ids": ids.astype(str) if ids.dtype == object else ids,
        "class_names": np.asarray(names, dtype=str),
        "labels": truth,
        "y_true": truth,
        "probabilities": probabilities,
        "y_prob": probabilities,
        "predictions": predictions,
        "y_pred": predictions,
    }
    for name, values in (extra_columns or {}).items():
        array = _extra_column(values, rows)
        archive[str(name)] = array.astype(str) if array.dtype == object else array
    if thresholds is not None:
        archive["thresholds"] = np.asarray(thresholds, dtype=float)
    if logits is not None:
        archive["logits"] = _matrix("logits", logits, rows, columns).astype(float)
    if metadata is not None:
        archive["metadata_json"] = np.asarray(
            json.dumps(dict(metadata), sort_keys=True, default=str)
        )
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", dir=str(destination), delete=False
    ) as temporary:
        temporary_npz = Path(temporary.name)
        np.savez_compressed(temporary, **archive)
    os.replace(str(temporary_npz), str(npz_path))
    return {"csv": csv_path, "npz": npz_path}


def export_predictions(
    output_dir: Union[str, Path],
    sample_ids: ArrayLike,
    y_true: ArrayLike,
    y_prob: ArrayLike,
    class_names: Sequence[str],
    y_pred: Optional[ArrayLike] = None,
    thresholds: Optional[ArrayLike] = None,
    extra_columns: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    logits: Optional[ArrayLike] = None,
    basename: str = "predictions",
    id_column: str = "sample_id",
) -> Dict[str, Path]:
    """Alias with the concise name used by evaluation runners."""
    return export_sample_predictions(
        output_dir=output_dir,
        sample_ids=sample_ids,
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        y_pred=y_pred,
        thresholds=thresholds,
        extra_columns=extra_columns,
        metadata=metadata,
        logits=logits,
        basename=basename,
        id_column=id_column,
    )


__all__ = ["export_predictions", "export_sample_predictions"]
