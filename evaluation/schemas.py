"""Normalized data contracts for prediction and evaluation interchange."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _readonly(array: np.ndarray) -> np.ndarray:
    # Never change the caller's array flags while normalizing ownership.
    normalized = np.array(array, copy=True, order="C")
    normalized.setflags(write=False)
    return normalized


def _records(value: Any, name: str) -> Tuple[Dict[str, Any], ...]:
    if value is None:
        return tuple()
    if isinstance(value, pd.DataFrame):
        value = value.to_dict(orient="records")
    if isinstance(value, Mapping):
        value = [value]
    try:
        records = tuple(dict(record) for record in value)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be a DataFrame, mapping, or iterable of mappings".format(name)) from error
    return records


@dataclass(frozen=True)
class BatchMapping:
    """Map arbitrary batch keys to the normalized prediction schema."""

    probability_key: str = "y_prob"
    target_key: Optional[str] = "y_true"
    sample_id_key: Optional[str] = "sample_ids"

    def extract(self, batch: Mapping[str, Any]) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Extract probabilities, targets, and sample IDs from one batch."""
        if self.probability_key not in batch:
            raise KeyError("batch is missing probability key {!r}".format(self.probability_key))
        probability = batch[self.probability_key]
        target = batch[self.target_key] if self.target_key is not None and self.target_key in batch else None
        sample_ids = batch[self.sample_id_key] if self.sample_id_key is not None and self.sample_id_key in batch else None
        return probability, target, sample_ids


@dataclass(frozen=True)
class PredictionSet:
    """Validated, immutable-by-convention multilabel prediction arrays."""

    y_true: Optional[Any]
    y_prob: Any
    class_names: Optional[Sequence[str]] = None
    sample_ids: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        probability = np.asarray(self.y_prob, dtype=float)
        if probability.ndim != 2 or probability.shape[0] == 0 or probability.shape[1] == 0:
            raise ValueError("y_prob must have non-empty shape (n_samples, n_classes)")
        if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("y_prob must contain finite probabilities in [0, 1]")
        truth = None
        if self.y_true is not None:
            truth = np.asarray(self.y_true)
            if truth.shape != probability.shape or not np.isin(truth, (0, 1, False, True)).all():
                raise ValueError("y_true must be binary and have the same shape as y_prob")
            truth = _readonly(truth.astype(np.uint8, copy=False))
        names = tuple(str(name) for name in self.class_names) if self.class_names is not None else tuple(
            "class_{}".format(index) for index in range(probability.shape[1])
        )
        if len(names) != probability.shape[1] or len(set(names)) != len(names):
            raise ValueError("class_names must contain one unique name per class")
        ids = np.arange(probability.shape[0]) if self.sample_ids is None else np.asarray(self.sample_ids)
        if ids.ndim != 1 or len(ids) != probability.shape[0]:
            raise ValueError("sample_ids must be one-dimensional with one ID per sample")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("sample_ids must be unique")
        object.__setattr__(self, "y_prob", _readonly(probability.astype(float, copy=False)))
        object.__setattr__(self, "y_true", truth)
        object.__setattr__(self, "class_names", names)
        object.__setattr__(self, "sample_ids", _readonly(ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_samples(self) -> int:
        """Number of prediction rows."""
        return int(self.y_prob.shape[0])

    @property
    def n_classes(self) -> int:
        """Number of label classes."""
        return int(self.y_prob.shape[1])

    @property
    def probabilities(self) -> np.ndarray:
        """Descriptive alias for ``y_prob``."""
        return self.y_prob

    @property
    def labels(self) -> Optional[np.ndarray]:
        """Descriptive alias for ``y_true``."""
        return self.y_true

    @property
    def sample_id(self) -> np.ndarray:
        """Singular-name alias used by batch-oriented data adapters."""
        return self.sample_ids

    def to_mapping(self) -> Dict[str, Any]:
        """Return the normalized fields as a shallow mapping."""
        return {
            "sample_ids": self.sample_ids,
            "class_names": self.class_names,
            "y_true": self.y_true,
            "y_prob": self.y_prob,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        mapping: BatchMapping = BatchMapping(),
        class_names: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "PredictionSet":
        """Normalize one mapping using an explicit batch-key mapping."""
        probability, target, sample_ids = mapping.extract(data)
        return cls(target, probability, class_names, sample_ids, metadata or {})

    @classmethod
    def from_batches(
        cls,
        batches: Iterable[Mapping[str, Any]],
        mapping: BatchMapping = BatchMapping(),
        class_names: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "PredictionSet":
        """Concatenate mapped batches while rejecting partially labeled input."""
        probabilities, targets, identifiers = [], [], []
        target_presence, id_presence = [], []
        for batch in batches:
            probability, target, sample_ids = mapping.extract(batch)
            probabilities.append(np.asarray(probability))
            target_presence.append(target is not None)
            id_presence.append(sample_ids is not None)
            if target is not None:
                targets.append(np.asarray(target))
            if sample_ids is not None:
                identifiers.append(np.asarray(sample_ids))
        if not probabilities:
            raise ValueError("batches must not be empty")
        if any(target_presence) and not all(target_presence):
            raise ValueError("either every batch or no batch must contain targets")
        if any(id_presence) and not all(id_presence):
            raise ValueError("either every batch or no batch must contain sample IDs")
        y_true = np.concatenate(targets, axis=0) if targets else None
        sample_ids = np.concatenate(identifiers, axis=0) if identifiers else None
        return cls(y_true, np.concatenate(probabilities, axis=0), class_names, sample_ids, metadata or {})


@dataclass(frozen=True)
class EvaluationResult:
    """Normalized aggregate and tabular outputs for one evaluated prediction set."""

    overall_metrics: Mapping[str, float]
    per_class_metrics: Any = field(default_factory=tuple)
    calibration: Any = field(default_factory=tuple)
    robustness: Any = field(default_factory=tuple)
    thresholds: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overall = {str(name): float(value) for name, value in self.overall_metrics.items()}
        if not all(np.isfinite(value) or np.isnan(value) for value in overall.values()):
            raise ValueError("overall_metrics values must be finite or NaN")
        thresholds = None if self.thresholds is None else np.asarray(self.thresholds, dtype=float)
        if thresholds is not None:
            if thresholds.ndim == 0:
                thresholds = thresholds.reshape(1)
            if thresholds.ndim != 1 or not np.isfinite(thresholds).all():
                raise ValueError("thresholds must be a finite scalar or one-dimensional sequence")
            thresholds = _readonly(thresholds)
        object.__setattr__(self, "overall_metrics", overall)
        object.__setattr__(self, "per_class_metrics", _records(self.per_class_metrics, "per_class_metrics"))
        object.__setattr__(self, "calibration", _records(self.calibration, "calibration"))
        object.__setattr__(self, "robustness", _records(self.robustness, "robustness"))
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_mapping(self) -> Dict[str, Any]:
        """Return a serialization-friendly mapping of normalized values."""
        return {
            "overall_metrics": dict(self.overall_metrics),
            "per_class_metrics": [dict(row) for row in self.per_class_metrics],
            "calibration": [dict(row) for row in self.calibration],
            "robustness": [dict(row) for row in self.robustness],
            "thresholds": None if self.thresholds is None else self.thresholds.tolist(),
            "metadata": dict(self.metadata),
        }


# Descriptive alias for users who prefer the longer name.
PredictionBatchMapping = BatchMapping
