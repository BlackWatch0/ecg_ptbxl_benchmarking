"""Validated NumPy data adapter for standardized evaluation scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

try:
    from .config import DataConfig, ScenarioConfig
except ImportError:  # Support ``python evaluation/evaluate.py``.
    from config import DataConfig, ScenarioConfig


Batch = Dict[str, Any]
BATCH_KEYS = ("ecg", "features", "labels", "sample_id", "snr", "condition")


def _sample_vector(value: Any, count: int, name: str, default: Any) -> np.ndarray:
    if value is None:
        value = default
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), count)
    if array.ndim != 1 or len(array) != count:
        raise ValueError("{} must be scalar or have shape ({},), got {}".format(
            name, count, array.shape))
    return array


def _finite(name: str, array: Optional[np.ndarray]) -> None:
    if array is not None and np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ValueError("{} contains NaN or infinite values".format(name))


@dataclass(frozen=True)
class ScenarioData:
    name: str
    ecg: Optional[np.ndarray]
    features: Optional[np.ndarray]
    labels: np.ndarray
    sample_id: np.ndarray
    snr: np.ndarray
    condition: np.ndarray

    def __len__(self) -> int:
        return len(self.sample_id)

    def batches(self, batch_size: int) -> Iterator[Batch]:
        for start in range(0, len(self), batch_size):
            selection = slice(start, min(start + batch_size, len(self)))
            yield {
                "ecg": None if self.ecg is None else self.ecg[selection],
                "features": None if self.features is None else self.features[selection],
                "labels": self.labels[selection],
                "sample_id": self.sample_id[selection],
                "snr": self.snr[selection],
                "condition": self.condition[selection],
            }


class NPZDataAdapter:
    """Load scenarios using the fixed batch-dictionary contract.

    NPZ files require ``labels`` and ``sample_id``. They normally also contain
    channels-first ``ecg``; ``features``, ``snr``, and ``condition`` are
    optional and the latter two may be supplied by the scenario configuration.
    """

    def __init__(self, config: DataConfig, num_classes: int):
        self.config = config
        self.num_classes = num_classes

    def load(self, spec: ScenarioConfig) -> ScenarioData:
        path = Path(spec.path).expanduser()
        if not path.is_file():
            raise FileNotFoundError("Scenario NPZ not found: {}".format(path))
        with np.load(str(path), allow_pickle=False) as archive:
            required = {"labels", "sample_id"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError("{} is missing NPZ keys {}".format(path, missing))
            labels = np.asarray(archive["labels"])
            sample_id = np.asarray(archive["sample_id"])
            ecg = np.asarray(archive["ecg"]) if "ecg" in archive else None
            features = np.asarray(archive["features"]) if "features" in archive else None
            snr_value = np.asarray(archive["snr"]) if "snr" in archive else None
            condition_value = np.asarray(archive["condition"]) if "condition" in archive else None

        if sample_id.ndim != 1 or len(sample_id) == 0:
            raise ValueError("{} sample_id must be a non-empty 1-D array".format(path))
        count = len(sample_id)
        if len(np.unique(sample_id)) != count:
            raise ValueError("{} contains duplicate sample_id values".format(path))
        if labels.shape != (count, self.num_classes):
            raise ValueError("{} labels must have shape ({}, {}), got {}".format(
                path, count, self.num_classes, labels.shape))
        _finite("labels in {}".format(path), labels)
        if not np.isin(labels, (0, 1, False, True)).all():
            raise ValueError("{} labels must be binary".format(path))

        if ecg is None:
            if self.config.require_ecg:
                raise ValueError("{} is missing required NPZ key ecg".format(path))
        else:
            if ecg.ndim != 3 or ecg.shape[0] != count:
                raise ValueError("{} ecg must have shape [N,C,T] or [N,T,C], got {}".format(
                    path, ecg.shape))
            if self.config.ecg_layout == "NTC":
                ecg = np.transpose(ecg, (0, 2, 1))
            if ecg.shape[1] != self.config.input_channels:
                raise ValueError("{} ecg has {} channels; expected {}".format(
                    path, ecg.shape[1], self.config.input_channels))
            _finite("ecg in {}".format(path), ecg)
            ecg = np.asarray(ecg, dtype=np.float32)

        if features is None:
            if self.config.require_features:
                raise ValueError("{} is missing required NPZ key features".format(path))
        else:
            if features.ndim < 2 or features.shape[0] != count:
                raise ValueError("{} features must start with sample dimension {}, got {}".format(
                    path, count, features.shape))
            _finite("features in {}".format(path), features)
            if self.config.expected_feature_shape and tuple(features.shape[1:]) != tuple(
                    self.config.expected_feature_shape):
                raise ValueError("{} feature shape {} does not match expected {}".format(
                    path, tuple(features.shape[1:]), self.config.expected_feature_shape))
            features = np.asarray(features, dtype=np.float32)

        snr = _sample_vector(snr_value, count, "snr", spec.snr)
        condition = _sample_vector(condition_value, count, "condition",
                                   spec.condition or spec.name)
        if np.issubdtype(snr.dtype, np.number) and np.isinf(snr).any():
            raise ValueError("snr in {} contains infinite values".format(path))
        if condition.dtype.kind == "O" or sample_id.dtype.kind == "O":
            raise ValueError("{} uses object arrays; save IDs/conditions as numeric or Unicode".format(path))

        return ScenarioData(
            name=spec.name,
            ecg=ecg,
            features=features,
            labels=np.asarray(labels, dtype=np.float32),
            sample_id=sample_id,
            snr=snr,
            condition=condition,
        )

    def load_scenarios(self) -> Tuple[ScenarioData, ...]:
        scenarios: List[ScenarioData] = [self.load(spec) for spec in self.config.scenarios]
        if self.config.validate_alignment and len(scenarios) > 1:
            reference = scenarios[0]
            for scenario in scenarios[1:]:
                if not np.array_equal(scenario.sample_id, reference.sample_id):
                    raise ValueError("Scenario {} sample IDs/order do not match {}".format(
                        scenario.name, reference.name))
                if not np.array_equal(scenario.labels, reference.labels):
                    raise ValueError("Scenario {} labels do not match {}".format(
                        scenario.name, reference.name))
        return tuple(scenarios)

    def iter_batches(self) -> Iterator[Tuple[ScenarioData, Batch]]:
        for scenario in self.load_scenarios():
            for batch in scenario.batches(self.config.batch_size):
                if tuple(batch.keys()) != BATCH_KEYS:
                    raise RuntimeError("Internal batch contract violation")
                yield scenario, batch
