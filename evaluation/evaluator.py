"""Inference orchestration and analysis-facing evaluation records."""

from __future__ import annotations

import importlib
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .config import EvaluationConfig
    from .data import NPZDataAdapter, ScenarioData
    from .model_registry import ModelAdapter, build_model_adapter
except ImportError:  # Support ``python evaluation/evaluate.py``.
    from config import EvaluationConfig
    from data import NPZDataAdapter, ScenarioData
    from model_registry import ModelAdapter, build_model_adapter


@dataclass(frozen=True)
class EvaluationRecord:
    """Predictions and timing for one model/scenario pair."""

    model_name: str
    scenario_name: str
    probabilities: np.ndarray
    logits: Optional[np.ndarray]
    labels: np.ndarray
    sample_id: np.ndarray
    snr: np.ndarray
    condition: np.ndarray
    model_seconds: float
    end_to_end_seconds: float
    batch_count: int
    batch_losses: np.ndarray
    efficiency: Mapping[str, Any]

    @property
    def sample_count(self) -> int:
        return len(self.sample_id)

    @property
    def model_ms_per_sample(self) -> float:
        return 1000.0 * self.model_seconds / self.sample_count

    @property
    def end_to_end_ms_per_sample(self) -> float:
        return 1000.0 * self.end_to_end_seconds / self.sample_count

    def metric_inputs(self) -> Dict[str, Any]:
        return {
            "y_true": self.labels,
            "y_prob": self.probabilities,
            "sample_id": self.sample_id,
            "snr": self.snr,
            "condition": self.condition,
        }


@dataclass(frozen=True)
class AnalysisResults:
    metrics: Mapping[str, Any]
    calibration: Mapping[str, Any]
    robustness: Any


@dataclass(frozen=True)
class EvaluationRun:
    records: Tuple[EvaluationRecord, ...]
    analyses: Optional[AnalysisResults] = None


def _analysis_module(name: str) -> Any:
    package = __package__
    try:
        if package:
            return importlib.import_module(".{}".format(name), package=package)
        repository_root = str(Path(__file__).resolve().parents[1])
        if repository_root not in sys.path:
            sys.path.insert(0, repository_root)
        return importlib.import_module("evaluation.{}".format(name))
    except ImportError as error:
        raise RuntimeError(
            "Analysis module evaluation.{} is unavailable; prediction records are still "
            "available through Evaluator.collect()".format(name)) from error


class Evaluator:
    """Collect predictions first, then optionally invoke analysis modules.

    The analysis contract is deliberately small: ``metrics.compute_metrics`` and
    ``calibration.compute_calibration`` receive array keyword arguments per
    record; ``robustness.compute_robustness`` receives all records together.
    """

    def __init__(self, config: EvaluationConfig,
                 model: Optional[ModelAdapter] = None,
                 data: Optional[NPZDataAdapter] = None):
        self.config = config
        random.seed(config.run.seed)
        np.random.seed(config.run.seed)
        try:
            import torch
            torch.manual_seed(config.run.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.run.seed)
        except ImportError:
            pass
        self.model = model or build_model_adapter(config.model, config.inference.device,
                                                   config.inference.dtype)
        self.data = data or NPZDataAdapter(config.data, config.model.num_classes)
        self.records: List[EvaluationRecord] = []

    def _warm_up(self, scenarios: Sequence[ScenarioData]) -> None:
        if self.config.inference.warmup_batches == 0:
            return
        if not scenarios:
            raise ValueError("No evaluation scenarios were loaded")
        first_batch = next(scenarios[0].batches(self.config.data.batch_size), None)
        if first_batch is None:
            raise ValueError("Cannot warm up on an empty scenario")
        for _ in range(self.config.inference.warmup_batches):
            self.model.predict_batch(first_batch)

    def collect(self) -> Tuple[EvaluationRecord, ...]:
        scenarios = self.data.load_scenarios()
        self._warm_up(scenarios)
        collected: List[EvaluationRecord] = []
        for scenario in scenarios:
            prediction_batches: List[np.ndarray] = []
            logit_batches: List[np.ndarray] = []
            batch_losses: List[float] = []
            model_seconds = 0.0
            batch_count = 0
            scenario_started = time.perf_counter()
            for batch in scenario.batches(self.config.data.batch_size):
                prediction = self.model.predict_batch(batch)
                values = np.asarray(prediction.probabilities)
                expected = (len(batch["sample_id"]), self.config.model.num_classes)
                if values.shape != expected:
                    raise ValueError("Adapter returned shape {}; expected {} for scenario {}".format(
                        values.shape, expected, scenario.name))
                prediction_batches.append(values)
                if prediction.logits is not None:
                    logit_batches.append(np.asarray(prediction.logits))
                clipped = np.clip(values, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
                labels = np.asarray(batch["labels"])
                batch_losses.append(float(-np.mean(labels * np.log(clipped) +
                                                   (1 - labels) * np.log(1 - clipped))))
                model_seconds += prediction.model_seconds
                batch_count += 1
            end_to_end_seconds = time.perf_counter() - scenario_started
            if not prediction_batches:
                raise ValueError("Scenario {} contains no batches".format(scenario.name))
            probabilities = np.concatenate(prediction_batches, axis=0)
            logits = np.concatenate(logit_batches, axis=0) if logit_batches else None
            if logit_batches and logits.shape != probabilities.shape:
                raise ValueError("Logit shape does not match probabilities")
            if not np.isfinite(probabilities).all():
                raise ValueError("Scenario {} predictions contain non-finite values".format(
                    scenario.name))
            timing = self.config.inference.timing
            record = EvaluationRecord(
                model_name=self.config.model.name,
                scenario_name=scenario.name,
                probabilities=probabilities,
                logits=logits,
                labels=scenario.labels,
                sample_id=scenario.sample_id,
                snr=scenario.snr,
                condition=scenario.condition,
                model_seconds=model_seconds if timing in ("model", "both") else float("nan"),
                end_to_end_seconds=end_to_end_seconds if timing in ("end_to_end", "both") else float("nan"),
                batch_count=batch_count,
                batch_losses=np.asarray(batch_losses, dtype=float),
                efficiency=dict(self.model.efficiency_metadata()),
            )
            collected.append(record)
        self.records = collected
        return tuple(collected)

    def analyze(self, records: Optional[Sequence[EvaluationRecord]] = None) -> AnalysisResults:
        selected = tuple(records if records is not None else self.records)
        if not selected:
            raise ValueError("No prediction records are available; call collect() first")
        analysis = self.config.analysis
        if analysis.thresholds:
            thresholds: Any = analysis.thresholds[0] if len(analysis.thresholds) == 1 else analysis.thresholds
        else:
            threshold_module = _analysis_module("metrics")
            threshold = analysis.threshold
            manager = threshold_module.ThresholdManager(
                threshold.mode, threshold.global_threshold, threshold.per_class or None,
                threshold.file, threshold.source_split)
            thresholds = manager.resolve(self.config.model.num_classes,
                                         analysis.class_names or None)
        common = {
            "class_names": analysis.class_names or None,
            "thresholds": thresholds,
        }
        metric_results: Dict[str, Any] = {}
        calibration_results: Dict[str, Any] = {}
        if analysis.metrics:
            metric_function = getattr(_analysis_module("metrics"), "compute_metrics")
            for record in selected:
                metric_results[record.scenario_name] = metric_function(
                    y_true=record.labels,
                    y_prob=record.probabilities,
                    class_names=common["class_names"],
                    thresholds=common["thresholds"],
                    batch_losses=record.batch_losses,
                )
        if analysis.calibration:
            calibration_function = getattr(
                _analysis_module("calibration"), "compute_calibration")
            for record in selected:
                calibration_results[record.scenario_name] = calibration_function(
                    y_true=record.labels,
                    y_prob=record.probabilities,
                    class_names=common["class_names"],
                )
        robustness_result = None
        if analysis.robustness:
            robustness_function = getattr(
                _analysis_module("robustness"), "compute_robustness")
            robustness_result = robustness_function(records=selected,
                                                     class_names=common["class_names"])
        return AnalysisResults(metric_results, calibration_results, robustness_result)

    def run(self, analyze: bool = True) -> EvaluationRun:
        records = self.collect()
        return EvaluationRun(records, self.analyze(records) if analyze else None)
