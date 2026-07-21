"""Typed configuration for the independent evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _reject_unknown(section: str, value: Mapping[str, Any], cls: Any) -> None:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError("Unknown {} config keys: {}".format(section, ", ".join(unknown)))


def _string_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise TypeError("{} must be a string or sequence of strings".format(field_name))
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ValueError("{} cannot contain empty values".format(field_name))
    return result


@dataclass(frozen=True)
class ScenarioConfig:
    """One standardized NPZ evaluation scenario."""

    name: str
    path: str
    condition: Optional[str] = None
    snr: Optional[float] = None

    @classmethod
    def from_value(cls, value: Any, default_name: Optional[str] = None) -> "ScenarioConfig":
        if isinstance(value, (str, Path)):
            if default_name is None:
                default_name = Path(value).stem
            return cls(name=default_name, path=str(value), condition=default_name)
        if not isinstance(value, Mapping):
            raise TypeError("Each data scenario must be a path or mapping")
        _reject_unknown("scenario", value, cls)
        data = dict(value)
        if default_name is not None:
            data.setdefault("name", default_name)
        if "name" not in data or "path" not in data:
            raise ValueError("Each scenario requires name and path")
        data.setdefault("condition", data["name"])
        return cls(**data)


@dataclass(frozen=True)
class DataConfig:
    scenarios: Tuple[ScenarioConfig, ...]
    batch_size: int = 128
    num_workers: int = 0
    input_channels: int = 12
    ecg_layout: str = "NCT"
    require_ecg: bool = True
    require_features: bool = False
    validate_alignment: bool = True
    expected_feature_shape: Tuple[int, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DataConfig":
        _reject_unknown("data", value, cls)
        data = dict(value)
        raw_scenarios = data.pop("scenarios", ())
        data["expected_feature_shape"] = tuple(int(item) for item in
                                                 (data.get("expected_feature_shape") or ()))
        scenarios: List[ScenarioConfig] = []
        if isinstance(raw_scenarios, Mapping):
            scenarios = [ScenarioConfig.from_value(item, str(name))
                         for name, item in raw_scenarios.items()]
        elif isinstance(raw_scenarios, Sequence) and not isinstance(raw_scenarios, str):
            scenarios = [ScenarioConfig.from_value(item) for item in raw_scenarios]
        else:
            raise TypeError("data.scenarios must be a mapping or sequence")
        result = cls(scenarios=tuple(scenarios), **data)
        if not result.scenarios:
            raise ValueError("At least one data scenario is required")
        if result.batch_size < 1 or result.input_channels < 1 or result.num_workers < 0:
            raise ValueError("batch_size/input_channels must be positive and num_workers non-negative")
        if result.ecg_layout not in ("NCT", "NTC"):
            raise ValueError("data.ecg_layout must be NCT or NTC")
        names = [scenario.name for scenario in result.scenarios]
        if len(set(names)) != len(names):
            raise ValueError("Scenario names must be unique")
        return result


@dataclass(frozen=True)
class ModelConfig:
    """Model construction and checkpoint interpretation."""

    name: str
    adapter: str = "factory"
    architecture: Optional[str] = None
    factory: Optional[str] = None
    checkpoint: Optional[str] = None
    checkpoint_key: Optional[str] = None
    strip_prefixes: Tuple[str, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    num_classes: int = 5
    input_channels: int = 12
    activation: str = "sigmoid"
    call_mode: str = "single"
    input_key: str = "ecg"
    variant: Optional[str] = None
    strict_checkpoint: bool = True
    precomputed_path: Optional[str] = None
    prediction_key: Optional[str] = None
    crop_length: Optional[int] = None
    crop_stride: Optional[int] = None
    crop_aggregation: str = "max"
    allow_uninitialized: bool = False
    trusted_legacy_checkpoint: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelConfig":
        _reject_unknown("model", value, cls)
        data = dict(value)
        data["strip_prefixes"] = _string_tuple(data.get("strip_prefixes"),
                                                 "model.strip_prefixes")
        data["kwargs"] = dict(data.get("kwargs") or {})
        result = cls(**data)
        if result.num_classes < 1 or result.input_channels < 1:
            raise ValueError("num_classes and input_channels must be positive")
        if result.activation not in ("sigmoid", "softmax", "identity"):
            raise ValueError("model.activation must be sigmoid, softmax, or identity")
        if result.call_mode not in ("single", "feature_only", "late_fusion"):
            raise ValueError("model.call_mode must be single, feature_only, or late_fusion")
        if result.input_key not in ("ecg", "features"):
            raise ValueError("model.input_key must be ecg or features")
        if result.adapter == "precomputed_npz" and not result.precomputed_path:
            raise ValueError("precomputed_npz requires model.precomputed_path")
        if result.adapter == "factory" and not result.factory:
            raise ValueError("factory adapter requires model.factory as module:function")
        if result.crop_length is not None and result.crop_length < 1:
            raise ValueError("model.crop_length must be positive")
        if result.crop_stride is not None and result.crop_stride < 1:
            raise ValueError("model.crop_stride must be positive")
        if result.crop_aggregation not in ("max", "mean"):
            raise ValueError("model.crop_aggregation must be max or mean")
        return result


@dataclass(frozen=True)
class InferenceConfig:
    device: str = "cpu"
    warmup_batches: int = 1
    timing: str = "both"
    dtype: str = "float32"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InferenceConfig":
        _reject_unknown("inference", value, cls)
        result = cls(**dict(value))
        if result.warmup_batches < 0:
            raise ValueError("warmup_batches cannot be negative")
        if result.timing not in ("model", "end_to_end", "both"):
            raise ValueError("inference.timing must be model, end_to_end, or both")
        if result.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("inference.dtype must be float32, float16, or bfloat16")
        return result


@dataclass(frozen=True)
class ThresholdConfig:
    mode: str = "fixed_global"
    global_threshold: float = 0.5
    per_class: Tuple[float, ...] = ()
    file: Optional[str] = None
    source_split: str = "validation"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ThresholdConfig":
        _reject_unknown("threshold", value, cls)
        data = dict(value)
        per_class = data.get("per_class") or ()
        if isinstance(per_class, (int, float)):
            per_class = (float(per_class),)
        data["per_class"] = tuple(float(item) for item in per_class)
        result = cls(**data)
        if result.mode not in ("fixed_global", "fixed_per_class", "load_from_file"):
            raise ValueError("threshold.mode must be fixed_global, fixed_per_class, or load_from_file")
        if result.source_split.lower() == "test":
            raise ValueError("test-derived thresholds are forbidden")
        if result.mode == "fixed_per_class" and not result.per_class:
            raise ValueError("fixed_per_class requires threshold.per_class")
        if result.mode == "load_from_file" and not result.file:
            raise ValueError("load_from_file requires threshold.file")
        return result


@dataclass(frozen=True)
class AnalysisConfig:
    class_names: Tuple[str, ...] = ()
    thresholds: Tuple[float, ...] = ()
    metrics: bool = True
    calibration: bool = True
    robustness: bool = True
    bootstrap: int = 0
    calibration_bins: int = 10
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AnalysisConfig":
        _reject_unknown("analysis", value, cls)
        data = dict(value)
        data["class_names"] = _string_tuple(data.get("class_names"),
                                              "analysis.class_names")
        thresholds = data.get("thresholds") or ()
        if isinstance(thresholds, (int, float)):
            thresholds = (float(thresholds),)
        data["thresholds"] = tuple(float(item) for item in thresholds)
        data["threshold"] = ThresholdConfig.from_mapping(data.get("threshold", {}))
        result = cls(**data)
        if result.class_names and result.thresholds and len(result.thresholds) not in (
                1, len(result.class_names)):
            raise ValueError("thresholds must have length one or match class_names")
        if result.bootstrap < 0 or result.calibration_bins < 1:
            raise ValueError("bootstrap must be non-negative and calibration_bins positive")
        if not result.metrics:
            raise ValueError("standard classification metrics are mandatory and cannot be disabled")
        return result


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str
    output_dir: str
    seed: int = 42
    dataset_split: str = "test"
    overwrite: bool = False
    history_file: Optional[str] = None
    dataset_name: str = "unspecified"
    dataset_version: str = "unspecified"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunConfig":
        _reject_unknown("run", value, cls)
        result = cls(**dict(value))
        if not result.experiment_name or not result.output_dir:
            raise ValueError("run.experiment_name and run.output_dir are required")
        return result


@dataclass(frozen=True)
class OutputConfig:
    save_predictions: bool = True
    save_logits: bool = False
    save_plots: bool = True
    measure_efficiency: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OutputConfig":
        _reject_unknown("output", value, cls)
        return cls(**dict(value))


@dataclass(frozen=True)
class EvaluationConfig:
    run: RunConfig
    model: ModelConfig
    data: DataConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationConfig":
        _reject_unknown("root", value, cls)
        if "run" not in value or "model" not in value or "data" not in value:
            raise ValueError("Configuration requires run, model and data sections")
        result = cls(
            run=RunConfig.from_mapping(value["run"]),
            model=ModelConfig.from_mapping(value["model"]),
            data=DataConfig.from_mapping(value["data"]),
            inference=InferenceConfig.from_mapping(value.get("inference", {})),
            analysis=AnalysisConfig.from_mapping(value.get("analysis", {})),
            output=OutputConfig.from_mapping(value.get("output", {})),
        )
        if result.analysis.class_names and len(result.analysis.class_names) != result.model.num_classes:
            raise ValueError("class_names length does not match model.num_classes")
        if result.data.input_channels != result.model.input_channels:
            raise ValueError("data.input_channels does not match model.input_channels")
        return result


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_paths(value: Mapping[str, Any], base: Path) -> Dict[str, Any]:
    data = dict(value)
    run = dict(data.get("run") or {})
    for key in ("output_dir", "history_file"):
        if run.get(key):
            path = Path(str(run[key])).expanduser()
            run[key] = str(path if path.is_absolute() else base / path)
    data["run"] = run
    model = dict(data.get("model") or {})
    for key in ("checkpoint", "precomputed_path"):
        if model.get(key):
            path = Path(str(model[key])).expanduser()
            model[key] = str(path if path.is_absolute() else base / path)
    data["model"] = model
    data_section = dict(data.get("data") or {})
    scenarios = data_section.get("scenarios")
    if isinstance(scenarios, Mapping):
        resolved: Dict[str, Any] = {}
        for name, item in scenarios.items():
            if isinstance(item, Mapping):
                current = dict(item)
                path = Path(str(current["path"])).expanduser()
                current["path"] = str(path if path.is_absolute() else base / path)
                resolved[str(name)] = current
            else:
                path = Path(str(item)).expanduser()
                resolved[str(name)] = str(path if path.is_absolute() else base / path)
        data_section["scenarios"] = resolved
    elif isinstance(scenarios, Sequence) and not isinstance(scenarios, str):
        resolved_items: List[Any] = []
        for item in scenarios:
            if isinstance(item, Mapping):
                current = dict(item)
                path = Path(str(current["path"])).expanduser()
                current["path"] = str(path if path.is_absolute() else base / path)
                resolved_items.append(current)
            else:
                path = Path(str(item)).expanduser()
                resolved_items.append(str(path if path.is_absolute() else base / path))
        data_section["scenarios"] = resolved_items
    data["data"] = data_section
    analysis = dict(data.get("analysis") or {})
    threshold = dict(analysis.get("threshold") or {})
    if threshold.get("file"):
        path = Path(str(threshold["file"])).expanduser()
        threshold["file"] = str(path if path.is_absolute() else base / path)
    analysis["threshold"] = threshold
    data["analysis"] = analysis
    return data


def load_config(path: str, overrides: Optional[Mapping[str, Any]] = None) -> EvaluationConfig:
    """Load YAML and apply a nested override mapping before validation."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("YAML configuration requires PyYAML") from error
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError("The YAML root must be a mapping")
    merged = _deep_merge(loaded, overrides or {})
    return EvaluationConfig.from_mapping(_resolve_paths(merged, config_path.parent))


def config_from_mapping(value: Mapping[str, Any],
                        overrides: Optional[Mapping[str, Any]] = None) -> EvaluationConfig:
    return EvaluationConfig.from_mapping(_deep_merge(value, overrides or {}))
