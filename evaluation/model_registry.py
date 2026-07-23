"""Lazy model registry and inference adapters."""

from __future__ import annotations

import importlib
import platform
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .config import ModelConfig
except ImportError:  # Support ``python evaluation/evaluate.py``.
    from config import ModelConfig


class ClassOutputMismatch(ValueError):
    """Raised when checkpoint or runtime outputs do not match configured classes."""


@dataclass(frozen=True)
class BatchPrediction:
    probabilities: np.ndarray
    model_seconds: float
    logits: Optional[np.ndarray] = None


class ModelAdapter:
    """Minimal adapter interface consumed by :class:`Evaluator`."""

    num_classes: int

    def predict_batch(self, batch: Mapping[str, Any]) -> BatchPrediction:
        raise NotImplementedError

    def efficiency_metadata(self) -> Mapping[str, Any]:
        return {}


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("This model adapter requires PyTorch") from error
    return torch


def _import_symbol(reference: str) -> Callable[..., Any]:
    if ":" not in reference:
        raise ValueError("Factory must use module:function syntax: {!r}".format(reference))
    module_name, symbol_name = reference.split(":", 1)
    if not module_name or not symbol_name:
        raise ValueError("Factory must use module:function syntax: {!r}".format(reference))
    # Existing model modules import each other as top-level ``models``.
    code_root = str(Path(__file__).resolve().parents[1] / "code")
    if module_name == "models" or module_name.startswith("models."):
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
    module = importlib.import_module(module_name)
    value = getattr(module, symbol_name, None)
    if value is None or not callable(value):
        raise ValueError("Factory {!r} is not callable".format(reference))
    return value


def _load_torch_file(path: str, device: Any, trusted_legacy: bool = False) -> Any:
    torch = _require_torch()
    try:
        return torch.load(path, map_location=device, weights_only=not trusted_legacy)
    except TypeError:
        if not trusted_legacy:
            raise RuntimeError("This PyTorch version cannot safely load weights-only checkpoints; "
                               "set trusted_legacy_checkpoint only for a trusted artifact")
        return torch.load(path, map_location=device)


def _mapping_at_key(value: Any, dotted_key: str) -> Any:
    current = value
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError("Checkpoint does not contain key {!r}".format(dotted_key))
        current = current[part]
    return current


def _extract_state_dict(checkpoint: Any, checkpoint_key: Optional[str]) -> Mapping[str, Any]:
    selected = _mapping_at_key(checkpoint, checkpoint_key) if checkpoint_key else checkpoint
    if checkpoint_key is None and isinstance(selected, Mapping):
        if "state_dict" in selected:
            selected = selected["state_dict"]
        elif "model" in selected:
            selected = selected["model"]
    if hasattr(selected, "state_dict") and callable(selected.state_dict):
        selected = selected.state_dict()
    if not isinstance(selected, Mapping) or not selected:
        raise TypeError("Checkpoint must be a module or a non-empty model/state_dict/raw mapping")
    torch = _require_torch()
    non_tensors = [str(key) for key, item in selected.items() if not torch.is_tensor(item)]
    if non_tensors:
        raise TypeError("Selected checkpoint mapping has non-tensor entries: {}".format(
            ", ".join(non_tensors[:10])))
    return selected


def _strip_prefixes(state: Mapping[str, Any], prefixes: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for original_key, value in state.items():
        key = str(original_key)
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in result:
            raise ValueError("Prefix stripping produced duplicate checkpoint key {!r}".format(key))
        result[key] = value
    return result


def _metadata_class_count(checkpoint: Any) -> Optional[int]:
    if not isinstance(checkpoint, Mapping):
        return None
    sources = [checkpoint]
    for key in ("config", "hyper_parameters", "hparams"):
        if isinstance(checkpoint.get(key), Mapping):
            sources.append(checkpoint[key])
    for source in sources:
        for key in ("num_classes", "n_classes", "class_count"):
            value = source.get(key)
            if isinstance(value, (int, np.integer)):
                return int(value)
    return None


def _is_output_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in (
        "output", "classifier", "classification", "head", "logits", "final", "fc."
    ))


def load_checkpoint_strict(model: Any, path: str, device: Any, num_classes: int,
                           checkpoint_key: Optional[str] = None,
                            strip_prefixes: Sequence[str] = (), strict: bool = True,
                            trusted_legacy: bool = False) -> None:
    """Load model/state_dict/raw checkpoints without implicit key rewriting."""

    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    checkpoint = _load_torch_file(str(checkpoint_path), device, trusted_legacy)
    metadata_classes = _metadata_class_count(checkpoint)
    if metadata_classes is not None and metadata_classes != num_classes:
        raise ClassOutputMismatch("Checkpoint declares {} classes; configured model has {}".format(
            metadata_classes, num_classes))
    state = _strip_prefixes(_extract_state_dict(checkpoint, checkpoint_key), strip_prefixes)
    model_state = model.state_dict()
    for key, checkpoint_value in state.items():
        model_value = model_state.get(key)
        if model_value is None or tuple(checkpoint_value.shape) == tuple(model_value.shape):
            continue
        if _is_output_key(key):
            raise ClassOutputMismatch(
                "Checkpoint output tensor {} has shape {}; expected {} for {} classes".format(
                    key, tuple(checkpoint_value.shape), tuple(model_value.shape), num_classes))
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as error:
        raise RuntimeError("Strict checkpoint load failed for {}: {}".format(
            checkpoint_path, error)) from error


def _cuda_sync(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TorchModelAdapter(ModelAdapter):
    def __init__(self, model: Any, config: ModelConfig, device: str,
                 call_mode: Optional[str] = None):
        torch = _require_torch()
        self.config = config
        self.num_classes = config.num_classes
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.model = model.to(self.device)
        if not config.checkpoint and not config.allow_uninitialized:
            raise ValueError("Torch evaluation requires a checkpoint; allow_uninitialized is smoke-only")
        if config.checkpoint:
            load_checkpoint_strict(
                self.model, config.checkpoint, self.device, config.num_classes,
                checkpoint_key=config.checkpoint_key,
                strip_prefixes=config.strip_prefixes,
                strict=config.strict_checkpoint,
                trusted_legacy=config.trusted_legacy_checkpoint,
            )
        self.model.eval()
        self.call_mode = call_mode or config.call_mode
        self.input_dtype = torch.float32
        self._memory_samples: list = []
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def efficiency_metadata(self) -> Mapping[str, Any]:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(parameter.numel() for parameter in self.model.parameters()
                        if parameter.requires_grad)
        first = next(self.model.parameters(), None)
        model_size = sum(parameter.numel() * parameter.element_size()
                         for parameter in self.model.parameters()) / 1024 ** 2
        torch = _require_torch()
        device_name = (torch.cuda.get_device_name(self.device)
                       if self.device.type == "cuda" else platform.processor() or "CPU")
        peak = (torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
                if self.device.type == "cuda" else float("nan"))
        return {"total_parameters": int(total), "trainable_parameters": int(trainable),
                "device_name": device_name, "dtype": str(first.dtype) if first is not None else "unknown",
                "model_size_mb": model_size, "peak_gpu_memory_mb": peak,
                "average_gpu_memory_mb": (float(np.mean(self._memory_samples))
                                            if self._memory_samples else float("nan"))}

    def _tensor(self, value: Any, name: str) -> Any:
        if value is None:
            raise ValueError("Batch is missing {} required by model {}".format(name, self.config.name))
        torch = _require_torch()
        return torch.as_tensor(value, dtype=self.input_dtype, device=self.device)

    def set_dtype(self, name: str) -> None:
        torch = _require_torch()
        self.input_dtype = {"float32": torch.float32, "float16": torch.float16,
                            "bfloat16": torch.bfloat16}[name]
        self.model.to(dtype=self.input_dtype)

    def _forward(self, batch: Mapping[str, Any]) -> Any:
        if self.call_mode == "late_fusion":
            return self.model(self._tensor(batch.get("ecg"), "ecg"),
                              self._tensor(batch.get("features"), "features"))
        return self.model(self._tensor(batch.get(self.config.input_key), self.config.input_key))

    @staticmethod
    def _output_tensor(output: Any) -> Any:
        if isinstance(output, (tuple, list)):
            if not output:
                raise ValueError("Model returned an empty output sequence")
            output = output[0]
        elif isinstance(output, Mapping):
            for key in ("logits", "output", "predictions"):
                if key in output:
                    output = output[key]
                    break
            else:
                raise ValueError("Model output mapping requires logits, output, or predictions")
        return output

    def predict_batch(self, batch: Mapping[str, Any]) -> BatchPrediction:
        torch = _require_torch()
        self.model.eval()
        context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with context():
            # Input conversion and host-to-device transfer are excluded from model-only timing.
            if self.call_mode == "late_fusion":
                ecg = self._tensor(batch.get("ecg"), "ecg")
                features = self._tensor(batch.get("features"), "features")
                forward = lambda: self.model(ecg, features)
            elif self.call_mode == "feature_only":
                features = self._tensor(batch.get("features"), "features")
                forward = lambda: self.model(features=features)
            else:
                model_input = self._tensor(batch.get(self.config.input_key), self.config.input_key)
                crop_count = 1
                if (self.config.input_key == "ecg" and self.config.crop_length is not None and
                        model_input.ndim == 3 and model_input.shape[-1] > self.config.crop_length):
                    length = self.config.crop_length
                    stride = self.config.crop_stride or length
                    starts = list(range(0, model_input.shape[-1] - length + 1, stride))
                    final = model_input.shape[-1] - length
                    if starts[-1] != final:
                        starts.append(final)
                    crop_count = len(starts)
                    model_input = torch.stack([model_input[:, :, start:start + length]
                                               for start in starts], dim=1)
                    batch_count = model_input.shape[0]
                    model_input = model_input.reshape(batch_count * crop_count,
                                                      model_input.shape[2], length)
                forward = lambda: self.model(model_input)
            _cuda_sync(torch, self.device)
            started = time.perf_counter()
            output = self._output_tensor(forward())
            _cuda_sync(torch, self.device)
            model_seconds = time.perf_counter() - started
            if not torch.is_tensor(output) or output.ndim != 2:
                shape = getattr(output, "shape", None)
                raise ValueError("Model output must be a rank-2 tensor, got {}".format(shape))
            if output.shape[1] != self.num_classes:
                raise ClassOutputMismatch("Model returned {} classes; expected {}".format(
                    output.shape[1], self.num_classes))
            raw_output = output
            if 'crop_count' in locals() and crop_count > 1:
                raw_output = raw_output.reshape(-1, crop_count, self.num_classes)
                raw_output = (raw_output.max(dim=1).values if self.config.crop_aggregation == "max"
                              else raw_output.mean(dim=1))
                output = raw_output
            if self.config.activation == "sigmoid":
                output = torch.sigmoid(raw_output)
            elif self.config.activation == "softmax":
                output = torch.softmax(raw_output, dim=1)
            probabilities = output.detach().cpu().numpy()
            logits = raw_output.detach().cpu().numpy() if self.config.activation != "identity" else None
            if self.device.type == "cuda":
                self._memory_samples.append(torch.cuda.memory_allocated(self.device) / 1024 ** 2)
        if not np.isfinite(probabilities).all():
            raise ValueError("Model predictions contain NaN or infinite values")
        return BatchPrediction(np.asarray(probabilities), model_seconds,
                               None if logits is None else np.asarray(logits))


class PrecomputedNPZAdapter(ModelAdapter):
    """ID-aligned prediction source for model-independent metric evaluation."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.num_classes = config.num_classes
        path = Path(str(config.precomputed_path)).expanduser()
        if not path.is_file():
            raise FileNotFoundError("Precomputed prediction NPZ not found: {}".format(path))
        with np.load(str(path), allow_pickle=False) as archive:
            if "sample_id" not in archive:
                raise ValueError("Precomputed NPZ requires sample_id")
            key = config.prediction_key
            if key is None:
                key = next((item for item in ("probabilities", "predictions", "logits")
                            if item in archive), None)
            if key is None or key not in archive:
                raise ValueError("Precomputed NPZ requires probabilities, predictions, logits, or prediction_key")
            self.sample_id = np.asarray(archive["sample_id"])
            values = np.asarray(archive[key])
            self.logits = np.asarray(archive["logits"]) if "logits" in archive else None
            self.condition = np.asarray(archive["condition"]) if "condition" in archive else None
            self.snr = np.asarray(archive["snr"]) if "snr" in archive else None
        if self.sample_id.ndim != 1 or values.shape != (len(self.sample_id), self.num_classes):
            raise ValueError("Precomputed predictions must have shape (N, {})".format(self.num_classes))
        if not np.isfinite(values).all():
            raise ValueError("Precomputed predictions contain NaN or infinite values")
        if key == "logits":
            if config.activation == "sigmoid":
                values = 1.0 / (1.0 + np.exp(-values))
            elif config.activation == "softmax":
                shifted = values - values.max(axis=1, keepdims=True)
                exponent = np.exp(shifted)
                values = exponent / exponent.sum(axis=1, keepdims=True)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Precomputed probabilities must be in [0, 1]")
        for metadata_name in ("condition", "snr"):
            metadata = getattr(self, metadata_name)
            if metadata is not None:
                if metadata.ndim == 0:
                    metadata = np.repeat(metadata.reshape(1), len(self.sample_id))
                    setattr(self, metadata_name, metadata)
                elif metadata.shape != self.sample_id.shape:
                    raise ValueError("Precomputed {} must be scalar or align with sample_id".format(
                        metadata_name))
        self.probabilities = values
        self._unique_ids = len(np.unique(self.sample_id)) == len(self.sample_id)
        if not self._unique_ids and self.condition is None and self.snr is None:
            raise ValueError("Duplicate precomputed sample IDs require condition or snr keys")

    @staticmethod
    def _equal(values: np.ndarray, target: Any) -> np.ndarray:
        if np.issubdtype(values.dtype, np.number):
            try:
                return np.isclose(values.astype(float), float(target), equal_nan=True)
            except (TypeError, ValueError):
                pass
        return values.astype(str) == str(target)

    def predict_batch(self, batch: Mapping[str, Any]) -> BatchPrediction:
        indices = []
        ids = np.asarray(batch["sample_id"])
        conditions = np.asarray(batch["condition"])
        snrs = np.asarray(batch["snr"])
        for sample_id, condition, snr in zip(ids, conditions, snrs):
            matches = self._equal(self.sample_id, sample_id)
            if not self._unique_ids and self.condition is not None:
                matches &= self._equal(self.condition, condition)
            if not self._unique_ids and self.snr is not None:
                matches &= self._equal(self.snr, snr)
            found = np.flatnonzero(matches)
            if len(found) != 1:
                raise ValueError("Expected one precomputed prediction for sample_id={!r}, found {}".format(
                    sample_id, len(found)))
            indices.append(int(found[0]))
        logits = None if self.logits is None else np.asarray(self.logits[indices])
        return BatchPrediction(np.asarray(self.probabilities[indices]), 0.0, logits)


class KerasModelAdapter(ModelAdapter):
    """Inference-only adapter for Keras ECG, feature-only, or fusion models."""

    def __init__(self, config: ModelConfig):
        if not config.checkpoint:
            raise ValueError("Keras evaluation requires a .keras/.h5 checkpoint")
        try:
            import tensorflow as tf
        except ImportError as error:
            raise RuntimeError("Keras adapter requires TensorFlow") from error
        self.tf = tf
        self.config = config
        self.num_classes = config.num_classes
        self.model = tf.keras.models.load_model(config.checkpoint, compile=False)

    def predict_batch(self, batch: Mapping[str, Any]) -> BatchPrediction:
        if self.config.call_mode == "late_fusion":
            inputs = [np.asarray(batch["ecg"], dtype=np.float32),
                      np.asarray(batch["features"], dtype=np.float32)]
        else:
            inputs = np.asarray(batch[self.config.input_key], dtype=np.float32)
        started = time.perf_counter()
        output = np.asarray(self.model(inputs, training=False))
        elapsed = time.perf_counter() - started
        if output.ndim != 2 or output.shape[1] != self.num_classes:
            raise ClassOutputMismatch("Keras model returned shape {}; expected [B, {}]".format(
                output.shape, self.num_classes))
        logits = output.copy() if self.config.activation != "identity" else None
        if self.config.activation == "sigmoid":
            output = 1.0 / (1.0 + np.exp(-output))
        elif self.config.activation == "softmax":
            shifted = output - output.max(axis=1, keepdims=True)
            output = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        if np.any((output < 0) | (output > 1)) or not np.isfinite(output).all():
            raise ValueError("Keras model probabilities must be finite values in [0, 1]")
        return BatchPrediction(output, elapsed, logits)

    def efficiency_metadata(self) -> Mapping[str, Any]:
        total = int(self.model.count_params())
        trainable = int(sum(np.prod(value.shape) for value in self.model.trainable_weights))
        return {"total_parameters": total, "trainable_parameters": trainable,
                "device_name": "TensorFlow runtime", "dtype": "float32"}


AdapterBuilder = Callable[[ModelConfig, str], ModelAdapter]
MODEL_REGISTRY: MutableMapping[str, AdapterBuilder] = {}


def register_model_adapter(name: str, builder: AdapterBuilder) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Adapter name cannot be empty")
    if normalized in MODEL_REGISTRY:
        raise ValueError("Model adapter {!r} is already registered".format(normalized))
    MODEL_REGISTRY[normalized] = builder


def available_adapters() -> Tuple[str, ...]:
    return tuple(sorted(MODEL_REGISTRY))


def _build_original(config: ModelConfig, device: str) -> ModelAdapter:
    factory = _import_symbol(config.factory or
                             "models.original_model_factory:build_original_model")
    kwargs = dict(config.kwargs)
    kwargs.setdefault("num_classes", config.num_classes)
    kwargs.setdefault("input_channels", config.input_channels)
    architecture = config.architecture or config.name
    return TorchModelAdapter(factory(architecture, **kwargs), config, device, call_mode="single")


_CBAM_VARIANTS: Dict[str, Dict[str, Any]] = {
    "baseline": {"use_cbam": False, "use_se": False, "input_mode": "ecg_only"},
    "cbam": {"use_cbam": True, "use_se": False, "input_mode": "ecg_only"},
    "ecg_only": {"use_cbam": True, "use_se": False, "input_mode": "ecg_only"},
    "se": {"use_cbam": False, "use_se": True, "input_mode": "ecg_only"},
    "feature_only": {"use_cbam": True, "use_se": False, "input_mode": "feature_only"},
    "late_fusion": {"use_cbam": True, "use_se": False, "input_mode": "late_fusion",
                     "fusion_type": "concat"},
    "xresnet_emd_concat": {"use_cbam": False, "use_se": False,
                            "input_mode": "late_fusion", "fusion_type": "concat"},
    "cbam_emd_concat": {"use_cbam": True, "use_se": False, "input_mode": "late_fusion",
                        "fusion_type": "concat"},
    "cbam_emd_gated": {"use_cbam": True, "use_se": False, "input_mode": "late_fusion",
                         "fusion_type": "gated"},
    "cbam_emd_bottleneck_gated_emb32": {"use_cbam": True, "use_se": False,
                                          "model_variant": "emd_bottleneck_gated",
                                          "emd_embedding_dim": 32},
    "cbam_emd_bottleneck_gated_emb64": {"use_cbam": True, "use_se": False,
                                          "model_variant": "emd_bottleneck_gated",
                                          "emd_embedding_dim": 64},
    "emd_only_bottleneck_emb32": {"use_cbam": False, "use_se": False,
                                    "model_variant": "emd_only_bottleneck",
                                    "emd_embedding_dim": 32},
    "emd_only_bottleneck_emb64": {"use_cbam": False, "use_se": False,
                                    "model_variant": "emd_only_bottleneck",
                                    "emd_embedding_dim": 64},
    "se_emd_concat": {"use_cbam": False, "use_se": True,
                       "input_mode": "late_fusion", "fusion_type": "concat"},
}


def _build_cbam(config: ModelConfig, device: str) -> ModelAdapter:
    variant = (config.variant or "cbam").lower()
    if variant not in _CBAM_VARIANTS:
        raise ValueError("Unknown CBAM variant {!r}; expected {}".format(
            variant, ", ".join(sorted(_CBAM_VARIANTS))))
    kwargs = dict(_CBAM_VARIANTS[variant])
    kwargs.update(config.kwargs)
    kwargs.setdefault("num_classes", config.num_classes)
    kwargs.setdefault("input_channels", config.input_channels)
    if "model_variant" in kwargs:
        factory = _import_symbol(config.factory or "models.cbam_xresnet1d:build_model")
        model = factory("xresnet1d101", **kwargs)
        call_mode = "feature_only" if kwargs["model_variant"] == "emd_only_bottleneck" else "late_fusion"
    else:
        factory = _import_symbol(config.factory or "models.cbam_xresnet1d:cbam_xresnet1d101")
        model = factory(**kwargs)
        input_mode = kwargs.get("input_mode")
        call_mode = input_mode if input_mode in ("late_fusion", "feature_only") else "single"
    return TorchModelAdapter(model, config, device, call_mode=call_mode)


def _build_factory(config: ModelConfig, device: str) -> ModelAdapter:
    factory = _import_symbol(str(config.factory))
    kwargs = dict(config.kwargs)
    model = factory(**kwargs)
    return TorchModelAdapter(model, config, device)


def _build_precomputed(config: ModelConfig, device: str) -> ModelAdapter:
    del device
    return PrecomputedNPZAdapter(config)


def _build_lightning_checkpoint(config: ModelConfig, device: str) -> ModelAdapter:
    if not config.checkpoint:
        raise ValueError("lightning_checkpoint adapter requires model.checkpoint")
    if not config.trusted_legacy_checkpoint:
        raise ValueError("Lightning legacy checkpoint loading requires trusted_legacy_checkpoint=true")
    loader = _import_symbol(config.factory or
                            "models.lightning_checkpoint_models:load_checkpoint_model")
    architecture = config.architecture or config.name
    model = loader(config.checkpoint, architecture, config.num_classes, device)
    # The repository loader already validates every key and tensor shape.
    return TorchModelAdapter(model, replace(config, checkpoint=None, allow_uninitialized=True),
                             device, call_mode="single")


def _build_keras(config: ModelConfig, device: str) -> ModelAdapter:
    del device
    return KerasModelAdapter(config)


register_model_adapter("original", _build_original)
register_model_adapter("original_model_factory", _build_original)
register_model_adapter("cbam_xresnet1d", _build_cbam)
register_model_adapter("factory", _build_factory)
register_model_adapter("generic", _build_factory)
register_model_adapter("precomputed_npz", _build_precomputed)
register_model_adapter("precomputed", _build_precomputed)
register_model_adapter("lightning_checkpoint", _build_lightning_checkpoint)
register_model_adapter("keras", _build_keras)
register_model_adapter("wavelet_keras", _build_keras)


def build_model_adapter(config: ModelConfig, device: str = "cpu",
                        dtype: str = "float32") -> ModelAdapter:
    adapter_name = config.adapter.strip().lower()
    builder = MODEL_REGISTRY.get(adapter_name)
    if builder is None:
        raise ValueError("Unknown model adapter {!r}; available: {}".format(
            config.adapter, ", ".join(available_adapters())))
    adapter = builder(config, device)
    if isinstance(adapter, TorchModelAdapter):
        adapter.set_dtype(dtype)
    return adapter
