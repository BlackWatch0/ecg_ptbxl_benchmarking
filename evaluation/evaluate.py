"""Command-line entry point for prediction collection and analysis callbacks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

try:
    from .config import EvaluationConfig, load_config
    from .evaluator import EvaluationRun, Evaluator
    from .pipeline import run_standard_evaluation
except ImportError:  # Support ``python evaluation/evaluate.py``.
    import sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from evaluation.config import EvaluationConfig, load_config
    from evaluation.evaluator import EvaluationRun, Evaluator
    from evaluation.pipeline import run_standard_evaluation


RunCallback = Callable[[EvaluationConfig], Any]


def _boolean_override(parser: argparse.ArgumentParser, name: str, destination: str,
                      help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--" + name, dest=destination, action="store_true", help=help_text)
    group.add_argument("--no-" + name, dest=destination, action="store_false",
                       help="Disable " + help_text.lower())
    parser.set_defaults(**{destination: None})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ECG models against standardized NPZ scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="YAML evaluation configuration")
    parser.add_argument("--output-dir")
    parser.add_argument("--dataset-split", choices=("train", "validation", "test"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                        help="Set any dotted config key after loading YAML")

    model = parser.add_argument_group("model")
    model.add_argument("--model-name")
    model.add_argument("--adapter", choices=("original", "original_model_factory",
                                               "cbam_xresnet1d", "factory", "generic",
                                               "precomputed_npz", "precomputed",
                                               "lightning_checkpoint", "keras",
                                               "wavelet_keras"))
    model.add_argument("--architecture")
    model.add_argument("--factory", help="Generic module:function model factory")
    model.add_argument("--checkpoint")
    model.add_argument("--checkpoint-key", help="Explicit dotted key containing model weights")
    model.add_argument("--strip-prefix", action="append", default=None,
                       help="Explicit state-dict prefix to strip; repeat in stripping order")
    model.add_argument("--model-kwargs", help="JSON object passed to the selected factory")
    model.add_argument("--num-classes", type=int)
    model.add_argument("--input-channels", type=int)
    model.add_argument("--activation", choices=("sigmoid", "softmax", "identity"))
    model.add_argument("--call-mode", choices=("single", "feature_only", "late_fusion"))
    model.add_argument("--input-key", choices=("ecg", "features"))
    model.add_argument("--variant", help="CBAM variant name")
    model.add_argument("--precomputed", dest="precomputed_path",
                       help="ID-aligned prediction NPZ (also selects precomputed_npz adapter)")
    model.add_argument("--prediction-key")
    _boolean_override(model, "strict-checkpoint", "strict_checkpoint",
                      "Require exact checkpoint keys and shapes")

    data = parser.add_argument_group("data")
    data.add_argument("--scenario", action="append", default=None, metavar="NAME=PATH",
                      help="Replace YAML scenarios; repeat for each scenario")
    data.add_argument("--batch-size", type=int)
    data.add_argument("--ecg-layout", choices=("NCT", "NTC"))
    _boolean_override(data, "require-ecg", "require_ecg", "Require ECG arrays")
    _boolean_override(data, "require-features", "require_features", "Require feature arrays")
    _boolean_override(data, "validate-alignment", "validate_alignment",
                      "Require identical IDs, order, and labels across scenarios")

    inference = parser.add_argument_group("inference")
    inference.add_argument("--device")
    inference.add_argument("--warmup-batches", type=int)
    inference.add_argument("--timing", choices=("model", "end_to_end", "both"))

    analysis = parser.add_argument_group("analysis")
    analysis.add_argument("--class-name", action="append", default=None,
                          help="Class name in output order; repeat per class")
    analysis.add_argument("--threshold", action="append", type=float, default=None,
                          help="Decision threshold; provide once or once per class")
    _boolean_override(analysis, "metrics", "metrics", "Compute classification metrics")
    _boolean_override(analysis, "calibration", "calibration", "Compute calibration analyses")
    _boolean_override(analysis, "robustness", "robustness", "Compute robustness analyses")
    analysis.add_argument("--threshold-mode", choices=("fixed", "fixed_global",
                                                        "fixed_per_class", "load_from_file"))
    analysis.add_argument("--threshold-file")
    analysis.add_argument("--bootstrap", type=int)
    analysis.add_argument("--snr-list", nargs="+", type=float)
    _boolean_override(analysis, "evaluate-clean", "evaluate_clean", "Evaluate clean scenarios")
    _boolean_override(analysis, "evaluate-noisy", "evaluate_noisy", "Evaluate noisy scenarios")
    _boolean_override(analysis, "evaluate-denoised", "evaluate_denoised", "Evaluate denoised scenarios")
    output = parser.add_argument_group("output")
    _boolean_override(output, "save-predictions", "save_predictions", "Save sample predictions")
    _boolean_override(output, "save-logits", "save_logits", "Save raw logits when available")
    _boolean_override(output, "save-plots", "save_plots", "Generate plots")
    return parser


def _put(target: Dict[str, Any], section: str, name: str, value: Any) -> None:
    if value is not None:
        target.setdefault(section, {})[name] = value


def _parse_scenarios(values: Sequence[str]) -> List[Dict[str, str]]:
    scenarios = []
    for value in values:
        if "=" not in value:
            raise ValueError("--scenario must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ValueError("--scenario must use non-empty NAME=PATH")
        scenarios.append({"name": name, "path": str(Path(raw_path).expanduser().resolve()),
                          "condition": name})
    return scenarios


def _parse_json_object(value: str, flag: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("{} must be valid JSON: {}".format(flag, error)) from error
    if not isinstance(parsed, Mapping):
        raise ValueError("{} must be a JSON object".format(flag))
    return parsed


def _set_dotted(target: Dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError("--set must use dotted.key=JSON")
    dotted, raw_value = expression.split("=", 1)
    parts = dotted.split(".")
    if not all(parts):
        raise ValueError("--set key cannot be empty")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    current = target
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError("--set path conflicts at {}".format(part))
        current = child
    current[parts[-1]] = value


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    _put(overrides, "run", "output_dir", (str(Path(args.output_dir).expanduser().resolve())
                                            if args.output_dir else None))
    for name in ("dataset_split", "seed", "overwrite"):
        _put(overrides, "run", name, getattr(args, name))
    for name in ("name", "adapter", "architecture", "factory", "checkpoint_key",
                 "num_classes", "activation", "call_mode", "input_key", "variant",
                 "prediction_key", "strict_checkpoint"):
        argument_name = "model_name" if name == "name" else name
        _put(overrides, "model", name, getattr(args, argument_name))
    if args.checkpoint is not None:
        _put(overrides, "model", "checkpoint",
             str(Path(args.checkpoint).expanduser().resolve()))
    if args.strip_prefix is not None:
        _put(overrides, "model", "strip_prefixes", args.strip_prefix)
    if args.model_kwargs is not None:
        _put(overrides, "model", "kwargs", _parse_json_object(args.model_kwargs,
                                                                "--model-kwargs"))
    if args.precomputed_path is not None:
        _put(overrides, "model", "precomputed_path",
             str(Path(args.precomputed_path).expanduser().resolve()))
        _put(overrides, "model", "adapter", "precomputed_npz")
    if args.input_channels is not None:
        _put(overrides, "model", "input_channels", args.input_channels)
        _put(overrides, "data", "input_channels", args.input_channels)

    if args.scenario is not None:
        _put(overrides, "data", "scenarios", _parse_scenarios(args.scenario))
    for name in ("batch_size", "num_workers", "ecg_layout", "require_ecg", "require_features",
                  "validate_alignment"):
        _put(overrides, "data", name, getattr(args, name))
    for name in ("device", "warmup_batches", "timing"):
        _put(overrides, "inference", name, getattr(args, name))
    _put(overrides, "analysis", "class_names", args.class_name)
    _put(overrides, "analysis", "thresholds", args.threshold)
    for name in ("metrics", "calibration", "robustness"):
        _put(overrides, "analysis", name, getattr(args, name))
    _put(overrides, "analysis", "bootstrap", args.bootstrap)
    if args.threshold_mode is not None:
        _put(overrides, "analysis", "threshold", {
            "mode": "fixed_global" if args.threshold_mode == "fixed" else args.threshold_mode,
            **({"file": str(Path(args.threshold_file).expanduser().resolve())}
               if args.threshold_file else {})})
    elif args.threshold_file is not None:
        _put(overrides, "analysis", "threshold", {
            "mode": "load_from_file",
            "file": str(Path(args.threshold_file).expanduser().resolve())})
    for name in ("save_predictions", "save_logits", "save_plots"):
        _put(overrides, "output", name, getattr(args, name))
    for expression in args.set:
        _set_dotted(overrides, expression)
    return overrides


def run_evaluation(config: EvaluationConfig) -> EvaluationRun:
    evaluator = Evaluator(config)
    analyze = config.analysis.metrics or config.analysis.calibration or config.analysis.robustness
    return evaluator.run(analyze=analyze)


def _filter_scenarios(config: EvaluationConfig, args: argparse.Namespace) -> EvaluationConfig:
    flags = {"clean": args.evaluate_clean, "noisy": args.evaluate_noisy,
             "denoised": args.evaluate_denoised}
    scenarios = config.data.scenarios
    if any(value is True for value in flags.values()):
        requested = {name for name, enabled in flags.items() if enabled is True}
        scenarios = tuple(item for item in scenarios if str(item.condition).lower() in requested)
    elif any(value is False for value in flags.values()):
        scenarios = tuple(item for item in scenarios
                          if flags.get(str(item.condition).lower()) is not False)
    if args.snr_list is not None:
        selected = np.asarray(args.snr_list, dtype=float)
        scenarios = tuple(item for item in scenarios if item.snr is None or
                          np.isclose(float(item.snr), selected).any())
    if not scenarios:
        raise ValueError("CLI condition/SNR filters removed every configured scenario")
    return replace(config, data=replace(config.data, scenarios=scenarios))


def main(argv: Optional[Sequence[str]] = None,
         run_callback: Optional[RunCallback] = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _filter_scenarios(load_config(args.config, cli_overrides(args)), args)
    except (TypeError, ValueError, OSError) as error:
        parser.error(str(error))
    callback = run_callback or run_standard_evaluation
    return callback(config)


if __name__ == "__main__":
    main()
