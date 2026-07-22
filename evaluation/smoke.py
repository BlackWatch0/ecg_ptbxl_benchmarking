"""Run a synthetic-input smoke test against an existing repository checkpoint."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from .config import config_from_mapping
    from .pipeline import run_standard_evaluation
except ImportError:
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from evaluation.config import config_from_mapping
    from evaluation.pipeline import run_standard_evaluation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architecture", default="xresnet",
                        choices=("lenet", "lstm", "resnet", "inception", "xresnet"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--length", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rng = np.random.RandomState(42)
    ecg = rng.normal(size=(args.samples, 12, args.length)).astype(np.float32)
    labels = np.zeros((args.samples, args.num_classes), dtype=np.float32)
    labels[np.arange(args.samples), np.arange(args.samples) % args.num_classes] = 1
    class_names = ["class_{}".format(index) for index in range(args.num_classes)]
    with tempfile.TemporaryDirectory(prefix="ecg-evaluation-smoke-") as temporary:
        data_path = Path(temporary) / "synthetic_clean.npz"
        np.savez(data_path, ecg=ecg, labels=labels,
                 sample_id=np.arange(args.samples), condition=np.asarray("clean"))
        config = config_from_mapping({
            "run": {"experiment_name": "current_checkpoint_smoke",
                    "output_dir": str(Path(args.output_dir).resolve()), "seed": 42,
                    "dataset_split": "test", "overwrite": args.overwrite,
                    "dataset_name": "synthetic-smoke", "dataset_version": "1"},
            "model": {"name": args.architecture, "adapter": "lightning_checkpoint",
                      "architecture": args.architecture,
                      "checkpoint": str(Path(args.checkpoint).resolve()),
                      "num_classes": args.num_classes, "input_channels": 12,
                      "activation": "sigmoid", "trusted_legacy_checkpoint": True},
            "data": {"scenarios": [{"name": "clean", "path": str(data_path),
                                      "condition": "clean"}],
                     "batch_size": args.samples, "input_channels": 12,
                     "require_ecg": True, "validate_alignment": True},
            "inference": {"device": "cpu", "warmup_batches": 1, "timing": "both"},
            "analysis": {"class_names": class_names, "bootstrap": 0,
                         "calibration": True, "robustness": False},
            "output": {"save_predictions": True, "save_logits": True,
                       "save_plots": True},
        })
        output = run_standard_evaluation(config)
    print("Synthetic checkpoint smoke completed: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
