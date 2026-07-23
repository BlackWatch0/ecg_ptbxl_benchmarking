"""Validate the modern AWS training environment and accelerator visibility."""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path


PACKAGE_CHECKS = (
    ("numpy", "numpy", (1, 26), (2, 0)),
    ("pandas", "pandas", (2, 2), (2, 3)),
    ("scipy", "scipy", (1, 13), (1, 14)),
    ("sklearn", "scikit-learn", (1, 5), (1, 6)),
    ("matplotlib", "matplotlib", (3, 9), (3, 10)),
    ("yaml", "PyYAML", (6, 0), (7, 0)),
    ("pywt", "PyWavelets", (1, 7), (2, 0)),
    ("wfdb", "wfdb", (4, 1), (4, 2)),
    ("tqdm", "tqdm", (4, 66), (5, 0)),
    ("torch", "torch", (2, 5), (2, 6)),
)


def version_tuple(value):
    numbers = re.findall(r"\d+", str(value).split("+", 1)[0])
    return tuple(int(number) for number in numbers[:3])


def version_in_range(value, lower, upper):
    parsed = version_tuple(value)
    width = max(len(parsed), len(lower), len(upper))
    parsed += (0,) * (width - len(parsed))
    low = tuple(lower) + (0,) * (width - len(lower))
    high = tuple(upper) + (0,) * (width - len(upper))
    return low <= parsed < high


def installed_version(distribution, module):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def validate_environment(require_cuda=False, skip_wavelet=False, check_compute=False):
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
        "pytorch": {},
        "tensorflow": {"status": "skipped"} if skip_wavelet else {},
        "issues": [],
    }
    if not ((3, 10) <= sys.version_info[:2] < (3, 11)):
        report["issues"].append("Python must be >=3.10,<3.11")

    modules = {}
    for module_name, distribution, lower, upper in PACKAGE_CHECKS:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            report["issues"].append("{} import failed: {}".format(distribution, error))
            continue
        modules[module_name] = module
        version = installed_version(distribution, module)
        report["packages"][distribution] = version
        if not version_in_range(version, lower, upper):
            report["issues"].append(
                "{} {} is outside [{}, {})".format(distribution, version, lower, upper))

    torch = modules.get("torch")
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        report["pytorch"] = {
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
        if require_cuda and not cuda_available:
            report["issues"].append("PyTorch CUDA is required but unavailable")
        if require_cuda and torch.version.cuda != "12.1":
            report["issues"].append(
                "Expected PyTorch CUDA runtime 12.1, got {}".format(torch.version.cuda))
        if cuda_available:
            report["pytorch"].update({
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "device_count": torch.cuda.device_count(),
            })
        if check_compute and cuda_available:
            value = torch.randn(128, 128, device="cuda", requires_grad=True)
            loss = (value @ value).square().mean()
            loss.backward()
            report["pytorch"]["compute_smoke"] = bool(torch.isfinite(loss).item())

    if not skip_wavelet:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            tensorflow = importlib.import_module("tensorflow")
            try:
                version = importlib.metadata.version("tensorflow-cpu")
            except importlib.metadata.PackageNotFoundError:
                version = str(getattr(tensorflow, "__version__", "unknown"))
                report["issues"].append(
                    "tensorflow-cpu distribution is missing; do not install tensorflow GPU")
            visible_gpus = tensorflow.config.list_physical_devices("GPU")
            report["tensorflow"] = {
                "version": version,
                "visible_gpus": [str(device) for device in visible_gpus],
            }
            if not version_in_range(version, (2, 15), (2, 16)):
                report["issues"].append(
                    "tensorflow-cpu {} is outside [2.15, 2.16)".format(version))
            if visible_gpus:
                report["issues"].append(
                    "TensorFlow must remain CPU-only in the shared training environment")
            if check_compute:
                import numpy as np
                model = tensorflow.keras.Sequential([
                    tensorflow.keras.layers.Input((864,)),
                    tensorflow.keras.layers.Dense(128, activation="relu"),
                    tensorflow.keras.layers.Dense(5, activation="sigmoid"),
                ])
                model.compile(optimizer="adamax", loss="binary_crossentropy")
                loss = model.train_on_batch(
                    np.zeros((2, 864), dtype=np.float32),
                    np.zeros((2, 5), dtype=np.float32),
                )
                report["tensorflow"]["compute_smoke"] = bool(np.isfinite(loss))
        except Exception as error:
            report["issues"].append("tensorflow-cpu import/smoke failed: {}".format(error))

    report["status"] = "ok" if not report["issues"] else "failed"
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true",
                        help="Fail unless PyTorch sees a CUDA 12.1 device")
    parser.add_argument("--skip-wavelet", action="store_true",
                        help="Skip tensorflow-cpu validation")
    parser.add_argument("--check-compute", action="store_true",
                        help="Run small PyTorch and Keras backward/train steps")
    parser.add_argument("--output", help="Optional JSON report path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = validate_environment(
        require_cuda=args.require_cuda,
        skip_wavelet=args.skip_wavelet,
        check_compute=args.check_compute,
    )
    content = json.dumps(report, indent=2, sort_keys=True)
    print(content)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content + "\n", encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
