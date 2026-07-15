import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import build_original_models_benchmark_report as report


def synthetic_runner_output(root, include_wavelet=True):
    models = report.MODELS if include_wavelet else report.MODELS[:-1]
    metrics, classes, histories = [], [], []
    scenarios = [("clean", None)] + [(domain, snr) for domain in ("noisy", "denoised") for snr in report.SNRS]
    for model_index, model in enumerate(models):
        for seed in (42, 123):
            clean = 0.80 + model_index * 0.01 + (seed == 123) * 0.002
            for domain, snr in scenarios:
                penalty = 0 if domain == "clean" else (24 - snr) * 0.001
                improvement = 0.015 if domain == "denoised" else 0
                value = clean - penalty + improvement
                suffix = "" if domain == "clean" else "_snrm{}".format(abs(snr)) if snr < 0 else "_snr{}".format(snr)
                common = {"experiment_name": model, "model_name": model, "seed": seed,
                          "ecg_scenario": domain + suffix, "target_snr_db": snr}
                for strategy in ("threshold_0.5", "best_global_threshold", "per_class_thresholds"):
                    metadata = {**common, "threshold_strategy": strategy}
                    metrics.append({**metadata, "macro_roc_auc": value, "macro_f1": value - 0.1,
                                    "macro_pr_auc": value - 0.05, "parameter_count": 1000 + model_index,
                                    "trainable_parameter_count": 900 + model_index,
                                    "inference_time_per_sample_ms": 1 + model_index / 10})
                    for class_index, class_name in enumerate(report.CLASSES):
                        classes.append({**metadata, "class_name": class_name,
                                        "roc_auc": value - class_index / 100,
                                        "pr_auc": value - 0.05, "f1": value - 0.1})
    (root / "metrics").mkdir(parents=True)
    (root / "training_logs").mkdir()
    pd.DataFrame(metrics).to_csv(root / "metrics" / "runner_metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(root / "metrics" / "runner_per_class.csv", index=False)
    for model_index, model in enumerate(models):
        model_root = root / "training_logs" / model
        model_root.mkdir()
        for seed in (42, 123):
            history = [{"epoch": epoch, "train_loss": 1 / (epoch + model_index + 1),
                        "valid_loss": 1.1 / (epoch + model_index + 1)} for epoch in (1, 2, 3)]
            pd.DataFrame(history).to_csv(model_root / "seed_{}.csv".format(seed), index=False)


class OriginalModelsBenchmarkReportTest(unittest.TestCase):
    def test_synthetic_end_to_end_report_uses_measured_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runner"
            synthetic_runner_output(root)
            output = report.build_report(root, None, [42, 123])
            expected = {
                "benchmark_summary.csv", "clean_comparison.csv", "noisy_snr_comparison.csv",
                "denoised_snr_comparison.csv", "denoising_contributions.csv", "robustness_metrics.csv",
                "mean_domain_metrics.csv", "per_class_metrics.csv", "model_complexity.csv",
                "best_model_summary.json", "ORIGINAL_MODELS_BENCHMARK_RESULTS.md",
            }
            self.assertEqual(expected, {path.name for path in output.iterdir() if path.is_file()})
            contributions = pd.read_csv(output / "denoising_contributions.csv")
            self.assertTrue((contributions.macro_roc_auc_improvement.round(12) == 0.015).all())
            summary = pd.read_csv(output / "benchmark_summary.csv")
            source = pd.read_csv(root / "metrics" / "runner_metrics.csv")
            self.assertEqual(len(summary) * 3, len(source))
            self.assertAlmostEqual(summary.macro_roc_auc.max(), source.macro_roc_auc.max())
            best = json.loads((output / "best_model_summary.json").read_text())
            self.assertEqual(best["best_clean"]["model_name"], "Wavelet+NN")
            self.assertEqual(best["wavelet_status"], "included")
            figure_names = {path.stem for path in (output / "figures").glob("*.png")}
            self.assertEqual(12 + len(report.MODELS), len(figure_names))
            self.assertEqual(figure_names, {path.stem for path in (output / "figures").glob("*.pdf")})
            markdown = (output / "ORIGINAL_MODELS_BENCHMARK_RESULTS.md").read_text()
            self.assertIn("No metric is manually entered or estimated", markdown)
            self.assertIn("Wavelet + neural network", markdown)

    def test_explicit_wavelet_exclusion_has_no_fabricated_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runner"
            synthetic_runner_output(root, include_wavelet=False)
            (root / "config").mkdir()
            (root / "config" / "wavelet_nn_status.json").write_text(json.dumps({
                "model": "Wavelet+NN", "status": "unsupported_separate_pipeline",
                "reason": "dependency unavailable in runner environment",
            }))
            output = report.build_report(root, None, [42, 123])
            summary = pd.read_csv(output / "benchmark_summary.csv")
            self.assertNotIn(report.WAVELET_MODEL, set(summary.model_name))
            best = json.loads((output / "best_model_summary.json").read_text())
            self.assertEqual(best["wavelet_status"],
                             "unsupported_separate_pipeline: dependency unavailable in runner environment")
            self.assertIn("No Wavelet values were generated",
                          (output / "ORIGINAL_MODELS_BENCHMARK_RESULTS.md").read_text())

    def test_rejects_missing_domain_snr_and_seed_combinations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runner"
            synthetic_runner_output(root)
            path = root / "metrics" / "runner_metrics.csv"
            frame = pd.read_csv(path).iloc[:-3]
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one row"):
                report.build_report(root, None, [42, 123])

    def test_wavelet_exclusion_must_not_hide_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runner"
            synthetic_runner_output(root)
            with self.assertRaisesRegex(ValueError, "cannot also have an excluded status"):
                report.build_report(root, None, [42, 123], "excluded")


if __name__ == "__main__":
    unittest.main()
