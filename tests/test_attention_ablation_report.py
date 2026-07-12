import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import build_attention_ablation_report as report


def build_and_check(tmp_path):
    four_root, se_root, output = tmp_path / "four", tmp_path / "se", tmp_path / "report"
    rows_by_model = {}
    classes_by_model = {}
    for model_index, model in enumerate(report.MODELS):
        rows, class_rows = [], []
        for scenario_index, scenario in enumerate(report.SCENARIOS):
            value = 0.90 + model_index / 100 - scenario_index / 100
            common = {"experiment_name": model, "seed": 42, "ecg_scenario": scenario,
                      "threshold_strategy": report.STRATEGY, "parameter_count": 1_000_000 + model_index,
                      "trainable_parameter_count": 1_000_000 + model_index,
                      "inference_time_per_sample_ms": 1.0 + model_index / 10,
                      "training_time_seconds": 100 + model_index, "best_epoch": 2,
                      "best_valid_loss": 0.1, "actual_batch_size": 128}
            rows.append({**common, **{metric: value for metric in report.METRICS}})
            for class_name in report.CLASSES:
                class_rows.append({**common, "class_name": class_name, "f1": value,
                                   "roc_auc": value, "pr_auc": value})
        rows_by_model[model] = rows
        classes_by_model[model] = class_rows
    for model in report.MODELS:
        root = four_root if model in report.MODELS[:4] else se_root
        directory = root / "metrics" / model
        directory.mkdir(parents=True)
        pd.DataFrame(rows_by_model[model]).to_csv(directory / "seed_42.csv", index=False)
        pd.DataFrame(classes_by_model[model]).to_csv(directory / "seed_42_per_class.csv", index=False)
    for model in report.MODELS[-2:]:
        directory = se_root / "training_logs" / model
        directory.mkdir(parents=True)
        pd.DataFrame({"epoch": [1, 2], "train_loss": [.5, .4], "valid_loss": [.6, .45]}).to_csv(
            directory / "seed_42.csv", index=False)

    with patch.object(sys, "argv", ["report", "--four-model-root", str(four_root),
                                    "--se-root", str(se_root), "--output-dir", str(output)]):
        report.main()

    expected = ["attention_ablation_summary.csv", "attention_clean_comparison.csv",
                "attention_snr_comparison.csv", "attention_contributions.csv", "robustness_metrics.csv",
                "mean_noisy_metrics.csv", "per_class_metrics.csv", "model_complexity.csv",
                "best_model_summary.json", "SE_XRESNET_ABLATION_RESULTS.md"]
    assert all((output / name).exists() for name in expected)
    assert len(list((output / "figures").glob("*.png"))) == 12
    assert len(list((output / "figures").glob("*.pdf"))) == 12
    complexity = pd.read_csv(output / "model_complexity.csv")
    assert {"training_time_seconds", "best_epoch", "best_valid_loss", "actual_batch_size"}.issubset(complexity)
    markdown = (output / "SE_XRESNET_ABLATION_RESULTS.md").read_text()
    assert markdown.count("\n## ") == 12
    assert set(report.DISPLAY.values()) == {"xResNet", "CBAM-xResNet", "SE-xResNet",
                                            "xResNet + EMD", "CBAM-xResNet + EMD", "SE-xResNet + EMD"}
    contributions = pd.read_csv(output / "attention_contributions.csv")
    se_clean = contributions[(contributions.comparison == "SE vs baseline") &
                             (contributions.scenario == "clean")].iloc[0]
    assert abs(se_clean.delta_macro_roc_auc - 0.04) < 1e-12


class AttentionAblationReportTest(unittest.TestCase):
    def test_build_report_from_metric_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            build_and_check(Path(directory))

    def test_output_directory_at_result_final_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "se" / "metrics").mkdir(parents=True)
            report_dir, figures_dir = report.output_directories(root / "se" / "final_report")
            self.assertEqual(report_dir, root / "se" / "final_report")
            self.assertEqual(figures_dir, root / "se" / "figures")


if __name__ == "__main__":
    unittest.main()
