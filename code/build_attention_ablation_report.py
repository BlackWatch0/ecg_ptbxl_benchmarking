#!/usr/bin/env python3
"""Merge the immutable four-model and SE ablation result trees into one report."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = [
    "xresnet1d101_baseline",
    "cbam_xresnet1d101",
    "xresnet1d101_emd_late_fusion",
    "cbam_xresnet1d101_emd_late_fusion",
    "se_xresnet1d101",
    "se_xresnet1d101_emd_late_fusion",
]
SCENARIOS = ["clean", "snr24", "snr12", "snr6", "snr0", "snrm6"]
NOISY = SCENARIOS[1:]
STRATEGY = "per_class_thresholds"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
METRICS = ["macro_roc_auc", "macro_pr_auc", "macro_f1", "micro_f1", "exact_match_accuracy"]
DISPLAY = {
    "xresnet1d101_baseline": "xResNet",
    "cbam_xresnet1d101": "CBAM-xResNet",
    "se_xresnet1d101": "SE-xResNet",
    "xresnet1d101_emd_late_fusion": "xResNet + EMD",
    "cbam_xresnet1d101_emd_late_fusion": "CBAM-xResNet + EMD",
    "se_xresnet1d101_emd_late_fusion": "SE-xResNet + EMD",
}
METRIC_LABEL = {
    "macro_roc_auc": "Macro ROC-AUC",
    "macro_pr_auc": "Macro PR-AUC",
    "macro_f1": "Macro F1",
    "micro_f1": "Micro F1",
    "exact_match_accuracy": "Exact-match accuracy",
}
ALIASES = {
    "model": "experiment_name", "experiment": "experiment_name", "model_name": "experiment_name",
    "scenario": "ecg_scenario", "snr_scenario": "ecg_scenario", "thresholding_strategy": "threshold_strategy",
    "macro_auc": "macro_roc_auc", "macro_auroc": "macro_roc_auc", "macro_auprc": "macro_pr_auc",
    "class": "class_name", "parameters": "parameter_count", "num_parameters": "parameter_count",
    "trainable_parameters": "trainable_parameter_count",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-model-root", type=Path, help="Existing baseline/CBAM result root")
    parser.add_argument("--se-root", type=Path, help="New two-SE-model result root")
    parser.add_argument("--training-root", type=Path,
                        help="Root containing SE training_logs (defaults to --se-root)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/attention_ablation_report"))
    return parser.parse_args()


def output_directories(output):
    output = output.expanduser().resolve()
    if output.name == "final_report":
        return output, output.parent / "figures"
    if (output / "metrics").exists() or (output / "training_logs").exists():
        return output / "final_report", output / "figures"
    return output, output / "figures"


def resolve_root(requested, candidates, label):
    choices = [requested] if requested else []
    choices.extend(Path(value) for value in candidates)
    for path in choices:
        if path and path.expanduser().exists():
            return path.expanduser().resolve()
    raise FileNotFoundError("Could not find {} root. Pass its explicit command-line option; checked: {}".format(
        label, ", ".join(str(path) for path in choices)))


def normalize(frame):
    frame = frame.rename(columns={column: ALIASES.get(column.strip().lower(), column.strip().lower())
                                  for column in frame.columns})
    if "ecg_scenario" in frame:
        frame["ecg_scenario"] = (frame.ecg_scenario.astype(str).str.lower().str.replace("-6", "m6", regex=False)
                                 .str.replace("snr_", "snr", regex=False))
    if "threshold_strategy" not in frame and "experiment_name" in frame:
        frame["threshold_strategy"] = STRATEGY
    return frame


def csv_candidates(root, per_class=False):
    preferred = ([root / "per_class_metrics.csv", root / "final_report" / "per_class_metrics.csv"] if per_class else
                 [root / "attention_ablation_summary.csv", root / "ablation_summary.csv",
                  root / "final_report" / "attention_ablation_summary.csv", root / "final_report" / "ablation_summary.csv"])
    existing = [path for path in preferred if path.exists()]
    if existing:
        return existing
    paths = list((root / "metrics").glob("*/*per_class.csv" if per_class else "*/seed_*.csv"))
    if not per_class:
        paths = [path for path in paths if "per_class" not in path.name and "threshold" not in path.name]
    return sorted(paths)


def load_frames(root, per_class=False):
    paths = csv_candidates(root, per_class)
    if not paths:
        raise FileNotFoundError("No {}metric CSVs found under {}".format("per-class " if per_class else "", root))
    frames = []
    for path in paths:
        frame = normalize(pd.read_csv(path))
        required = {"experiment_name", "ecg_scenario"}
        if required.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        raise ValueError("No CSV under {} has required columns experiment_name and ecg_scenario".format(root))
    return pd.concat(frames, ignore_index=True, sort=False)


def selected_rows(frame):
    frame = frame[frame.threshold_strategy.astype(str).str.lower() == STRATEGY].copy()
    frame = frame[frame.experiment_name.isin(MODELS) & frame.ecg_scenario.isin(SCENARIOS)]
    return frame


def validate(summary, per_class):
    missing_columns = [column for column in ["experiment_name", "seed", "ecg_scenario", "threshold_strategy"] + METRICS
                       if column not in summary]
    if missing_columns:
        raise ValueError("Overall metrics are missing columns: {}".format(missing_columns))
    found_models = set(summary.experiment_name)
    if found_models != set(MODELS):
        raise ValueError("Expected exactly six models; missing={}, unexpected={}".format(
            sorted(set(MODELS) - found_models), sorted(found_models - set(MODELS))))
    counts = summary.groupby(["experiment_name", "seed", "ecg_scenario"]).size()
    if (counts != 1).any():
        raise ValueError("Expected one per_class strategy row per model/seed/scenario; bad groups: {}".format(
            counts[counts != 1].to_dict()))
    expected = {(model, seed, scenario) for model in MODELS
                for seed in summary.loc[summary.experiment_name == model, "seed"].unique() for scenario in SCENARIOS}
    actual = set(summary[["experiment_name", "seed", "ecg_scenario"]].itertuples(index=False, name=None))
    missing = expected - actual
    if missing:
        raise ValueError("Missing model/seed/scenario rows: {}".format(sorted(missing)))
    seed_sets = {model: set(summary.loc[summary.experiment_name == model, "seed"]) for model in MODELS}
    if len({tuple(sorted(values)) for values in seed_sets.values()}) != 1:
        raise ValueError("All six models must have the same seeds: {}".format(seed_sets))
    if len(per_class):
        required = {"experiment_name", "seed", "ecg_scenario", "class_name", "f1", "roc_auc", "pr_auc"}
        if not required.issubset(per_class.columns):
            raise ValueError("Per-class metrics are missing columns: {}".format(sorted(required - set(per_class.columns))))
        bad_classes = set(per_class.class_name) - set(CLASSES)
        if bad_classes:
            raise ValueError("Unexpected classes: {}".format(sorted(bad_classes)))
        class_counts = per_class.groupby(["experiment_name", "seed", "ecg_scenario"]).class_name.nunique()
        if (class_counts != len(CLASSES)).any():
            raise ValueError("Expected all five classes per model/seed/scenario; bad groups: {}".format(
                class_counts[class_counts != len(CLASSES)].to_dict()))


def mean_values(summary):
    return summary.groupby(["experiment_name", "ecg_scenario"], as_index=False)[METRICS].mean()


def build_contributions(summary):
    means = mean_values(summary).set_index(["experiment_name", "ecg_scenario"])
    comparisons = [
        ("CBAM vs baseline", "cbam_xresnet1d101", "xresnet1d101_baseline"),
        ("SE vs baseline", "se_xresnet1d101", "xresnet1d101_baseline"),
        ("EMD with baseline", "xresnet1d101_emd_late_fusion", "xresnet1d101_baseline"),
        ("EMD with CBAM", "cbam_xresnet1d101_emd_late_fusion", "cbam_xresnet1d101"),
        ("EMD with SE", "se_xresnet1d101_emd_late_fusion", "se_xresnet1d101"),
        ("CBAM with EMD", "cbam_xresnet1d101_emd_late_fusion", "xresnet1d101_emd_late_fusion"),
        ("SE with EMD", "se_xresnet1d101_emd_late_fusion", "xresnet1d101_emd_late_fusion"),
    ]
    rows = []
    for label, model, reference in comparisons:
        for scenario in ["clean", "snr0", "snrm6"]:
            delta = means.loc[(model, scenario), METRICS] - means.loc[(reference, scenario), METRICS]
            rows.append({"comparison": label, "model": model, "reference_model": reference,
                         "scenario": scenario, **{"delta_" + metric: delta[metric] for metric in METRICS}})
        model_noisy = means.loc[(model, NOISY), METRICS].mean()
        ref_noisy = means.loc[(reference, NOISY), METRICS].mean()
        delta = model_noisy - ref_noisy
        rows.append({"comparison": label, "model": model, "reference_model": reference,
                     "scenario": "mean_noisy", **{"delta_" + metric: delta[metric] for metric in METRICS}})
    return pd.DataFrame(rows)


def build_robustness(summary):
    means = mean_values(summary).set_index(["experiment_name", "ecg_scenario"])
    rows = []
    for model in MODELS:
        clean = means.loc[(model, "clean")]
        for scenario in NOISY:
            noisy = means.loc[(model, scenario)]
            row = {"experiment_name": model, "display_name": DISPLAY[model], "scenario": scenario}
            for metric in METRICS:
                row[metric + "_drop"] = clean[metric] - noisy[metric]
                row[metric + "_retention"] = noisy[metric] / clean[metric] if clean[metric] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def save_figure(figures, name):
    plt.tight_layout()
    plt.savefig(figures / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.savefig(figures / (name + ".pdf"), bbox_inches="tight")
    plt.close()


def make_plots(figures, summary, complexity, per_class, training_root):
    figures.mkdir(parents=True, exist_ok=True)
    means = mean_values(summary)
    clean = means[means.ecg_scenario == "clean"].set_index("experiment_name").loc[MODELS]
    colors = plt.cm.tab10(np.arange(len(MODELS)))
    for metric in ["macro_roc_auc", "macro_f1"]:
        plt.figure(figsize=(8.5, 4.5)); plt.bar([DISPLAY[x] for x in MODELS], clean[metric], color=colors)
        plt.ylabel(METRIC_LABEL[metric]); plt.title("Clean-test {}".format(METRIC_LABEL[metric])); plt.xticks(rotation=20)
        save_figure(figures, "clean_" + metric)
    snr_x = [-6, 0, 6, 12, 24]
    snr_order = ["snrm6", "snr0", "snr6", "snr12", "snr24"]
    for metric in ["macro_roc_auc", "macro_f1"]:
        plt.figure(figsize=(8, 5))
        for model, color in zip(MODELS, colors):
            values = means[means.experiment_name == model].set_index("ecg_scenario").loc[snr_order, metric]
            plt.plot(snr_x, values, marker="o", label=DISPLAY[model], color=color)
        plt.xlabel("Signal-to-noise ratio (dB)"); plt.ylabel(METRIC_LABEL[metric]); plt.title("{} by noise level".format(METRIC_LABEL[metric])); plt.legend(ncol=2)
        save_figure(figures, metric + "_vs_snr")
    for scenario, title, name in [("clean", "Clean per-class F1", "clean_per_class_f1"),
                                  ("snrm6", "-6 dB per-class F1", "snrm6_per_class_f1")]:
        pc = per_class[per_class.ecg_scenario == scenario].groupby(["class_name", "experiment_name"]).f1.mean().unstack()
        pc.reindex(CLASSES)[MODELS].rename(columns=DISPLAY).plot(kind="bar", figsize=(10, 5))
        plt.ylabel("F1"); plt.xlabel("Diagnostic superclass"); plt.title(title); plt.xticks(rotation=0)
        save_figure(figures, name)
    for x_column, x_label, name in [
            ("parameter_count", "Parameter count", "parameters_vs_mean_noisy_macro_roc_auc"),
            ("inference_time_per_sample_ms", "Inference time per sample (ms)",
             "inference_time_vs_mean_noisy_macro_roc_auc")]:
        plt.figure(figsize=(8, 5)); plt.scatter(complexity[x_column], complexity.mean_noisy_macro_roc_auc, c=colors)
        for row in complexity.itertuples():
            plt.annotate(row.display_name, (getattr(row, x_column), row.mean_noisy_macro_roc_auc),
                         xytext=(4, 4), textcoords="offset points")
        plt.xlabel(x_label); plt.ylabel("Mean noisy Macro ROC-AUC"); plt.title("Efficiency and noisy performance")
        save_figure(figures, name)
    for model in MODELS[-2:]:
        paths = sorted((training_root / "training_logs" / model).glob("seed_*.csv"))
        if not paths:
            raise FileNotFoundError("No training log found for {} under {}".format(model, training_root))
        plt.figure(figsize=(8, 5))
        for path in paths:
            history = normalize(pd.read_csv(path))
            required = {"epoch", "train_loss", "valid_loss"}
            if not required.issubset(history.columns):
                raise ValueError("Training log {} is missing {}".format(path, sorted(required - set(history.columns))))
            seed = path.stem.replace("seed_", "")
            plt.plot(history.epoch, history.train_loss, label="Train (seed {})".format(seed))
            plt.plot(history.epoch, history.valid_loss, linestyle="--", label="Valid (seed {})".format(seed))
        plt.xlabel("Epoch"); plt.ylabel("BCE loss"); plt.title("{} training history".format(DISPLAY[model])); plt.legend()
        save_figure(figures, "training_loss_" + model)
    comparison_models = ["cbam_xresnet1d101", "se_xresnet1d101",
                         "cbam_xresnet1d101_emd_late_fusion", "se_xresnet1d101_emd_late_fusion"]
    for scenario, title, name in [("clean", "SE vs CBAM on clean ECG", "se_vs_cbam_clean"),
                                  ("snrm6", "SE vs CBAM at -6 dB", "se_vs_cbam_snrm6")]:
        values = means[means.ecg_scenario == scenario].set_index("experiment_name").loc[comparison_models]
        values[["macro_roc_auc", "macro_f1"]].rename(index=DISPLAY, columns=METRIC_LABEL).plot(
            kind="bar", figsize=(9, 5))
        plt.ylabel("Score"); plt.xlabel(""); plt.title(title); plt.xticks(rotation=15)
        save_figure(figures, name)


def write_markdown(path, clean, noisy, contributions, complexity, best):
    def table(frame):
        values = [[str(column) for column in frame.columns]]
        for row in frame.itertuples(index=False, name=None):
            values.append(["{:.4f}".format(value) if isinstance(value, (float, np.floating)) else str(value)
                           for value in row])
        widths = [max(len(row[index]) for row in values) for index in range(len(values[0]))]
        line = lambda row: "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        return "\n".join([line(values[0]), line(["-" * width for width in widths])] +
                         [line(row) for row in values[1:]])
    clean_table = clean[["display_name"] + METRICS].rename(columns={"display_name": "Model", **METRIC_LABEL})
    noisy_table = noisy[["display_name", "mean_noisy_macro_roc_auc", "mean_noisy_macro_f1",
                         "mean_macro_roc_auc_drop", "mean_macro_f1_drop"]].rename(columns={
                             "display_name": "Model", "mean_noisy_macro_roc_auc": "Mean noisy Macro ROC-AUC",
                             "mean_noisy_macro_f1": "Mean noisy Macro F1", "mean_macro_roc_auc_drop": "Mean ROC-AUC drop",
                             "mean_macro_f1_drop": "Mean F1 drop"})
    delta_cols = ["comparison", "scenario", "delta_macro_roc_auc", "delta_macro_f1"]
    clean_index = clean.set_index("experiment_name")
    noisy_index = noisy.set_index("experiment_name")
    complexity_table = complexity[["display_name", "parameter_count", "trainable_parameter_count",
                                   "inference_time_per_sample_ms", "mean_noisy_macro_roc_auc",
                                   "mean_noisy_auc_per_million_parameters"]].rename(columns={
                                       "display_name": "Model", "parameter_count": "Parameters",
                                       "trainable_parameter_count": "Trainable parameters",
                                       "inference_time_per_sample_ms": "Inference ms/sample",
                                       "mean_noisy_macro_roc_auc": "Mean noisy Macro ROC-AUC",
                                       "mean_noisy_auc_per_million_parameters": "AUC per million parameters"})
    se_base = contributions.set_index(["comparison", "scenario"])
    se_vs_base_clean = se_base.loc[("SE vs baseline", "clean"), "delta_macro_roc_auc"]
    se_vs_cbam_clean = (clean_index.loc["se_xresnet1d101", "macro_roc_auc"] -
                        clean_index.loc["cbam_xresnet1d101", "macro_roc_auc"])
    se_vs_cbam_m6 = (clean_index.loc["se_xresnet1d101", "macro_roc_auc"] * 0 +
                     se_base.loc[("SE vs baseline", "snrm6"), "delta_macro_roc_auc"] -
                     se_base.loc[("CBAM vs baseline", "snrm6"), "delta_macro_roc_auc"])
    emd_se_noisy = se_base.loc[("EMD with SE", "mean_noisy"), "delta_macro_roc_auc"]
    se_drop = noisy_index.loc["se_xresnet1d101", "mean_macro_roc_auc_drop"]
    se_emd_drop = noisy_index.loc["se_xresnet1d101_emd_late_fusion", "mean_macro_roc_auc_drop"]
    efficient = complexity.set_index("experiment_name").mean_noisy_auc_per_million_parameters.idxmax()
    text = """# SE/XResNet Attention Ablation Results

## 1. Study Scope

This report merges six models evaluated on clean ECG and five noisy scenarios. All values are calculated from result CSVs using `per_class_thresholds`; no values are estimated or manually entered.

## 2. Models Compared

The models are xResNet, CBAM-xResNet, SE-xResNet, and each architecture's EMD late-fusion counterpart.

## 3. Clean Test Performance

{}

## 4. Noisy Test Performance

{}

## 5. Performance at -6 dB

The best model at -6 dB Macro ROC-AUC is **{}**.

## 6. SE vs xResNet Baseline

On clean ECG, SE changes Macro ROC-AUC by **{:+.4f}** relative to xResNet.

## 7. SE vs CBAM

SE minus CBAM Macro ROC-AUC is **{:+.4f}** on clean ECG and **{:+.4f}** at -6 dB.

## 8. EMD Effect on SE

Adding EMD to SE changes mean noisy Macro ROC-AUC by **{:+.4f}**.

## 9. EMD Effect on SE Robustness

The mean Macro ROC-AUC drop is **{:.4f}** for SE-xResNet and **{:.4f}** for SE-xResNet + EMD. The smaller value is more robust.

## 10. Attention and EMD Deltas

Positive deltas favor the named model over its reference.

{}

## 11. Best Models and Parameter Efficiency

- Best clean Macro ROC-AUC: **{}**
- Best mean noisy Macro ROC-AUC: **{}**
- Best -6 dB Macro ROC-AUC: **{}**
- Smallest mean Macro ROC-AUC drop: **{}**
- Highest mean noisy Macro ROC-AUC per million parameters: **{}**

{}

## 12. Conclusions

The measured results select **{}** for clean ECG, **{}** on average across noisy ECG, and **{}** at -6 dB. Parameter efficiency favors **{}**. SE's clean result is {} the baseline and {} CBAM; adding EMD to SE {} its mean noisy Macro ROC-AUC.
""".format(table(clean_table), table(noisy_table), DISPLAY[best["best_minus6db_model"]],
           se_vs_base_clean, se_vs_cbam_clean, se_vs_cbam_m6, emd_se_noisy, se_drop, se_emd_drop,
           table(contributions[delta_cols]), DISPLAY[best["best_clean_model"]],
           DISPLAY[best["best_mean_noisy_model"]], DISPLAY[best["best_minus6db_model"]],
           DISPLAY[best["smallest_performance_drop_model"]], DISPLAY[efficient], table(complexity_table),
           DISPLAY[best["best_clean_model"]], DISPLAY[best["best_mean_noisy_model"]],
           DISPLAY[best["best_minus6db_model"]], DISPLAY[efficient],
           "above" if se_vs_base_clean > 0 else "below" if se_vs_base_clean < 0 else "equal to",
           "above" if se_vs_cbam_clean > 0 else "below" if se_vs_cbam_clean < 0 else "equal to",
           "improves" if emd_se_noisy > 0 else "reduces" if emd_se_noisy < 0 else "does not change")
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    four = resolve_root(args.four_model_root,
                        ["/content/drive/MyDrive/ECG/ablation_results_full_ptbxl",
                         "/content/drive/MyDrive/ECG/ablation_results", "results/ablation_results_full_ptbxl",
                         "results/ablation_study"], "four-model")
    se = resolve_root(args.se_root,
                      ["/content/drive/MyDrive/ECG/se_xresnet_ablation_results",
                       "/content/drive/MyDrive/ECG/ablation_results_se", "results/se_xresnet_ablation_results",
                       "results/ablation_results_se"], "SE")
    training_root = resolve_root(args.training_root, [se], "SE training")
    report_dir, figures_dir = output_directories(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)
    summary = selected_rows(pd.concat([load_frames(four), load_frames(se)], ignore_index=True, sort=False))
    per_class = selected_rows(pd.concat([load_frames(four, True), load_frames(se, True)], ignore_index=True, sort=False))
    validate(summary, per_class)
    summary = summary.sort_values(["experiment_name", "seed", "ecg_scenario"])
    per_class = per_class.sort_values(["experiment_name", "seed", "ecg_scenario", "class_name"])
    summary.to_csv(report_dir / "attention_ablation_summary.csv", index=False)
    summary[summary.ecg_scenario == "clean"].to_csv(report_dir / "attention_clean_comparison.csv", index=False)
    summary[summary.ecg_scenario != "clean"].to_csv(report_dir / "attention_snr_comparison.csv", index=False)
    per_class.to_csv(report_dir / "per_class_metrics.csv", index=False)
    contributions = build_contributions(summary); contributions.to_csv(report_dir / "attention_contributions.csv", index=False)
    robustness = build_robustness(summary); robustness.to_csv(report_dir / "robustness_metrics.csv", index=False)
    means = mean_values(summary)
    noisy = means[means.ecg_scenario.isin(NOISY)].groupby("experiment_name")[METRICS].mean()
    noisy.columns = ["mean_noisy_" + column for column in noisy.columns]
    drops = robustness.groupby("experiment_name")[[metric + "_drop" for metric in METRICS]].mean()
    drops.columns = ["mean_" + column for column in drops.columns]
    noisy = noisy.join(drops).reindex(MODELS).reset_index(); noisy["display_name"] = noisy.experiment_name.map(DISPLAY)
    noisy.to_csv(report_dir / "mean_noisy_metrics.csv", index=False)
    clean_means = means[means.ecg_scenario == "clean"].set_index("experiment_name").reindex(MODELS)
    clean = clean_means.reset_index(); clean["display_name"] = clean.experiment_name.map(DISPLAY)
    complexity_columns = ["parameter_count", "trainable_parameter_count", "inference_time_per_sample_ms",
                          "training_time_seconds", "best_epoch", "best_valid_loss", "actual_batch_size"]
    missing_complexity = [column for column in complexity_columns if column not in summary]
    if missing_complexity:
        raise ValueError("Overall metrics are missing model complexity columns: {}".format(missing_complexity))
    complexity = summary.groupby("experiment_name")[complexity_columns].mean().reindex(MODELS).reset_index()
    complexity["display_name"] = complexity.experiment_name.map(DISPLAY)
    complexity["clean_macro_roc_auc"] = clean_means.macro_roc_auc.values
    complexity["mean_noisy_macro_roc_auc"] = noisy.set_index("experiment_name").loc[MODELS, "mean_noisy_macro_roc_auc"].values
    complexity["mean_noisy_auc_per_million_parameters"] = (
        complexity.mean_noisy_macro_roc_auc / (complexity.parameter_count / 1_000_000))
    complexity.to_csv(report_dir / "model_complexity.csv", index=False)
    minus6 = means[means.ecg_scenario == "snrm6"].set_index("experiment_name")
    best = {
        "best_clean_model": clean_means.macro_roc_auc.idxmax(),
        "best_mean_noisy_model": noisy.set_index("experiment_name").mean_noisy_macro_roc_auc.idxmax(),
        "best_minus6db_model": minus6.macro_roc_auc.idxmax(),
        "smallest_performance_drop_model": noisy.set_index("experiment_name").mean_macro_roc_auc_drop.idxmin(),
        "best_clean_macro_f1_model": clean_means.macro_f1.idxmax(),
    }
    (report_dir / "best_model_summary.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    make_plots(figures_dir, summary, complexity, per_class, training_root)
    write_markdown(report_dir / "SE_XRESNET_ABLATION_RESULTS.md", clean, noisy, contributions, complexity, best)
    print("Merged {} and {} into {}; figures in {}".format(four, se, report_dir, figures_dir))


if __name__ == "__main__":
    main()
