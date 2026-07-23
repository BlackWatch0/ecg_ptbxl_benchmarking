#!/usr/bin/env python3
"""Build the final three-domain report for the original PTB-XL models."""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = [
    "xresnet1d101",
    "resnet1d_wang",
    "lstm",
    "lstm_bidir",
    "fcn_wang",
    "inception1d",
    "Wavelet+NN",
]
WAVELET_MODEL = "Wavelet+NN"
DOMAINS = ["clean", "noisy", "denoised"]
SNRS = [24, 12, 6, 0, -6]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
REQUIRED_METRICS = ["macro_roc_auc", "macro_f1"]
DISPLAY = {
    "xresnet1d101": "XResNet1D-101",
    "resnet1d_wang": "ResNet1D (Wang)",
    "lstm": "LSTM",
    "lstm_bidir": "Bidirectional LSTM",
    "fcn_wang": "FCN (Wang)",
    "inception1d": "Inception1D",
    "Wavelet+NN": "Wavelet + neural network",
}
ALIASES = {
    "model": "model_name", "experiment_name": "model_name", "experiment": "model_name",
    "dataset_domain": "domain", "ecg_domain": "domain", "scenario": "domain", "ecg_scenario": "domain",
    "target_snr_db": "snr_db", "snr": "snr_db", "random_seed": "seed",
    "class": "class_name", "label": "class_name", "macro_auc": "macro_roc_auc",
    "macro_auroc": "macro_roc_auc", "macro_f1_score": "macro_f1",
    "parameters": "parameter_count", "num_parameters": "parameter_count",
    "trainable_parameters": "trainable_parameter_count",
    "inference_ms": "inference_time_per_sample_ms",
    "inference_ms_per_sample": "inference_time_per_sample_ms",
    "validation_loss": "valid_loss", "val_loss": "valid_loss", "training_loss": "train_loss",
}
METRIC_LABELS = {"macro_roc_auc": "Macro ROC-AUC", "macro_f1": "Macro F1"}
IDENTITY = {"model_name", "display_name", "seed", "domain", "snr_db", "class_name", "epoch"}
COMPLEXITY_REQUIRED = ["parameter_count", "trainable_parameter_count", "inference_time_per_sample_ms"]
RUN_METADATA = ["best_epoch", "best_valid_loss", "training_time_seconds", "actual_batch_size",
                "crop_length", "crop_stride"]
COMPLEXITY = COMPLEXITY_REQUIRED + RUN_METADATA


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", "--results-dir", dest="input_root", type=Path, required=True,
                        help="Runner output containing metric, per-class, and history CSVs")
    parser.add_argument("--output-dir", type=Path,
                        help="Report directory (default: INPUT_ROOT/final_report)")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--excluded-wavelet-status",
                        help="Explicit reason Wavelet+NN was excluded; never creates Wavelet measurements")
    return parser.parse_args(argv)


def normalize(frame):
    frame = frame.copy()
    canonical_present = {column.strip().lower() for column in frame.columns}
    drop, rename = [], {}
    for column in frame.columns:
        normalized = column.strip().lower()
        target = ALIASES.get(normalized, normalized)
        if target != normalized and target in canonical_present:
            drop.append(column)
        else:
            rename[column] = target
    frame = frame.drop(columns=drop).rename(columns=rename)
    if "domain" in frame:
        domains = frame.domain.astype(str).str.strip().str.lower()
        frame["domain"] = domains.replace({
            "clean_ecg": "clean", "original": "clean", "noise": "noisy",
            "noisy_ecg": "noisy", "denoised_ecg": "denoised", "filtered": "denoised",
        })
        frame.loc[domains.str.startswith("noisy_"), "domain"] = "noisy"
        frame.loc[domains.str.startswith("denoised_"), "domain"] = "denoised"
    if "snr_db" in frame:
        frame["snr_db"] = pd.to_numeric(frame.snr_db, errors="coerce")
    if "seed" in frame:
        frame["seed"] = pd.to_numeric(frame.seed, errors="raise").astype(int)
    return frame


def load_inputs(root):
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError("Input root does not exist: {}".format(root))
    frames = {"metrics": [], "per_class": [], "history": []}
    for path in sorted(root.rglob("*.csv")):
        if "final_report" in path.parts:
            continue
        frame = normalize(pd.read_csv(path))
        if "training_logs" in path.parts and {"epoch", "train_loss", "valid_loss"}.issubset(frame.columns):
            if "model_name" not in frame:
                frame["model_name"] = path.parent.name
            if "seed" not in frame:
                try:
                    frame["seed"] = int(path.stem.replace("seed_", ""))
                except ValueError:
                    pass
        columns = set(frame.columns)
        if {"model_name", "seed", "epoch", "train_loss", "valid_loss"}.issubset(columns):
            frames["history"].append(frame)
        elif {"model_name", "seed", "domain", "class_name"}.issubset(columns):
            frames["per_class"].append(frame)
        elif {"model_name", "seed", "domain"}.issubset(columns):
            frames["metrics"].append(frame)
    missing = [name for name, values in frames.items() if not values]
    if missing:
        raise FileNotFoundError("No runner {} CSVs found under {}".format(
            ", ".join(name.replace("_", "-") for name in missing), root))
    loaded = tuple(pd.concat(frames[name], ignore_index=True, sort=False)
                   for name in ("metrics", "per_class", "history"))
    selected = []
    for frame in loaded[:2]:
        if "threshold_strategy" in frame:
            frame = frame[frame.threshold_strategy == "per_class_thresholds"].copy()
        selected.append(frame)
    return selected[0], selected[1], loaded[2]


def wavelet_exclusion_from_runner(root):
    path = Path(root) / "config" / "wavelet_nn_status.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("model") != WAVELET_MODEL or value.get("status") not in {
            "excluded", "unsupported", "unsupported_separate_pipeline"} or not value.get("reason"):
        raise ValueError("Invalid explicit Wavelet status in {}".format(path))
    return "{}: {}".format(value["status"], value["reason"])


def expected_keys(models, seeds):
    return {(model, seed, "clean", None) for model in models for seed in seeds} | {
        (model, seed, domain, snr) for model in models for seed in seeds
        for domain in ("noisy", "denoised") for snr in SNRS
    }


def row_keys(frame):
    return {(row.model_name, int(row.seed), row.domain,
             None if row.domain == "clean" else int(row.snr_db))
            for row in frame.itertuples()}


def validate(metrics, per_class, history, seeds, wavelet_status):
    expected_models = MODELS if not wavelet_status else MODELS[:-1]
    found_models = set(metrics.model_name)
    if wavelet_status and WAVELET_MODEL in found_models:
        raise ValueError("Wavelet+NN has measurements and cannot also have an excluded status")
    if found_models != set(expected_models):
        raise ValueError("Expected models do not match: missing={}, unexpected={}".format(
            sorted(set(expected_models) - found_models), sorted(found_models - set(expected_models))))
    for label, frame in (("overall metrics", metrics), ("per-class metrics", per_class)):
        required = {"model_name", "seed", "domain", "snr_db"}
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError("{} missing columns: {}".format(label, sorted(missing_columns)))
        if set(frame.domain) != set(DOMAINS):
            raise ValueError("{} domains must be exactly {} (found {})".format(
                label, DOMAINS, sorted(set(frame.domain))))
        if set(frame.seed) != set(seeds):
            raise ValueError("{} seeds must be exactly {} (found {})".format(
                label, sorted(seeds), sorted(set(frame.seed))))
        bad_clean = frame[(frame.domain == "clean") & frame.snr_db.notna()]
        noisy = frame[frame.domain != "clean"]
        if len(bad_clean) or set(noisy.snr_db.astype(int)) != set(SNRS) or noisy.snr_db.isna().any():
            raise ValueError("{} must use no SNR for clean and SNRs {} for noisy/denoised".format(label, SNRS))
    missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in metrics]
    if missing_metrics:
        raise ValueError("Overall metrics missing required values: {}".format(missing_metrics))
    missing_complexity = [column for column in COMPLEXITY_REQUIRED if column not in metrics]
    if missing_complexity:
        raise ValueError("Overall metrics missing model complexity columns: {}".format(missing_complexity))
    if set(row_keys(metrics)) != expected_keys(expected_models, seeds) or len(metrics) != len(expected_keys(expected_models, seeds)):
        raise ValueError("Overall metrics must contain exactly one row per expected model/seed/domain/SNR")
    if metrics[REQUIRED_METRICS].isna().any().any():
        raise ValueError("Required overall metric values cannot be missing")
    class_metric_columns = [column for column in ("roc_auc", "pr_auc", "f1") if column in per_class]
    if not class_metric_columns:
        raise ValueError("Per-class metrics require at least one of roc_auc, pr_auc, or f1")
    groups = per_class.groupby(["model_name", "seed", "domain", "snr_db"], dropna=False)
    class_sets = groups.class_name.apply(lambda values: tuple(sorted(set(values))))
    if len(set(class_sets)) != 1 or groups.size().ne(len(class_sets.iloc[0])).any():
        raise ValueError("Every model/seed/domain/SNR requires the same unique class set")
    if set(row_keys(per_class.drop_duplicates(["model_name", "seed", "domain", "snr_db"]))) != expected_keys(expected_models, seeds):
        raise ValueError("Per-class metrics are missing expected model/seed/domain/SNR groups")
    history_required = {"model_name", "seed", "epoch", "train_loss", "valid_loss"}
    if not history_required.issubset(history):
        raise ValueError("History missing columns: {}".format(sorted(history_required - set(history))))
    history_pairs = set(history[["model_name", "seed"]].itertuples(index=False, name=None))
    expected_pairs = {(model, seed) for model in expected_models for seed in seeds}
    if history_pairs != expected_pairs:
        raise ValueError("History model/seed pairs do not match expected pairs")
    duplicate_history = history.duplicated(["model_name", "seed", "epoch"])
    if duplicate_history.any():
        raise ValueError("History contains duplicate model/seed/epoch rows")
    return expected_models


def numeric_value_columns(frame, excluded=()):
    excluded = IDENTITY | set(excluded)
    return [column for column in frame.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]


def aggregate(frame, groups, values):
    result = frame.groupby(groups, dropna=False)[values].agg(["mean", "std", "count"])
    result.columns = ["{}_{}".format(column, statistic) for column, statistic in result.columns]
    return result.reset_index()


def metric_mean_column(metric):
    return metric + "_mean"


def save_figure(figures, name, fig=None):
    fig = plt.gcf() if fig is None else fig
    fig.tight_layout()
    fig.savefig(figures / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(figures / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_slug(model):
    return model.replace("+", "plus").lower()


def line_plot(data, models, domain, metric, figures):
    subset = data[data.domain == domain]
    plt.figure(figsize=(9, 5.5))
    for model in models:
        rows = subset[subset.model_name == model].sort_values("snr_db")
        plt.plot(rows.snr_db, rows[metric], marker="o", label=DISPLAY[model])
    plt.xlabel("Signal-to-noise ratio (dB)")
    plt.ylabel(METRIC_LABELS[metric])
    plt.title("{} on {} ECG".format(METRIC_LABELS[metric], domain))
    plt.legend(fontsize=8, ncol=2)
    save_figure(figures, "{}_{}_vs_snr".format(domain, metric))


def make_plots(figures, metrics, per_class, history, complexity, models):
    figures.mkdir(parents=True, exist_ok=True)
    means = metrics.groupby(["model_name", "domain", "snr_db"], dropna=False)[REQUIRED_METRICS].mean().reset_index()
    clean = means[means.domain == "clean"].set_index("model_name")
    for domain in ("noisy", "denoised"):
        for metric in REQUIRED_METRICS:
            line_plot(means, models, domain, metric, figures)
    for metric in REQUIRED_METRICS:
        plt.figure(figsize=(9, 5.5))
        for domain, linestyle in (("noisy", "-"), ("denoised", "--")):
            for model in models:
                rows = means[(means.domain == domain) & (means.model_name == model)].sort_values("snr_db")
                drops = clean.loc[model, metric] - rows[metric]
                plt.plot(rows.snr_db, drops, linestyle=linestyle, marker="o",
                         label="{} ({})".format(DISPLAY[model], domain))
        plt.xlabel("Signal-to-noise ratio (dB)"); plt.ylabel("Drop in {} from clean".format(METRIC_LABELS[metric]))
        plt.title("Performance degradation by domain"); plt.legend(fontsize=6, ncol=2)
        save_figure(figures, "{}_drops_from_clean".format(metric))
    class_metric = "f1" if "f1" in per_class else "roc_auc"
    for domain, snr, name, title in (("clean", None, "clean_per_class", "Clean ECG per-class performance"),
                                     ("noisy", -6, "minus6db_per_class", "Noisy and denoised ECG at -6 dB")):
        if domain == "clean":
            rows = per_class[per_class.domain == "clean"]
            table = rows.groupby(["class_name", "model_name"])[class_metric].mean().unstack()
        else:
            rows = per_class[(per_class.domain != "clean") & (per_class.snr_db == snr)].copy()
            rows["series"] = rows.model_name.map(DISPLAY) + " (" + rows.domain + ")"
            table = rows.groupby(["class_name", "series"])[class_metric].mean().unstack()
        table.plot(kind="bar", figsize=(12, 6))
        plt.xlabel("Diagnostic class"); plt.ylabel("Per-class {}".format(class_metric.replace("_", " ").upper()))
        plt.title(title); plt.xticks(rotation=0); plt.legend(fontsize=6, ncol=2)
        save_figure(figures, name + "_" + class_metric)
    mean_domain = means[means.domain != "clean"].groupby(["model_name", "domain"])["macro_roc_auc"].mean()
    for column, label, name in (("parameter_count", "Number of parameters", "parameters_tradeoff"),
                                ("inference_time_per_sample_ms", "Inference time per sample (ms)", "inference_tradeoff")):
        plt.figure(figsize=(8.5, 5.5))
        for model in models:
            row = complexity.set_index("model_name").loc[model]
            plt.scatter(row[column], mean_domain.loc[(model, "denoised")], s=55)
            plt.annotate(DISPLAY[model], (row[column], mean_domain.loc[(model, "denoised")]),
                         xytext=(4, 4), textcoords="offset points", fontsize=8)
        plt.xlabel(label); plt.ylabel("Mean denoised Macro ROC-AUC"); plt.title("Performance and efficiency trade-off")
        save_figure(figures, name)
    for model in models:
        plt.figure(figsize=(8, 5))
        for seed, rows in history[history.model_name == model].groupby("seed"):
            rows = rows.sort_values("epoch")
            plt.plot(rows.epoch, rows.train_loss, label="Training loss (seed {})".format(seed))
            plt.plot(rows.epoch, rows.valid_loss, linestyle="--", label="Validation loss (seed {})".format(seed))
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("{} training history".format(DISPLAY[model])); plt.legend()
        save_figure(figures, "training_loss_" + figure_slug(model))
    accuracy_columns = {"train_accuracy", "valid_accuracy"}
    for model in models:
        for seed, rows in history[history.model_name == model].groupby("seed"):
            if not accuracy_columns.issubset(rows.columns) or rows[list(accuracy_columns)].isna().any().any():
                warnings.warn(
                    "Skipping accuracy figure for {} seed {} because history lacks complete accuracy fields".format(
                        model, seed), RuntimeWarning)
                continue
            rows = rows.sort_values("epoch")
            fig, axis = plt.subplots(figsize=(8, 5))
            axis.plot(rows.epoch, rows.train_accuracy, label="Training accuracy")
            axis.plot(rows.epoch, rows.valid_accuracy, linestyle="--", label="Validation accuracy")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Accuracy")
            axis.set_ylim(0, 1)
            axis.set_title("{} training and validation accuracy (seed {})".format(
                DISPLAY[model], seed))
            axis.legend()
            axis.grid(True, alpha=0.3)
            save_figure(figures, "training_validation_accuracy_{}_seed_{}".format(
                figure_slug(model), seed), fig)
    for metric in REQUIRED_METRICS:
        comparison = means[means.domain != "clean"].pivot_table(
            index=["model_name", "snr_db"], columns="domain", values=metric).reset_index()
        plt.figure(figsize=(7, 6))
        for model in models:
            rows = comparison[comparison.model_name == model]
            plt.scatter(rows.noisy, rows.denoised, label=DISPLAY[model])
        limits = [comparison[["noisy", "denoised"]].min().min(), comparison[["noisy", "denoised"]].max().max()]
        plt.plot(limits, limits, color="black", linestyle=":", label="Equal performance")
        plt.xlabel("Noisy ECG {}".format(METRIC_LABELS[metric])); plt.ylabel("Denoised ECG {}".format(METRIC_LABELS[metric]))
        plt.title("Noisy versus denoised ECG"); plt.legend(fontsize=7, ncol=2)
        save_figure(figures, "noisy_vs_denoised_" + metric)


def markdown_table(frame):
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]):
        display[column] = display[column].map(lambda value: "{:.4f}".format(value) if pd.notna(value) else "")
    values = [[str(column) for column in display.columns]]
    values.extend([["" if pd.isna(value) else str(value) for value in row]
                   for row in display.itertuples(index=False, name=None)])
    widths = [max(len(row[index]) for row in values) for index in range(len(values[0]))]
    render = lambda row: "| " + " | ".join(value.ljust(widths[index])
                                               for index, value in enumerate(row)) + " |"
    return "\n".join([render(values[0]), render(["-" * width for width in widths])] +
                     [render(row) for row in values[1:]])


def write_markdown(path, clean, noisy, denoised, contributions, robustness, complexity, best, wavelet_status):
    status = ("Wavelet+NN was explicitly excluded: **{}**. No Wavelet values were generated."
              .format(wavelet_status) if wavelet_status else
              "All seven original benchmark models, including Wavelet+NN, are represented by runner measurements.")
    columns = ["display_name", "macro_roc_auc_mean", "macro_roc_auc_std", "macro_f1_mean", "macro_f1_std"]
    text = """# Original Models Benchmark Results

## Scope and Data Integrity

This report compares original PTB-XL benchmark models on clean, noisy, and denoised ECG. No metric is manually entered or estimated; tables and figures are calculated from runner CSVs. {}

## Clean ECG

{}

## Noisy ECG by SNR

{}

## Denoised ECG by SNR

{}

## Denoising Contributions

Positive values indicate improvement after denoising the same model, seed, and SNR.

{}

## Robustness Relative to Clean ECG

{}

## Model Complexity

{}

## Best Measured Models

- Best clean Macro ROC-AUC: **{}** ({:.4f})
- Best mean noisy Macro ROC-AUC: **{}** ({:.4f})
- Best mean denoised Macro ROC-AUC: **{}** ({:.4f})
- Best noisy Macro ROC-AUC at -6 dB: **{}** ({:.4f})
- Best denoised Macro ROC-AUC at -6 dB: **{}** ({:.4f})

## Interpretation

Denoising contributions are paired differences, while robustness values are drops from each model's own clean score. Model selection statements above follow the measured Macro ROC-AUC values without substituting absent models.
""".format(status, markdown_table(clean[columns]), markdown_table(noisy), markdown_table(denoised),
           markdown_table(contributions), markdown_table(robustness), markdown_table(complexity),
           DISPLAY[best["best_clean"]["model_name"]], best["best_clean"]["macro_roc_auc"],
           DISPLAY[best["best_mean_noisy"]["model_name"]], best["best_mean_noisy"]["macro_roc_auc"],
           DISPLAY[best["best_mean_denoised"]["model_name"]], best["best_mean_denoised"]["macro_roc_auc"],
           DISPLAY[best["best_noisy_minus6db"]["model_name"]], best["best_noisy_minus6db"]["macro_roc_auc"],
           DISPLAY[best["best_denoised_minus6db"]["model_name"]], best["best_denoised_minus6db"]["macro_roc_auc"])
    path.write_text(text, encoding="utf-8")


def best_entry(frame):
    row = frame.loc[frame.macro_roc_auc.idxmax()]
    return {"model_name": row.model_name, "macro_roc_auc": float(row.macro_roc_auc)}


def build_report(input_root, output_dir, seeds, wavelet_status=None):
    metrics, per_class, history = load_inputs(input_root)
    wavelet_status = wavelet_status or wavelet_exclusion_from_runner(input_root)
    models = validate(metrics, per_class, history, seeds, wavelet_status)
    output_dir = (output_dir or input_root / "final_report").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"; figures.mkdir(exist_ok=True)
    metrics = metrics.sort_values(["model_name", "seed", "domain", "snr_db"], na_position="first")
    metrics.insert(1, "display_name", metrics.model_name.map(DISPLAY))
    per_class = per_class.sort_values(["model_name", "seed", "domain", "snr_db", "class_name"], na_position="first")
    metric_values = numeric_value_columns(metrics, COMPLEXITY)
    metrics.to_csv(output_dir / "benchmark_summary.csv", index=False)
    clean = aggregate(metrics[metrics.domain == "clean"], ["model_name", "display_name"], metric_values)
    noisy = aggregate(metrics[metrics.domain == "noisy"], ["model_name", "display_name", "snr_db"], metric_values)
    denoised = aggregate(metrics[metrics.domain == "denoised"], ["model_name", "display_name", "snr_db"], metric_values)
    clean.to_csv(output_dir / "clean_comparison.csv", index=False)
    noisy.to_csv(output_dir / "noisy_snr_comparison.csv", index=False)
    denoised.to_csv(output_dir / "denoised_snr_comparison.csv", index=False)
    pairing = metrics[metrics.domain != "clean"].pivot(index=["model_name", "display_name", "seed", "snr_db"],
                                                        columns="domain", values=metric_values)
    contributions = pd.DataFrame(index=pairing.index)
    for metric in metric_values:
        contributions["{}_noisy".format(metric)] = pairing[(metric, "noisy")]
        contributions["{}_denoised".format(metric)] = pairing[(metric, "denoised")]
        contributions["{}_improvement".format(metric)] = pairing[(metric, "denoised")] - pairing[(metric, "noisy")]
    contributions = contributions.reset_index()
    contributions.to_csv(output_dir / "denoising_contributions.csv", index=False)
    clean_seed = metrics[metrics.domain == "clean"].set_index(["model_name", "seed"])
    robust_rows = []
    for row in metrics[metrics.domain != "clean"].itertuples(index=False):
        clean_row = clean_seed.loc[(row.model_name, row.seed)]
        values = {"model_name": row.model_name, "display_name": row.display_name, "seed": row.seed,
                  "domain": row.domain, "snr_db": row.snr_db}
        for metric in metric_values:
            values[metric + "_clean"] = clean_row[metric]
            values[metric + "_value"] = getattr(row, metric)
            values[metric + "_drop"] = clean_row[metric] - getattr(row, metric)
            values[metric + "_retention"] = getattr(row, metric) / clean_row[metric] if clean_row[metric] else np.nan
        robust_rows.append(values)
    robustness = pd.DataFrame(robust_rows)
    robustness.to_csv(output_dir / "robustness_metrics.csv", index=False)
    mean_domain = aggregate(metrics, ["model_name", "display_name", "domain"], metric_values)
    mean_domain.to_csv(output_dir / "mean_domain_metrics.csv", index=False)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    complexity_values = [column for column in COMPLEXITY if column in metrics]
    complexity = aggregate(metrics, ["model_name", "display_name"], complexity_values)
    complexity.to_csv(output_dir / "model_complexity.csv", index=False)
    means = metrics.groupby(["model_name", "domain", "snr_db"], dropna=False)["macro_roc_auc"].mean().reset_index()
    domain_means = metrics[metrics.domain != "clean"].groupby(["model_name", "domain"], as_index=False).macro_roc_auc.mean()
    best = {
        "wavelet_status": wavelet_status or "included",
        "expected_seeds": list(seeds),
        "best_clean": best_entry(means[means.domain == "clean"]),
        "best_mean_noisy": best_entry(domain_means[domain_means.domain == "noisy"]),
        "best_mean_denoised": best_entry(domain_means[domain_means.domain == "denoised"]),
        "best_noisy_minus6db": best_entry(means[(means.domain == "noisy") & (means.snr_db == -6)]),
        "best_denoised_minus6db": best_entry(means[(means.domain == "denoised") & (means.snr_db == -6)]),
    }
    (output_dir / "best_model_summary.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    make_plots(figures, metrics, per_class, history, complexity.rename(columns={
        "parameter_count_mean": "parameter_count",
        "inference_time_per_sample_ms_mean": "inference_time_per_sample_ms",
    }), models)
    write_markdown(output_dir / "ORIGINAL_MODELS_BENCHMARK_RESULTS.md", clean, noisy, denoised,
                   contributions, robustness, complexity, best, wavelet_status)
    return output_dir


def main(argv=None):
    args = parse_args(argv)
    output = build_report(args.input_root, args.output_dir, args.expected_seeds, args.excluded_wavelet_status)
    print("Original-model benchmark report written to {}".format(output))


if __name__ == "__main__":
    main()
