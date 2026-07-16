"""CSV and Matplotlib reporting for time-domain robustness results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .analysis import METRIC_COLUMNS, snr_sort_key


def rankings(metrics):
    data = metrics[metrics.feature != "__macro__"].copy()
    data["rank_by_nae"] = data.groupby(["comparison", "SNR", "evaluation_level", "aggregation"], dropna=False).nae.rank(method="min")
    return data.sort_values(["comparison", "SNR", "evaluation_level", "aggregation", "rank_by_nae"])


def denoising_improvement(metrics):
    keys = ["SNR", "evaluation_level", "aggregation", "feature"]
    noisy, denoised = metrics[metrics.comparison == "noisy"].set_index(keys), metrics[metrics.comparison == "denoised"].set_index(keys)
    rows = []
    for key in noisy.index.intersection(denoised.index):
        row = dict(zip(keys, key))
        for metric in METRIC_COLUMNS:
            row[metric + "_noisy"], row[metric + "_denoised"] = noisy.loc[key, metric], denoised.loc[key, metric]
            row[metric + "_improvement"] = noisy.loc[key, metric] - denoised.loc[key, metric] if metric in {"mae", "rmse", "nae", "absolute_mean", "absolute_median"} else denoised.loc[key, metric] - noisy.loc[key, metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / (name + ".png"), dpi=150)
    fig.savefig(output / (name + ".pdf"))
    plt.close(fig)


def make_plots(output, metrics):
    data = metrics[(metrics.feature != "__macro__") & (metrics.evaluation_level == "beat")].copy()
    if data.empty:
        return
    data["SNR_order"] = data.SNR.map(snr_sort_key)
    for comparison, subset in data.groupby("comparison"):
        subset = subset.sort_values("SNR_order")
        pivot = subset.pivot_table(index="feature", columns="SNR", values="nae", aggfunc="first").sort_index(axis=1)
        fig, axis = plt.subplots(figsize=(9, 6))
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns)
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels(pivot.index)
        axis.set_title("{} NAE by SNR".format(comparison))
        fig.colorbar(image, ax=axis, label="NAE")
        _save(fig, output, "heatmap_{}_nae".format(comparison))
    fig, axis = plt.subplots(figsize=(10, 5))
    for (comparison, feature), subset in data.groupby(["comparison", "feature"]):
        subset = subset.sort_values("SNR_order")
        axis.plot(subset.SNR.fillna(-999), subset.nae, marker="o", label="{}: {}".format(comparison, feature))
    axis.set_xlabel("SNR (dB; -999 denotes unspecified)")
    axis.set_ylabel("NAE")
    axis.set_title("Robustness across SNR levels")
    axis.legend(fontsize=6, ncol=2)
    _save(fig, output, "snr_robustness")


def write_outputs(output_dir, quality, matching, metrics, bootstrap, errors, all_errors=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    quality.to_csv(output / "quality_report.csv", index=False)
    matching.to_csv(output / "matching_report.csv", index=False)
    metrics.to_csv(output / "feature_metrics.csv", index=False)
    metrics[metrics.feature == "__macro__"].to_csv(output / "macro_overall.csv", index=False)
    bootstrap.to_csv(output / "bootstrap_samples.csv", index=False)
    rankings(metrics).to_csv(output / "feature_ranking.csv", index=False)
    denoising_improvement(metrics).to_csv(output / "denoising_improvement.csv", index=False)
    errors.to_csv(output / ("sample_errors_all.csv" if all_errors else "sample_errors_top100.csv"), index=False)
    make_plots(output, metrics)
