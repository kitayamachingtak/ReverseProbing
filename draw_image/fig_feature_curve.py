#F1 against feature count, one panel per dataset.
#Each point is the mean over XGBoost and CatBoost, with error bars showing the spread between them.

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from draw_image import common as c

CLASSIFIERS = ["XGBoost", "CatBoost"]
X_POS = list(range(len(c.CONFIGS)))
X_LABELS = ["93", "120", "204", "MAX"]


def collect(dataset, metric):
    out = {}
    for model in c.MODELS:
        xs, means, errs = [], [], []
        for x, config in zip(X_POS, c.CONFIGS):
            metrics = c.load_metrics(model, config, dataset)
            if not metrics:
                continue
            values = [metrics[k][metric] for k in CLASSIFIERS if k in metrics]
            if values:
                xs.append(x)
                means.append(np.mean(values))
                errs.append(np.std(values))
        if len(xs) >= 2:
            out[model] = (xs, means, errs)
    return out


def plot(dataset, metric):
    fig, ax = plt.subplots(figsize=(7, 5))

    for model, (xs, means, errs) in collect(dataset, metric).items():
        ax.errorbar(xs, means, yerr=errs, label=model,
                    color=c.MODEL_COLORS[model], marker="o", markersize=5.5,
                    linewidth=1.8, capsize=3, capthick=1.2, elinewidth=1.0,
                    alpha=0.9, zorder=3)

    ax.set_xticks(X_POS)
    ax.set_xticklabels(X_LABELS, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Number of features", fontsize=15)
    ax.set_ylabel(metric.upper() if metric == "auprc" else "F1", fontsize=15)
    ax.set_title(c.DATASET_TITLES[dataset], fontsize=16,
                 fontweight="bold", pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="f1", choices=["f1", "auprc", "auc"])
    args = ap.parse_args()

    for dataset in c.DATASETS:
        fig = plot(dataset, args.metric)
        c.save(fig, f"feature_curve_{args.metric}_{dataset}")

    # MAX means 454 features for 7-8B models and 886 for 70B models, so the
    # last x position is not the same configuration across curves.
    print("note: the MAX tick is 454 features for 7-8B models, 886 for 70B")


if __name__ == "__main__":
    main()