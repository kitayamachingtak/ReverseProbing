"""Feature importance heatmap, one column per (model, configuration) pair.

    python draw_image/fig_importance_heatmap.py --classifier XGBoost
    python draw_image/fig_importance_heatmap.py --classifier CatBoost --palette ember
    python draw_image/fig_importance_heatmap.py --classifier LogisticRegression \
        --groups signal --config max --palette purple

Cells give the share of TOTAL importance carried by the top-n features of each
group, so a column sums to the top-n coverage rather than to 100 percent.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from draw_image import common as c

TITLE = "Top feature importance across models and configurations"


def build(classifier, groups, configs, top_n):
    """Returns (row labels, {dataset: matrix}, {dataset: auprc list})."""
    rows = c.FAMILIES if groups == "family" else c.SIGNALS
    matrices, auprcs = {}, {}

    for dataset in c.DATASETS:
        matrix = np.full((len(rows), len(c.MODELS) * len(configs)), np.nan)
        auprc = []
        for mi, model in enumerate(c.MODELS):
            for ci, config in enumerate(configs):
                col = mi * len(configs) + ci
                imp = c.load_importance(model, config, dataset, classifier)
                metrics = c.load_metrics(model, config, dataset)
                auprc.append(metrics[classifier]["auprc"]
                             if metrics and classifier in metrics else np.nan)
                if imp is None:
                    continue
                shares = c.group_importance(imp, groups, top_n)
                matrix[:, col] = [shares[r] for r in rows]
        matrices[dataset] = matrix
        auprcs[dataset] = auprc

    return rows, matrices, auprcs


def plot(rows, matrices, auprcs, configs, palette, annotate, label):
    cmap = LinearSegmentedColormap.from_list("imp", c.PALETTES[palette])
    cmap.set_bad("#ffffff")

    mats = [matrices[d] for d in c.DATASETS]
    vmax = max(np.nanmax(m) for m in mats if not np.all(np.isnan(m)))
    ncol = len(c.MODELS) * len(configs)
    labels = ["MAX" if x == "max" else x for x in configs]

    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=(1.9 + 0.42 * ncol,
                                      1.4 + 0.42 * len(rows) * 2))

    for ax, dataset, m in zip(axes, c.DATASETS, mats):
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=10.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if annotate:
            for i in range(m.shape[0]):
                for j in range(ncol):
                    v = m[i, j]
                    if np.isnan(v) or v < 1:
                        continue
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if v > vmax * 0.55 else "#333333")

        for k in range(1, ncol):
            ax.axvline(k - 0.5, color="white",
                       lw=1.6 if k % len(configs) == 0 else 0.5)
        for k in range(1, len(rows)):
            ax.axhline(k - 0.5, color="white", lw=0.5)

        for j, v in enumerate(auprcs[dataset]):
            if not np.isnan(v):
                ax.text(j, -0.78, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="#555555")
        ax.text(-0.75, -0.78, "AUPRC", ha="right", va="center", fontsize=8,
                color="#555555", style="italic")
        ax.text(-0.75, -1.5, c.DATASET_TITLES[dataset], ha="left",
                va="center", fontsize=11.5)

        ax.set_xticks(range(ncol))
        ax.set_xticklabels([l for _ in c.MODELS for l in labels],
                           fontsize=10, rotation=90)

    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
    cbar.set_label(label, fontsize=10.5)
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.outline.set_visible(False)

    for k, model in enumerate(c.MODELS):
        axes[1].text(k * len(configs) + (len(configs) - 1) / 2,
                     len(rows) + 0.9, model, ha="center", va="top",
                     fontsize=10, fontweight="bold")

    fig.suptitle(TITLE, fontsize=13, fontweight="bold", y=0.965)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default="XGBoost",
                    choices=["XGBoost", "CatBoost", "LogisticRegression"])
    ap.add_argument("--groups", default="family", choices=["family", "signal"])
    ap.add_argument("--config", nargs="*", default=c.CONFIGS)
    ap.add_argument("--palette", default=None, choices=list(c.PALETTES))
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--no-annot", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    palette = args.palette or {"XGBoost": "teal", "CatBoost": "ember",
                               "LogisticRegression": "purple"}[args.classifier]
    label = {"XGBoost": "% of total gain",
             "CatBoost": "% of total importance",
             "LogisticRegression": r"% of total $|\beta|$"}[args.classifier]

    rows, matrices, auprcs = build(
        args.classifier, args.groups, args.config, args.top_n)
    fig = plot(rows, matrices, auprcs, args.config, palette,
               not args.no_annot, label)

    name = args.name or f"importance_{args.classifier.lower()}_{args.groups}"
    c.save(fig, name)


if __name__ == "__main__":
    main()
