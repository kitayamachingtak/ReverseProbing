#AUPRC across base LLMs, comparing Reverse Probing against four baselines.

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from draw_image import common as c

DEFAULT_METHODS = ["SE Probes", "Sliding Window Entropy",
                   "Decomposing Uncertainty", "Semantic Entropy"]
OURS = "Reverse Probing"


def reverse_probing_auprc():
    out = {}
    for dataset in c.DATASETS:
        out[dataset] = {}
        for model in c.MODELS:
            metrics = c.load_metrics(model, "max", dataset)
            out[dataset][model] = (
                max(r["auprc"] for r in metrics.values()) if metrics else np.nan)
    return out


def build(methods):
    baselines = c.load_baselines()
    ours = reverse_probing_auprc()

    data = {}
    for method in methods:
        data[method] = {
            ds: [baselines.get(method, {}).get(ds, {}).get(m, np.nan)
                 for m in c.MODELS]
            for ds in c.DATASETS
        }
    data[OURS] = {ds: [ours[ds][m] for m in c.MODELS] for ds in c.DATASETS}
    return data


def plot(data, methods):
    all_methods = methods + [OURS]
    n_methods, n_models = len(all_methods), len(c.MODELS)
    bar_width = 0.13
    centers = np.arange(n_methods)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width
    colors = [c.MODEL_COLORS[m] for m in c.MODELS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, dataset in zip(axes, c.DATASETS):
        for mi, method in enumerate(all_methods):
            values = data[method][dataset]
            ours = method == OURS
            for li, value in enumerate(values):
                if np.isnan(value):
                    continue
                ax.bar(centers[mi] + offsets[li], value,
                       width=bar_width * 0.92, color=colors[li],
                       edgecolor="#8B0000" if ours else "white",
                       linewidth=1.2 if ours else 0.3, zorder=3)

        ax.set_xticks(centers)
        ax.set_xticklabels(all_methods, fontsize=15, rotation=15, ha="right")
        ax.set_title(c.DATASET_TITLES[dataset], fontsize=19,
                     fontweight="bold", pad=8)
        ax.set_ylabel("AUPRC", fontsize=19)
        ax.tick_params(axis="y", labelsize=15)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.55, centers[-1] + 0.55)
        for i in range(n_methods - 1):
            ax.axvline((centers[i] + centers[i + 1]) / 2, color="gray",
                       linewidth=0.5, linestyle=":", alpha=0.5)

    handles = [mpatches.Patch(facecolor=colors[i], edgecolor="white",
                              linewidth=0.5, label=c.MODELS[i])
               for i in range(n_models)]
    fig.legend(handles=handles, fontsize=15, frameon=False,
               loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("AUPRC across base LLMs", fontsize=21,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--name", default="auprc_bars")
    args = ap.parse_args()

    fig = plot(build(args.methods), args.methods)
    c.save(fig, args.name)


if __name__ == "__main__":
    main()