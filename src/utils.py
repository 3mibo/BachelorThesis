"""
utils.py – Reproducibility, plotting and result aggregation helpers.
"""

import os
import random
import json
import csv
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(
    loss_histories: List[List[float]],
    output_path: str,
    title: str = "Training loss per fold",
) -> None:
    """
    Save a line plot of per-fold training-loss curves to *output_path*.

    *loss_histories*: one list of floats per fold (each float is the average
    loss over one gradient-accumulation window).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, history in enumerate(loss_histories):
        ax.plot(history, label=f"Fold {i+1}")
    ax.set_xlabel("Grad-accum step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prompt_generalisation(
    fold_results: List[Dict[str, Dict]],
    labels: List[str],
    output_path: str,
    title: str = "UAR per prompt (mean ± std across folds)",
) -> None:
    """
    Bar chart showing mean ± std UAR across folds for each evaluated prompt.

    *fold_results*: list (one entry per fold) of dicts mapping prompt_name → metrics.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not fold_results:
        return

    prompt_names = list(fold_results[0].keys())
    means, stds = [], []
    for pname in prompt_names:
        uars = [fold[pname]["uar"] for fold in fold_results if pname in fold]
        means.append(np.mean(uars))
        stds.append(np.std(uars))

    x = np.arange(len(prompt_names))
    fig, ax = plt.subplots(figsize=(max(6, len(prompt_names) * 1.5), 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_names, rotation=30, ha="right")
    ax.set_ylabel("UAR")
    ax.set_ylim(0, 1)
    ax.set_title(title)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Save a confusion-matrix heatmap to *output_path*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    n = len(labels)
    fig, ax = plt.subplots(figsize=(n + 2, n + 1))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            text = format(val, fmt)
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def aggregate_fold_results(
    fold_results: List[Dict[str, Dict]],
) -> Dict[str, Dict]:
    """
    Aggregate per-fold evaluation results across folds.

    Returns a dict mapping prompt_name → {
        mean_uar, std_uar, mean_acc, std_acc, mean_wf1, std_wf1,
        per_fold_uar, per_fold_acc, per_fold_wf1
    }.
    """
    if not fold_results:
        return {}

    prompt_names = list(fold_results[0].keys())
    summary = {}
    for pname in prompt_names:
        uars = [f[pname]["uar"] for f in fold_results if pname in f]
        accs = [f[pname]["acc"] for f in fold_results if pname in f]
        wf1s = [f[pname]["wf1"] for f in fold_results if pname in f]
        summary[pname] = {
            "mean_uar": float(np.mean(uars)),
            "std_uar":  float(np.std(uars)),
            "mean_acc": float(np.mean(accs)),
            "std_acc":  float(np.std(accs)),
            "mean_wf1": float(np.mean(wf1s)),
            "std_wf1":  float(np.std(wf1s)),
            "per_fold_uar": [float(v) for v in uars],
            "per_fold_acc": [float(v) for v in accs],
            "per_fold_wf1": [float(v) for v in wf1s],
        }
    return summary


def save_results_json(results: Dict, output_path: str) -> None:
    """Serialize *results* to a JSON file, converting numpy arrays to lists."""
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_convert)


def save_summary_csv(summary: Dict[str, Dict], output_path: str) -> None:
    """
    Write a CSV with one row per prompt showing mean ± std UAR / ACC / WF1.
    """
    fieldnames = ["prompt", "mean_uar", "std_uar", "mean_acc", "std_acc", "mean_wf1", "std_wf1"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prompt_name, metrics in summary.items():
            writer.writerow({
                "prompt":    prompt_name,
                "mean_uar":  f"{metrics['mean_uar']:.4f}",
                "std_uar":   f"{metrics['std_uar']:.4f}",
                "mean_acc":  f"{metrics['mean_acc']:.4f}",
                "std_acc":   f"{metrics['std_acc']:.4f}",
                "mean_wf1":  f"{metrics['mean_wf1']:.4f}",
                "std_wf1":   f"{metrics['std_wf1']:.4f}",
            })
