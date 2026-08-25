"""
Evaluation plotting module.
Generates ROC curves, PR curves, Confusion Matrices, and Feature Importance charts.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_roc_comparison(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],  # name -> (y_true, y_prob)
    save_path: Path,
    title: str = "Receiver Operating Characteristic (ROC) Comparison",
):
    """Plot overlaid ROC curves for multiple models."""
    plt.figure(figsize=(7, 6), dpi=300)
    colors = ["#2b5c8f", "#2ca02c", "#d9534f", "#9467bd"]

    for i, (name, (y_true, y_prob)) in enumerate(models_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(y_true, y_prob)
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", color=color, linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_pr_comparison(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],  # name -> (y_true, y_prob)
    save_path: Path,
    title: str = "Precision-Recall (PR) Curve Comparison",
):
    """Plot overlaid Precision-Recall curves (critical for imbalanced data)."""
    plt.figure(figsize=(7, 6), dpi=300)
    colors = ["#2b5c8f", "#2ca02c", "#d9534f", "#9467bd"]

    for i, (name, (y_true, y_prob)) in enumerate(models_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        from sklearn.metrics import average_precision_score
        ap_score = average_precision_score(y_true, y_prob)
        color = colors[i % len(colors)]
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {ap_score:.3f})", color=color, linewidth=2)

    # Baseline positive rate
    first_y = next(iter(models_dict.values()))[0]
    base_rate = np.mean(first_y)
    plt.axhline(y=base_rate, color="k", linestyle="--", alpha=0.6, label=f"Baseline Pos Rate ({base_rate:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    save_path: Path,
):
    """Plot annotated confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Bankrupt (0)", "Bankrupt (1)"],
        yticklabels=["Non-Bankrupt (0)", "Bankrupt (1)"],
        cbar=False,
    )
    plt.title(f"{model_name} Confusion Matrix (Threshold={threshold:.2f})", fontsize=10, fontweight="bold", pad=10)
    plt.xlabel("Predicted Class", fontsize=9)
    plt.ylabel("True Class", fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
