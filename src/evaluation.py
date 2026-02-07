import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_matching(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_per_class": dict(zip(
            labels,
            precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        )),
        "recall_per_class": dict(zip(
            labels,
            recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        )),
        "f1_per_class": dict(zip(
            labels,
            f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        )),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }

    return results


def evaluate_po_matching(
    predicted_matches: List[Optional[str]],
    actual_matches: List[Optional[str]]
) -> Dict[str, float]:
    if len(predicted_matches) != len(actual_matches):
        raise ValueError("Prediction and actual lists must have same length")

    n = len(predicted_matches)
    if n == 0:
        return {
            "matching_accuracy": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "true_positive_rate": 0.0,
        }

    tp = sum(
        1 for p, a in zip(predicted_matches, actual_matches)
        if p is not None and a is not None and p == a
    )

    tn = sum(
        1 for p, a in zip(predicted_matches, actual_matches)
        if p is None and a is None
    )

    fp = sum(
        1 for p, a in zip(predicted_matches, actual_matches)
        if p is not None and (a is None or p != a)
    )

    fn = sum(
        1 for p, a in zip(predicted_matches, actual_matches)
        if p is None and a is not None
    )

    matching_accuracy = (tp + tn) / n if n > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "matching_accuracy": matching_accuracy,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positive_rate": true_positive_rate,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total_samples": n,
    }


def cross_validation_evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: Optional[List[str]] = None
) -> Dict[str, Any]:
    if scoring is None:
        scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for metric in scoring:
        try:
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=metric)
            results[metric] = {
                "scores": scores.tolist(),
                "mean": scores.mean(),
                "std": scores.std(),
            }
        except Exception as e:
            logger.warning(f"Could not compute {metric}: {e}")
            results[metric] = {
                "scores": [],
                "mean": 0.0,
                "std": 0.0,
                "error": str(e),
            }

    return results


def evaluate_variance_detection(
    predicted_amounts: List[float],
    actual_amounts: List[float],
    tolerance_pct: float = 5.0
) -> Dict[str, float]:
    if len(predicted_amounts) != len(actual_amounts):
        raise ValueError("Prediction and actual lists must have same length")

    pred = np.array(predicted_amounts)
    actual = np.array(actual_amounts)

    mae = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual) ** 2))

    with np.errstate(divide='ignore', invalid='ignore'):
        pct_errors = np.where(
            actual != 0,
            np.abs((pred - actual) / actual) * 100,
            np.where(pred == 0, 0, 100)
        )

    within_tolerance = np.mean(pct_errors <= tolerance_pct)

    return {
        "mae": mae,
        "rmse": rmse,
        "accuracy_within_tolerance": within_tolerance,
        "tolerance_pct": tolerance_pct,
        "mean_pct_error": np.mean(pct_errors[~np.isinf(pct_errors)]) if len(pct_errors) > 0 else 0,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=figsize)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    sns.heatmap(
        cm_normalized,
        annot=cm,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Normalized Frequency'}
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if y_scores.ndim == 1:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ax.plot(recall, precision, linewidth=2, label='Model')
    else:
        n_classes = y_scores.shape[1]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for i, (name, color) in enumerate(zip(class_names, colors)):
            y_true_binary = (y_true == i).astype(int)
            if y_true_binary.sum() == 0:
                continue

            precision, recall, _ = precision_recall_curve(y_true_binary, y_scores[:, i])
            ax.plot(recall, precision, color=color, linewidth=2, label=name)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Precision-recall curve saved to {save_path}")

    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: List[float],
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_importances, align='center', color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")

    return fig


def plot_class_distribution(
    labels: List[str],
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    label_counts = pd.Series(labels).value_counts()

    colors = sns.color_palette("husl", len(label_counts))
    bars = ax.bar(label_counts.index, label_counts.values, color=colors)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)

    for bar, count in zip(bars, label_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")

    return fig


def generate_evaluation_report(
    y_true: List[str],
    y_pred: List[str],
    variance_true: Optional[List[float]] = None,
    variance_pred: Optional[List[float]] = None,
    model_name: str = "InvoicePOMatcher"
) -> str:
    metrics = evaluate_matching(y_true, y_pred)

    report_lines = [
        "=" * 60,
        f"EVALUATION REPORT: {model_name}",
        "=" * 60,
        "",
        "OVERALL METRICS",
        "-" * 40,
        f"Accuracy:           {metrics['accuracy']:.4f}",
        f"Precision (macro):  {metrics['precision_macro']:.4f}",
        f"Recall (macro):     {metrics['recall_macro']:.4f}",
        f"F1 Score (macro):   {metrics['f1_macro']:.4f}",
        "",
        f"Precision (weighted): {metrics['precision_weighted']:.4f}",
        f"Recall (weighted):    {metrics['recall_weighted']:.4f}",
        f"F1 Score (weighted):  {metrics['f1_weighted']:.4f}",
        "",
        "PER-CLASS METRICS",
        "-" * 40,
    ]

    for label in metrics['labels']:
        p = metrics['precision_per_class'].get(label, 0)
        r = metrics['recall_per_class'].get(label, 0)
        f = metrics['f1_per_class'].get(label, 0)
        report_lines.append(f"{label}:")
        report_lines.append(f"  Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")

    report_lines.extend([
        "",
        "CONFUSION MATRIX",
        "-" * 40,
    ])

    cm = metrics['confusion_matrix']
    labels = metrics['labels']

    header = "Predicted -> " + " | ".join(f"{l[:8]:>8}" for l in labels)
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for i, label in enumerate(labels):
        row_str = f"{label[:10]:<10} " + " | ".join(f"{cm[i][j]:>8}" for j in range(len(labels)))
        report_lines.append(row_str)

    if variance_true is not None and variance_pred is not None:
        var_metrics = evaluate_variance_detection(variance_pred, variance_true)
        report_lines.extend([
            "",
            "VARIANCE DETECTION",
            "-" * 40,
            f"Mean Absolute Error:    ${var_metrics['mae']:,.2f}",
            f"Root Mean Square Error: ${var_metrics['rmse']:,.2f}",
            f"Within {var_metrics['tolerance_pct']}% Tolerance: {var_metrics['accuracy_within_tolerance']:.2%}",
        ])

    report_lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(report_lines)


if __name__ == "__main__":
    print("Evaluation Module Demo")
    print("=" * 50)

    y_true = ["EXACT_MATCH", "PRICE_VARIANCE", "QUANTITY_VARIANCE", "TAX_MISCODE",
              "EXACT_MATCH", "PRICE_VARIANCE", "MISSING_PO", "EXACT_MATCH"]
    y_pred = ["EXACT_MATCH", "PRICE_VARIANCE", "PRICE_VARIANCE", "TAX_MISCODE",
              "EXACT_MATCH", "QUANTITY_VARIANCE", "MISSING_PO", "EXACT_MATCH"]

    metrics = evaluate_matching(y_true, y_pred)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")

    print("\nFull Report:")
    report = generate_evaluation_report(y_true, y_pred)
    print(report)

    variance_true = [100.50, 250.00, 0.00, 50.25]
    variance_pred = [102.30, 245.00, 10.00, 51.00]
    var_metrics = evaluate_variance_detection(variance_pred, variance_true)
    print(f"\nVariance MAE: ${var_metrics['mae']:,.2f}")
    print(f"Variance RMSE: ${var_metrics['rmse']:,.2f}")
