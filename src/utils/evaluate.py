import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba) -> PrettyTable:
    """
    Evaluate metrics
    """
    scores = PrettyTable()
    f1 = f1_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    scores.field_names = ["Precision", "Recall", "F1-score", "PR-AUC"]
    scores.add_row([f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{pr_auc:.4f}"])

    return scores
