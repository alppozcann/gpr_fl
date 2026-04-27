import numpy as np
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore", message="The input matches the stored training data")


def get_global_metrics(clients, threshold=None):
    """
    Compute global metrics by having each client predict on its OWN test data,
    then concatenating all predictions for unified metrics.

    Uses client.predict() (returns probs, hard_labels) and client.optimal_threshold
    (learned on validation set) instead of accessing model internals directly.

    Args:
        clients: List of Client objects
        threshold: Optional override threshold; if None uses each client's optimal_threshold
    """
    all_y_true = []
    all_y_probs = []
    thresholds_used = []

    for client in clients:
        probs, _ = client.predict()  # (probs, hard_labels) on client.test_x
        y_true = client.test_y.cpu().numpy().flatten().astype(int)
        client_threshold = threshold if threshold is not None else client.optimal_threshold

        all_y_true.append(y_true)
        all_y_probs.append(np.array(probs).flatten())
        thresholds_used.append(client_threshold)
    
    # Concatenate all results
    y_true = np.concatenate(all_y_true)
    y_pred_probs = np.concatenate(all_y_probs)
    y_pred_binary = np.concatenate([
        (all_y_probs[i] > thresholds_used[i]).astype(int) for i in range(len(all_y_probs))
    ])
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        roc_auc = 0.5
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "threshold": float(np.mean(thresholds_used)) if len(thresholds_used) > 0 else 0.5,
        "thresholds_used": thresholds_used,
        "total_samples": len(y_true)
    }


def find_optimal_threshold(y_true, y_probs):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (y_probs > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)

def get_metrics(client, threshold=None):
    y_pred_probs, _ = client.predict()  # probs = sigmoid(f(x)) on client.test_x
    y_true = client.test_y.cpu().numpy().flatten().astype(int)
    y_pred_probs = np.array(y_pred_probs).flatten()

    # Use the threshold already optimised on the validation set; allow explicit override
    if threshold is None:
        threshold = client.optimal_threshold

    y_pred_binary = (y_pred_probs > threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        roc_auc = 0.5

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "threshold": threshold
    }