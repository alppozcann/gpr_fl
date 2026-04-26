import torch
import gpytorch
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore", message="The input matches the stored training data")


def get_global_metrics(clients, threshold=None):
    """
    Compute global metrics by having each client predict on its OWN test data,
    then concatenating all predictions for unified metrics.
    
    This is the correct approach for clustered FL where each client's model
    is specialized for its cluster's data distribution.
    
    Args:
        clients: List of Client objects
        threshold: Optional threshold for binary classification
        
    Returns:
        dict with global accuracy, precision, recall, f1, roc_auc
    """
    all_y_true = []
    all_y_probs = []
    thresholds_used = []
    
    # Each client predicts on its own data
    for client in clients:
        client.model.eval()
        client.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-3):
            pred_dist = client.likelihood(client.model(client.test_x))
            y_pred_probs = pred_dist.probs.cpu().numpy().flatten()

            if threshold is None:
                train_dist = client.likelihood(client.model(client.train_x))
                train_probs = train_dist.probs.cpu().numpy().flatten()
                train_true = client.train_y.cpu().numpy().flatten().astype(int)
                client_threshold = find_optimal_threshold(train_true, train_probs)
            else:
                client_threshold = threshold

        y_true = client.test_y.cpu().numpy().flatten().astype(int)
        all_y_true.append(y_true)
        all_y_probs.append(y_pred_probs)
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
    y_pred_probs, _ = client.predict()
    y_true = client.test_y.cpu().numpy().flatten().astype(int)
    y_pred_probs = np.array(y_pred_probs).flatten()

    if threshold is None:
        # Find optimal threshold on training data to avoid overfitting to test set
        client.model.eval()
        client.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-3):
            train_dist = client.likelihood(client.model(client.train_x))
            train_probs = train_dist.probs.cpu().numpy().flatten()
        train_true = client.train_y.cpu().numpy().flatten().astype(int)
        threshold = find_optimal_threshold(train_true, train_probs)

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