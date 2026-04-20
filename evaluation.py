import torch
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
        
        # Get predictions on test data
        with torch.no_grad():
            pred_dist = client.likelihood(client.model(client.test_x))
            y_pred_probs = pred_dist.probs.cpu().numpy().flatten()

        y_true = client.test_y.cpu().numpy().flatten().astype(int)

        if threshold is None:
            client_threshold = 0.5
        else:
            client_threshold = threshold
        y_pred_binary = (y_pred_probs > client_threshold).astype(int)

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
    return 0.5

def get_metrics(client, threshold=None):
    y_pred_probs, y_pred_hard = client.predict()
    y_true = client.test_y
    
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten().astype(int)
    
    if torch.is_tensor(y_pred_probs):
        y_pred_probs = y_pred_probs.cpu().numpy().flatten()
    
    threshold = 0.5
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