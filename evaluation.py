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
    all_train_y = []
    all_train_probs = []
    
    # Each client predicts on its own data
    for client in clients:
        client.model.eval()
        client.likelihood.eval()
        
        # Get predictions on test data
        with torch.no_grad():
            pred_dist = client.likelihood(client.model(client.test_x))
            y_pred_probs = pred_dist.mean.cpu().numpy().flatten()
        
        y_true = client.test_y.cpu().numpy().flatten().astype(int)
        all_y_true.append(y_true)
        all_y_probs.append(y_pred_probs)
        
        # Get predictions on training data for threshold optimization
        with torch.no_grad():
            train_dist = client.likelihood(client.model(client.train_x))
            train_probs = train_dist.mean.cpu().numpy().flatten()
        
        train_y = client.train_y.cpu().numpy().flatten().astype(int)
        all_train_y.append(train_y)
        all_train_probs.append(train_probs)
    
    # Concatenate all results
    y_true = np.concatenate(all_y_true)
    y_pred_probs = np.concatenate(all_y_probs)
    
    # Find optimal threshold if not provided
    if threshold is None:
        train_y = np.concatenate(all_train_y)
        train_probs = np.concatenate(all_train_probs)
        threshold = find_optimal_threshold(train_y, train_probs)
    
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
        "threshold": threshold,
        "total_samples": len(y_true)
    }


def find_optimal_threshold(y_true, y_probs):
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_probs > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

def get_metrics(client, threshold=None):
    y_pred_probs, _ = client.predict()
    y_true = client.test_y
    
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten().astype(int)
    
    if torch.is_tensor(y_pred_probs):
        y_pred_probs = y_pred_probs.cpu().numpy().flatten()
    
    if threshold is None:
        train_probs, _ = client.predict(client.train_x)
        train_y = client.train_y.cpu().numpy().flatten().astype(int)
        threshold = find_optimal_threshold(train_y, train_probs.flatten())
        
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