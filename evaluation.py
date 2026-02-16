import torch
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore", message="The input matches the stored training data")

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