import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_metrics(client):
    """
    GPyTorch modelinin performansını ölçer ve sözlük olarak döndürür.
    """
    # 1. Tahminleri Al (Bizim yazdığımız predict fonksiyonu)
    # y_pred_raw: Olasılıklar (0.0 - 1.0 arası)
    # y_var: Varyans (Şimdilik lazım değil ama fonksiyon döndürüyor)
    y_pred_probs, _ = client.predict()
    
    # 2. Gerçek Değerleri (Test Y) Al
    y_true = client.test_y
    
    # Tensor ise Numpy'a çevir ve Düzleştir (Flatten)
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten().astype(int)
    
    if torch.is_tensor(y_pred_probs):
        y_pred_probs = y_pred_probs.cpu().numpy().flatten()
        
    # 3. Binary Tahmin (0 veya 1)
    # Standart olarak 0.5 eşik değeri kullanılır.
    # İstersen daha önce yazdığımız 'optimal_threshold'u da buraya parametre olarak verebilirsin.
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    
    # 4. Metrikleri Hesapla
    # zero_division=0: Eğer hiç pozitif tahmin yoksa hata verme, 0 bas.
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # ROC AUC için olasılıklar (y_pred_probs) kullanılır, binary (0/1) değil!
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        roc_auc = 0.5 # Tek sınıf varsa hata verebilir
        
    # Sonuçları Sözlük Olarak Döndür
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }