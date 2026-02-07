import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support

def detailed_analysis(client):
    """
    GPyTorch Client'ı için detaylı analiz yapar.
    Veri tiplerini ve boyutlarını sklearn için zorla düzeltir.
    """
    
    # 1. Tahmin Al (GPyTorch'tan)
    # y_pred_raw bir tensör veya numpy array olabilir, garantileyeceğiz
    y_pred_raw, y_var = client.predict()
    
    # 2. Gerçek Değerleri (Test Y) Al ve Dönüştür
    y_true = client.test_y
    
    # Eğer Tensor ise Numpy'a çevir (CPU'ya alarak)
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    
    if torch.is_tensor(y_pred_raw):
        y_pred_raw = y_pred_raw.cpu().numpy()
        
    # --- KRİTİK DÜZELTME: FLATTEN (DÜZLEŞTİRME) ---
    # [[0], [1]] şeklindeki (N, 1) matrisi [0, 1] şeklindeki (N,) dizisine çevirir.
    y_true = y_true.flatten()
    y_pred_raw = y_pred_raw.flatten()
    
    # y_true'nun kesinlikle Tamsayı olduğundan emin ol
    y_true = y_true.astype(int)

    # Çok sınıflıysa ikiliye çevir (0: negatif, diğerleri: pozitif)
    unique_labels = np.unique(y_true)
    if unique_labels.size > 2:
        y_true = (y_true != 0).astype(int)
    else:
        # {0,1} dışında iki sınıf varsa yeniden eşle
        if not np.array_equal(unique_labels, np.array([0, 1])):
            y_true = (y_true == unique_labels.max()).astype(int)
    
    # 3. Optimal Threshold Bulma
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_raw)
    
    if len(thresholds) > 0:
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5
    
    # 4. Binary Tahmin
    y_pred_binary = (y_pred_raw > optimal_threshold).astype(int)
    
    # 5. Metrikler
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average='binary', zero_division=0)
    roc_auc = auc(fpr, tpr)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    return {
        'threshold': optimal_threshold,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auc': roc_auc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

# main.py uyumluluğu için boş fonksiyon
def predict_with_dynamic_threshold(client):
    pass
