import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

def find_optimal_threshold(y_true, y_pred_continuous):
    """
    ROC eğrisinden Youden's J statistic kullanarak optimal threshold bulur.
    J = Sensitivity + Specificity - 1 = TPR - FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_continuous)
    
    # Youden's J statistic: TPR - FPR'yi maksimize et
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    
    # Eğer threshold inf veya çok büyükse, tahmin aralığının medyanını kullan
    if np.isinf(optimal_threshold) or optimal_threshold > y_pred_continuous.max():
        optimal_threshold = np.median(y_pred_continuous)
        print(f"  ⚠️ ROC optimal threshold geçersiz, medyan kullanılıyor: {optimal_threshold:.4f}")
    
    return optimal_threshold, fpr, tpr, thresholds, best_idx

def predict_with_dynamic_threshold(client):
    # Ham olasılıkları al
    y_pred_raw, _ = client.learner.predict(client.X_test)
    y_true = client.y_test
    
    # ROC eğrisinden optimal threshold bul (Youden's J statistic)
    optimal_threshold, _, _, _, _ = find_optimal_threshold(y_true, y_pred_raw)
    
    print(f"Client {client.id} için Optimal Threshold (ROC): {optimal_threshold:.4f}")
    print(f"  - Tahmin Aralığı: [{y_pred_raw.min():.4f}, {y_pred_raw.max():.4f}]")
    print(f"  - Tahmin Ortalaması: {y_pred_raw.mean():.4f}")
    
    # Optimal threshold ile tahminler
    y_pred_binary = (y_pred_raw >= optimal_threshold).astype(int)
    
    return y_pred_binary, optimal_threshold

def detailed_analysis(client, use_optimal_threshold=True):
    print(f"\n--- Client {client.id} Detaylı Performans Analizi ---")
    
    # 1. Tahminleri Al (Regression çıktısı: 0.1, 0.9 vb.)
    y_pred_continuous, sigma = client.learner.predict(client.X_test)
    y_true = client.y_test
    
    # 2. Optimal threshold kullan veya sabit 0.25
    if use_optimal_threshold:
        threshold, fpr, tpr, thresholds, best_idx = find_optimal_threshold(y_true, y_pred_continuous)
        print(f"🎯 Kullanılan Optimal Threshold: {threshold:.4f}")
    else:
        threshold = 0.25
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_continuous)
        best_idx = None
    
    y_pred_binary = (y_pred_continuous >= threshold).astype(int)
    
    # 3. TP, TN, FP, FN Hesapla
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    print(f"✅ True Positive (Doğru Teşhis - Hasta): {tp}")
    print(f"✅ True Negative (Doğru Teşhis - Sağlam): {tn}")
    print(f"❌ False Positive (Yanlış Alarm - Sağlam ama Hasta dedik): {fp}")
    print(f"❌ False Negative (Kaçırılan - Hasta ama Sağlam dedik): {fn}")
    
    # Recall ve Precision hesapla
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 Hasta Sınıfı için Metrikler:")
    print(f"  - Recall (Sensitivity): {recall:.2%} - Hastaların ne kadarını yakaladık")
    print(f"  - Precision: {precision:.2%} - Hasta dediklerimizin ne kadarı gerçekten hasta")
    print(f"  - F1-Score: {f1:.2%}")
    
    # Skor Raporu
    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(y_true, y_pred_binary, zero_division=0))

    # --- GRAFİK ÇİZİMİ ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Grafik 1: Confusion Matrix (Karmaşıklık Matrisi)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred_binary), 
                                      display_labels=["Sağlam (0)", "Diyabet (1)"])
    cm_display.plot(ax=ax[0], cmap='Blues', values_format='d')
    ax[0].set_title(f'Confusion Matrix (Client {client.id})\nThreshold: {threshold:.4f}')
    
    # Grafik 2: ROC Eğrisi (Modelin ayrım gücü)
    roc_auc = auc(fpr, tpr)
    
    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.50)')
    
    # Optimal threshold noktasını işaretle
    if use_optimal_threshold and best_idx is not None:
        ax[1].scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5, 
                     label=f'Optimal Point (thresh={threshold:.3f})')
    
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title(f'ROC Eğrisi (Client {client.id})')
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"client_{client.id}_result.png")
    plt.show()
    
    return {'threshold': threshold, 'recall': recall, 'precision': precision, 'f1': f1, 'auc': roc_auc}
