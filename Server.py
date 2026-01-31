import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Clientlardan gelen parametreler (Senin log çıktın)
client_updates = {
    'client_1': np.array([0.14873328, 0.42246661]),
    'client_2': np.array([0.01563629, 1.69714153]),
    'client_3': np.array([6.90053139, -3.16840756])
}

def cluster_and_aggregate(updates_dict):
    # 1. Veriyi formata sok (List of arrays)
    ids = list(updates_dict.keys())
    data = np.array(list(updates_dict.values()))
    
    # 2. Kümeleme Yap (Distance Threshold mantığı)
    # distance_threshold=2.0 -> Eğer iki client arası uzaklık 2 birimden fazlaysa ayır.
    # n_clusters=None -> Kaç küme çıkacağına sen karar ver (otomatik).
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=2.0,  
        metric='euclidean', 
        linkage='ward'
    )
    
    labels = clustering.fit_predict(data)
    # labels çıktısı şöyle olacak: [0, 0, 1] -> Yani Client 1 ve 2 (Grup 0), Client 3 (Grup 1)
    
    # 3. Kümeleri Ayır ve Ortalamalarını Al
    clustered_models = {}
    unique_labels = set(labels)
    
    print(f"Oluşan Küme Yapısı: {labels}")
    
    for label in unique_labels:
        # Bu etikete sahip clientların indekslerini bul
        indices = [i for i, x in enumerate(labels) if x == label]
        group_ids = [ids[i] for i in indices]
        
        # O gruptaki parametreleri seç
        group_params = data[indices]
        
        # Ortalamasını al (FedAvg)
        new_global_params = np.mean(group_params, axis=0)
        
        clustered_models[label] = {
            "clients": group_ids,
            "params": new_global_params
        }
        
        print(f"Küme {label} ({group_ids}): Yeni Model Parametreleri -> {new_global_params}")

    return clustered_models
