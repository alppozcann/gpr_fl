import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from Client_2 import Client
from Server import weighted_average_aggregation  # Server'ı değiştireceğiz
from evaluation import get_metrics # Senin analiz kodun

# --- AYARLAR ---
csv_path = "diabetes_2.csv"
feature_to_cluster = "Pregnancies"  # Hoca BMI, Age, Glucose vb. istemişti
device = "cpu" # GPU hatası almamak için

# 1. Veriyi Yükle ve Hazırla
print(f"📂 Veri Yükleniyor: {csv_path}")
df = pd.read_csv(csv_path)

# 2. Optimal Küme Sayısını Bul (Makale Bölüm 4: Silhouette Score) [cite: 450]
print(f"\n🔍 '{feature_to_cluster}' özelliği için K-Means ve Silhouette Analizi yapılıyor...")

X_cluster = df[[feature_to_cluster]].values
best_score = -1
best_k = 2
best_labels = None

# 2'den 5'e kadar küme sayılarını dene
for k in range(2, 3):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster)
    score = silhouette_score(X_cluster, labels)
    print(f"   k={k} -> Silhouette Score: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

print(f"✅ En iyi küme sayısı (k): {best_k} (Skor: {best_score:.4f})")

# 3. Veriyi Parçala ve Geçici Dosyalara Kaydet
df['cluster_id'] = best_labels
clients = []

print("\n📦 Veri Kümeleri Client'lara Dağıtılıyor...")
for i in range(best_k):
    # O kümeye ait veriyi çek
    cluster_data = df[df['cluster_id'] == i].drop(columns=['cluster_id'])
    
    # Dosyaya kaydet (Client okuyabilsin diye)
    temp_file = f"cluster_{feature_to_cluster}_{i}.csv"
    cluster_data.to_csv(temp_file, index=False)
    
    # Client Oluştur
    print(f"   -> Client {i+1} oluşturuldu: {feature_to_cluster} Grubu {i} (Veri Sayısı: {len(cluster_data)})")
    client = Client(client_id=i+1, csv_path=temp_file)
    clients.append(client)

# 4. Eğitim Döngüsü
client_updates = []
client_sizes = []

print("\n🔄 Federe Eğitim Başlıyor...")

for client in clients:
    # Her client kendi "uzmanlık alanında" (cluster) eğitiliyor [cite: 390]
    client.train_local(training_iter=50)
    
    # Parametreleri ve Veri Sayısını al (Ağırlıklı ortalama için lazım)
    params = client.send_params()
    if params is not None:
        client_updates.append(params)
        client_sizes.append(len(client.train_x)) # Veri sayısı

# 5. Global Model Oluşturma (Aggregation)
# Makale: "weighted the cluster-specific coefficients according to sample sizes" 
print("\n🔗 Global Model Birleştiriliyor (Weighted Average)...")
global_params = weighted_average_aggregation(client_updates, client_sizes)

# 6. Sonuçları Dağıt ve Test Et
# ... (Önceki kodların aynı kalsın) ...

print("\n" + "="*60)
print("📊 DETAYLI SONUÇ RAPORU")
print("="*60)

for client in clients:
    # 1. Yerel Model (Specialized) Sonuçları
    print(f"\n🔹 Client {client.id} (Yerel / Specialized Model)")
    # Not: Parametreleri set etmeden önce ölçüm yapmalısın veya parametreleri saklayıp geri yüklemelisin.
    # Ancak akış gereği şu an model zaten yerel eğitilmiş durumda.
    metrics_local = get_metrics(client)
    
    print(f"   {'Metric':<15} {'Score':<10}")
    print(f"   {'-'*25}")
    for key, value in metrics_local.items():
        print(f"   {key.capitalize():<15} {value:.4f}")
    
    # 2. Global Model (Aggregated) Sonuçları
    # Server'dan gelen ortalama parametreleri yüklüyoruz
    client.set_params(global_params)
    
    print(f"\n🔸 Client {client.id} (Global / Aggregated Model)")
    metrics_global = get_metrics(client)
    
    print(f"   {'Metric':<15} {'Score':<10}")
    print(f"   {'-'*25}")
    for key, value in metrics_global.items():
        print(f"   {key.capitalize():<15} {value:.4f}")
        
    print("-" * 60)

# Temizlik (Geçici dosyaları sil)
for i in range(best_k): os.remove(f"cluster_{feature_to_cluster}_{i}.csv")