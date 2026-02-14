import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from Client_2 import Client
from Server import weighted_average_aggregation
from evaluation import get_metrics # Bu fonksiyonun evaluation.py'de olduğundan emin ol

# --- KONFİGÜRASYON ---
csv_file = "diabetes_2.csv" # Veya "diabetes_2.csv"
features_to_test = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"] # Denemek istediğin sütunlar
output_file = "deney_sonuclari.txt"

def run_experiment(feature_name):
    # 1. Veriyi Yükle
    df_full = pd.read_csv(csv_file)
    X_cluster = df_full[[feature_name]].values
    
    # 2. Optimal K Bulma (Silhouette)
    best_score, best_k, best_labels = -1, 2, None
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_cluster)
        score = silhouette_score(X_cluster, labels)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
            
    # 3. Clientları Oluştur ve Eğit
    df_full['cluster_id'] = best_labels
    clients = []
    client_updates = []
    client_sizes = []
    
    for i in range(best_k):
        temp_csv = f"temp_cluster_{i}.csv"
        df_full[df_full['cluster_id'] == i].drop(columns=['cluster_id']).to_csv(temp_csv, index=False)
        
        client = Client(client_id=i+1, csv_path=temp_csv)
        client.train_local(training_iter=50)
        
        params = client.send_params()
        if params is not None:
            clients.append(client)
            client_updates.append(params)
            client_sizes.append(len(client.train_x))

    # 4. Aggregation (Global Model)
    global_params = weighted_average_aggregation(client_updates, client_sizes)
    
    # 5. Sonuçları Topla (Metin olarak döndür)
    report = []
    report.append(f"\n{'='*50}\n")
    report.append(f"FEATURE: {feature_name.upper()} | OPTIMAL K: {best_k} (Score: {best_score:.4f})\n")
    report.append(f"{'='*50}\n")
    
    for client in clients:
        # Local Ölçüm
        m_local = get_metrics(client)
        # Global Ölçüm
        client.set_params(global_params)
        m_global = get_metrics(client)
        
        report.append(f"Client {client.id} Results:\n")
        report.append(f"  Metric      | Local    | Global   \n")
        report.append(f"  ------------|----------|----------\n")
        for key in m_local.keys():
            report.append(f"  {key.capitalize():<11} | {m_local[key]:.4f}   | {m_global[key]:.4f}\n")
        report.append(f"{'-'*40}\n")
        
    return "".join(report)

# --- ANA DÖNGÜ ---
with open(output_file, "w", encoding="utf-8") as f:
    f.write("🧪 FEDERATED LEARNING CLUSTERING DENEY RAPORU\n")
    f.write(f"Veri Seti: {csv_file}\n")
    
    for feature in features_to_test:
        print(f"🚀 {feature} deneyi başlatılıyor...")
        try:
            result_text = run_experiment(feature)
            f.write(result_text)
            f.flush() # Hemen dosyaya yaz
            print(f"✅ {feature} tamamlandı.")
        except Exception as e:
            print(f"❌ {feature} başarısız: {e}")

print(f"\n🎉 Tüm deneyler bitti! Sonuçlar '{output_file}' dosyasına yazıldı.")