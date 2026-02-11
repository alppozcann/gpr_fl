from Client_2 import Client

from Server import cluster_and_aggregate
import numpy as np
from evalutation import detailed_analysis
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import gpytorch

clients = [] 
client_updates = {}
for i in range(3):
    clients.append(Client(i+1,f"diabetes_{i+1}.csv"))

for client in clients:
    client.train_local()
    client_updates[f"client_{client.id}"] = client.send_params()

clustered_results = cluster_and_aggregate(client_updates)

print(f"Clustered Results are : {clustered_results}")

for cluster_label, cluster_info in clustered_results.items():
    new_params = cluster_info['params']
    client_list_names = cluster_info['clients']

    for client_name in client_list_names:
        c_id = int(client_name.split('_')[1])

        for client in clients:
            if client.id == c_id:
                client.set_params(new_params)

                print(f"Client {c_id} parametres updated")
                print(f"New params for Client {c_id} : {client.get_params()}")

results = {}
for client in clients:
    results[client.id] = detailed_analysis(client)

print("\n" + "="*60)
print("📋 TÜM CLIENT'LAR İÇİN ÖZET")
print("="*60)
print(f"{'Client':<10} {'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10} {'AUC':<10}")
print("-"*60)
for cid, r in results.items():
    print(f"Client {cid:<3} {r['threshold']:<12.4f} {r['recall']:<10.2%} {r['precision']:<12.2%} {r['f1']:<10.2%} {r['auc']:<10.2f}")

print("\n" + "="*60)
print("📋 TÜM CLIENT'LAR İÇİN TEST")

for client in clients:
    client.test_global_model()

for cid, r in results.items():

    cm = np.array([[r['tn'], r['fp']], 
                   [r['fn'], r['tp']]])
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=["Non-Diabetic (0)", "Diabetic (1)"])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    plt.title(f"Confusion Matrix for Client {cid}")
    
    filename = f"client_{cid}_confusion_matrix.png"
    plt.savefig(filename, dpi=300) 
    plt.close()