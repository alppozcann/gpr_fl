from Client import Client

from Server import cluster_and_aggregate

from evalutation import detailed_analysis,predict_with_dynamic_threshold

clients = [] 
client_updates = {}
for i in range(3):
    clients.append(Client(i+1,f"diabetes_{i+1}.csv"))

for client in clients:
    client.train_local()
    client_updates[f"client_{client.id}"] = client.send_params()

clustered_results = cluster_and_aggregate(client_updates)


for cluster_label, cluster_info in clustered_results.items():
    new_params = cluster_info['params']
    client_list_names = cluster_info['clients']

    for client_name in client_list_names:
        c_id = int(client_name.split('_')[1])

        for client in clients:
            if client.id == c_id:
                client.set_params(new_params)

                print(f"Client {c_id} parametres updated")

print("\n" + "="*60)
print("📊 SONUÇ ANALİZİ (Optimal Threshold ile)")
print("="*60)

results = {}
for client in clients:
    results[client.id] = detailed_analysis(client)

# Özet tablo
print("\n" + "="*60)
print("📋 TÜM CLIENT'LAR İÇİN ÖZET")
print("="*60)
print(f"{'Client':<10} {'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10} {'AUC':<10}")
print("-"*60)
for cid, r in results.items():
    print(f"Client {cid:<3} {r['threshold']:<12.4f} {r['recall']:<10.2%} {r['precision']:<12.2%} {r['f1']:<10.2%} {r['auc']:<10.2f}")
