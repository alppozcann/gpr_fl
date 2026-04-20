import pandas as pd
import os

from Client import Client, MIN_CLUSTER_SIZE
from Server import weighted_average_aggregation
from evaluation import get_metrics, get_global_metrics
from visualize import visualize_clients
from comparison import generate_all_comparisons, generate_paper_style_table
from clustering import get_medical_clusters, get_kmeans_clusters, MEDICAL_THRESHOLDS
from uncertainty_evaluation import analyze_uncertainty, generate_uncertainty_report

# =====================================================
# CONFIGURATION
# =====================================================
DATASET_NUMBER = 1
GP_TYPE = "sparse"       # kept for API compatibility — always uses SVGP classification
NUM_INDUCING_POINTS = 1000  # None = auto-scale (100/300/500 by dataset size); or set an int to override
KERNEL_TYPE = "matern"   # "rbf", "matern", "rbf_ard", "matern_ard"

NUM_FL_ROUNDS = 3
LOCAL_EPOCHS_PER_ROUND = 20
DATASET = f"dataset_{DATASET_NUMBER}/"
DATA_TABLE = f"diabetes_{DATASET_NUMBER}.csv"
csv_file = os.path.join(DATASET, DATA_TABLE)


#Features to test (for diabetes_3.csv)
if DATASET_NUMBER == 3: 
    features_to_test = ['HighBP', 'HighChol', 'BMI', 'HeartDiseaseorAttack', 'GenHlth', 'PhysHlth', 'DiffWalk', 'Age']

#Features to test 
else:
    _temp_df = pd.read_csv(csv_file)
    _target_columns = ["diabetes", "Outcome", "outcome", "Diabetes_012", "diabetes012", "Diabetes_binary"]
    features_to_test = [col for col in _temp_df.columns if col not in _target_columns]
    del _temp_df  # Clean up


# Output paths
output_file = os.path.join(DATASET, "results.txt")
paper_table_file = os.path.join(DATASET, "paper_style_results.txt")
plots_dir = os.path.join(DATASET, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Results storage
gp_fl_results = {}
gp_fl_local_results = {}
gp_fl_global_results = {}  # NEW: Store global metrics per feature
all_results_for_table = {}
all_uncertainty_results = {}  # NEW: Store uncertainty analysis results


# =====================================================
# EXPERIMENT PIPELINE
# =====================================================
def run_experiment(feature_name):
    """Run federated learning experiment for a given feature."""
    df_full = pd.read_csv(csv_file)
    
    # Get clusters
    if feature_name in MEDICAL_THRESHOLDS:
        result = get_medical_clusters(df_full, feature_name)
        if result[0] is not None:
            df_full, num_clusters, cluster_names = result
            clustering_method = "Medical Thresholds"
        else:
            df_full, num_clusters, cluster_names = get_kmeans_clusters(df_full, feature_name)
            clustering_method = "K-Means"
    else:
        df_full, num_clusters, cluster_names = get_kmeans_clusters(df_full, feature_name)
        clustering_method = "K-Means"
    
    # Create clients
    clients = []
    client_sizes = []
    valid_cluster_names = {}
    
    # First pass: compute expected columns from the full dataset for consistent dimensions
    df_for_cols = df_full.drop(columns=['cluster_id'])
    if 'diabetes' in df_for_cols.columns:
        df_for_cols = df_for_cols.drop(columns=['diabetes'])
    df_dummies = pd.get_dummies(df_for_cols, drop_first=True)
    expected_columns = list(df_dummies.columns)
    
    for i in range(num_clusters):
        cluster_df = df_full[df_full['cluster_id'] == i].drop(columns=['cluster_id'])
        if len(cluster_df) < MIN_CLUSTER_SIZE:
            continue
        temp_csv = f"temp_cluster_{i}.csv"
        cluster_df.to_csv(temp_csv, index=False)
        
        client = Client(client_id=i+1, csv_path=temp_csv, gp_type=GP_TYPE,
                       num_inducing_points=NUM_INDUCING_POINTS, expected_columns=expected_columns,
                       kernel_type=KERNEL_TYPE)
        if client.has_data:
            clients.append(client)
            client_sizes.append(len(client.train_x))
            valid_cluster_names[client.id] = cluster_names[i]
    
    if len(clients) < 2:
        for i in range(num_clusters):
            if os.path.exists(f"temp_cluster_{i}.csv"):
                os.remove(f"temp_cluster_{i}.csv")
        return f"\n{feature_name}: Not enough valid clients (need >= 2)\n"
    
    # Federated Learning — NUM_FL_ROUNDS rounds
    for fl_round in range(NUM_FL_ROUNDS):
        print(f"\n--- FL Round {fl_round + 1}/{NUM_FL_ROUNDS} ({feature_name}) ---")
        client_updates = []
        for client in clients:
            client.train_local(training_iter=LOCAL_EPOCHS_PER_ROUND)
            params = client.send_params()
            if params is not None:
                client_updates.append(params)

        global_params = weighted_average_aggregation(client_updates, client_sizes)
        for client in clients:
            client.set_params(global_params)
    
    # Cleanup temp files
    for i in range(num_clusters):
        if os.path.exists(f"temp_cluster_{i}.csv"):
            os.remove(f"temp_cluster_{i}.csv")
    
    # Generate report
    report = [f"\n{'='*60}\n",
              f"FEATURE: {feature_name.upper()} | CLUSTERS: {len(clients)} | METHOD: {clustering_method}\n",
              f"{'='*60}\n"]
    
    # Global metrics (NEW)
    global_m = get_global_metrics(clients)
    report.append(f"\n📊 GLOBAL MODEL METRICS (on all combined test data, n={global_m['total_samples']}):\n")
    report.append(f"  Metric      | Value    \n  ------------|----------\n")
    for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        report.append(f"  {key.capitalize():<11} | {global_m[key]:.4f}\n")
    report.append(f"{'='*60}\n")
    
    # Per-cluster metrics
    report.append(f"\n📋 PER-CLUSTER METRICS:\n")
    for client in clients:
        m = get_metrics(client)
        cluster_label = valid_cluster_names.get(client.id, f"Cluster {client.id}")
        report.append(f"Client {client.id} - {cluster_label} (n={len(client.train_x)+len(client.test_x)}):\n")
        report.append(f"  Metric      | Value    \n  ------------|----------\n")
        for key in m.keys():
            report.append(f"  {key.capitalize():<11} | {m[key]:.4f}\n")
        report.append(f"{'-'*40}\n")
    
    visualize_clients(clients, feature_name, plots_dir)
    
    # Store results
    gp_fl_results[feature_name] = []
    gp_fl_local_results[feature_name] = {}
    all_results_for_table[feature_name] = {}
    
    # Store global metrics (NEW)
    gp_fl_global_results[feature_name] = {
        'accuracy': global_m['accuracy'],
        'precision': global_m['precision'],
        'recall': global_m['recall'],
        'f1': global_m['f1'],
        'roc_auc': global_m['roc_auc'],
        'total_samples': global_m['total_samples']
    }
    
    for client in clients:
        m = get_metrics(client)
        cluster_label = valid_cluster_names.get(client.id, f"Cluster {client.id}")
        gp_fl_results[feature_name].append([m['accuracy'], m['precision'], m['recall'], m['f1']])
        gp_fl_local_results[feature_name][client.id] = [m['accuracy'], m['precision'], m['recall'], m['f1']]
        all_results_for_table[feature_name][cluster_label] = {"GP-FL": [m['accuracy'], m['precision'], m['recall'], m['f1']]}
    
    # Uncertainty analysis (GP's unique power!)
    uncertainty_results = analyze_uncertainty(clients, feature_name, plots_dir)
    all_uncertainty_results[feature_name] = uncertainty_results
        
    return "".join(report)


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Data Set: {csv_file}\n")
        
        for feature in features_to_test:
            print(f"🚀 {feature} starts")
            try:
                result_text = run_experiment(feature)
                f.write(result_text)
                f.flush()
                print(f"✅ {feature} completed.")
            except Exception as e:
                print(f"❌ Error {feature}: {e}")

    generate_paper_style_table(all_results_for_table, paper_table_file, gp_fl_global_results)
    generate_all_comparisons(gp_fl_results, gp_fl_local_results, plots_dir)
    
    # Generate uncertainty report (GP's unique power!)
    uncertainty_report_file = os.path.join(DATASET, "uncertainty_report.txt")
    generate_uncertainty_report(all_uncertainty_results, uncertainty_report_file)
