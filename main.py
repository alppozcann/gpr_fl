import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

from Client import Client, MIN_CLUSTER_SIZE
from Server import weighted_average_aggregation
from evaluation import get_metrics
from visualize import visualize_clients
from comparison import generate_all_comparisons, generate_paper_style_table

# =====================================================
# CONFIGURATION
# =====================================================
NUM_FL_ROUNDS = 3
LOCAL_EPOCHS_PER_ROUND = 20
DATASET = "dataset_3/"
DATA_TABLE = "diabetes_3.csv"
csv_file = os.path.join(DATASET, DATA_TABLE) 

# Automatically get all features from the dataset (excluding target column)
_temp_df = pd.read_csv(csv_file)
_target_columns = ["diabetes", "Outcome", "outcome", "Diabetes_012", "diabetes012", "Diabetes_binary"]
features_to_test = [col for col in _temp_df.columns if col not in _target_columns]
del _temp_df  # Clean up

output_file = os.path.join(DATASET, "results.txt")
paper_table_file = os.path.join(DATASET, "paper_style_results.txt")
plots_dir = os.path.join(DATASET, "plots")
os.makedirs(plots_dir, exist_ok=True)

# GP Model Configuration
# "exact" for ExactGP 
# "sparse" for SparseGP 
GP_TYPE = "sparse"
NUM_INDUCING_POINTS = 100  # Only used if GP_TYPE = "sparse"

# =====================================================
# FEATURE NAME NORMALIZATION (case-insensitive matching)
# =====================================================
# Maps various column names to standardized feature names
FEATURE_ALIASES = {
    # BMI variants
    "bmi": "BMI", "Bmi": "BMI", "BMI": "BMI",
    # Glucose variants
    "glucose": "Glucose", "Glucose": "Glucose", 
    "blood_glucose_level": "Glucose", "blood_gluc": "Glucose",
    "BloodGlucose": "Glucose", "fasting_glucose": "Glucose",
    # HbA1c variants
    "hba1c_level": "HbA1c", "HbA1c_level": "HbA1c", "hba1c": "HbA1c", "HbA1c": "HbA1c",
    # Gender variants
    "gender": "Gender", "Gender": "Gender", "sex": "Gender", "Sex": "Gender",
    # Age variants
    "age": "Age", "Age": "Age",
    # Hypertension variants
    "hypertension": "Hypertension", "Hypertension": "Hypertension", 
    "HighBP": "Hypertension", "highbp": "Hypertension", "high_bp": "Hypertension",
    # Heart disease variants
    "heart_disease": "HeartDisease", "HeartDisease": "HeartDisease",
    "HeartDiseaseorAttack": "HeartDisease", "heartdisease": "HeartDisease",
    # Cholesterol variants
    "HighChol": "HighCholesterol", "highchol": "HighCholesterol", 
    "high_cholesterol": "HighCholesterol", "HighCholesterol": "HighCholesterol",
    # Difficulty walking
    "DiffWalk": "DiffWalk", "diffwalk": "DiffWalk", "difficulty_walking": "DiffWalk",
    # General health
    "GenHlth": "GenHealth", "genhlth": "GenHealth", "general_health": "GenHealth",
    # Physical health
    "PhysHlth": "PhysHealth", "physhlth": "PhysHealth", "physical_health": "PhysHealth",
    # Smoking
    "smoking_history": "Smoking", "Smoker": "Smoking", "smoker": "Smoking",
    # Blood pressure
    "BloodPressure": "BloodPressure", "bloodpressure": "BloodPressure", "blood_pressure": "BloodPressure",
    # Insulin
    "Insulin": "Insulin", "insulin": "Insulin",
    # Pregnancies
    "Pregnancies": "Pregnancies", "pregnancies": "Pregnancies",
    # Skin thickness
    "SkinThickness": "SkinThickness", "skinthickness": "SkinThickness", "skin_thickness": "SkinThickness",
    # Diabetes pedigree function
    "DiabetesPedigreeFunction": "DiabetesPedigree", "diabetespedigreefunction": "DiabetesPedigree",
}

def normalize_feature_name(feature_name):
    """Get standardized feature name for threshold lookup."""
    return FEATURE_ALIASES.get(feature_name, feature_name)

# =====================================================
# MEDICAL THRESHOLDS (based on paper methodology)
# =====================================================
# Each threshold config has: type, and type-specific parameters
MEDICAL_THRESHOLDS = {
    # BMI: 7 clusters (No Info, Underweight, Healthy, Overweight, Obesity I, II, III)
    "BMI": {
        "type": "continuous_with_underweight",
        "bins": [18.5, 25, 30, 35, 40, float('inf')],
        "underweight_threshold": 18.5,
        "no_info_condition": lambda x: (x == 0),
        "cluster_names": {
            0: "No Info",
            1: "Underweight (<18.5)",
            2: "Healthy (18.5-24.9)",
            3: "Overweight (25-29.9)",
            4: "Obesity I (30-34.9)",
            5: "Obesity II (35-39.9)",
            6: "Obesity III (>=40)"
        }
    },
    # Glucose: 4 clusters (No Info, Normal, Pre-diabetes, Diabetes)
    # Values < 70 or == 0 are "No Info", then: Normal < 100, Pre-diabetes 100-125, Diabetes >= 126
    "Glucose": {
        "type": "continuous",
        "bins": [100, 126, float('inf')],
        "no_info_condition": lambda x: (x < 70) | (x == 0),
        "cluster_names": {
            0: "No Info",
            1: "Normal (70-99)",
            2: "Pre-diabetes (100-125)",
            3: "Diabetes (>=126)"
        }
    },
    # HbA1c: 4 clusters (No Info, Normal, Pre-diabetes, Diabetes)
    "HbA1c": {
        "type": "continuous",
        "bins": [5.7, 6.5, float('inf')],
        "no_info_condition": lambda x: (x == 0) | (x < 3.0),  # Invalid HbA1c values
        "cluster_names": {
            0: "No Info",
            1: "Normal (<5.7)",
            2: "Pre-diabetes (5.7-6.4)",
            3: "Diabetes (>=6.5)"
        }
    },
    # Gender: 3 clusters (Male, Female, Other/Not specified)
    "Gender": {
        "type": "categorical",
        "categories": {
            "Male": ["Male", "male", "M", "m", 1, "1"],
            "Female": ["Female", "female", "F", "f", 0, "0", 2, "2"],
            "Other": ["Other", "other", "Not specified", "not specified", "No Info", -1, ""]
        },
        "cluster_names": {0: "Male", 1: "Female", 2: "Other/Not Specified"}
    },
    # Binary features: 2 clusters (No=0, Yes=1) + optional No Info
    "Hypertension": {
        "type": "binary",
        "cluster_names": {0: "No Hypertension", 1: "Hypertension"}
    },
    "HeartDisease": {
        "type": "binary",
        "cluster_names": {0: "No Heart Disease", 1: "Heart Disease"}
    },
    "HighCholesterol": {
        "type": "binary",
        "cluster_names": {0: "Normal Cholesterol", 1: "High Cholesterol"}
    },
    "DiffWalk": {
        "type": "binary",
        "cluster_names": {0: "No Difficulty Walking", 1: "Difficulty Walking"}
    },
    # General Health: 5 categories (1=Excellent to 5=Poor)
    "GenHealth": {
        "type": "ordinal",
        "cluster_names": {
            1: "Excellent",
            2: "Very Good", 
            3: "Good",
            4: "Fair",
            5: "Poor"
        }
    },
    # Smoking: Categorical
    "Smoking": {
        "type": "categorical_string",
        "cluster_names": {}  # Will be populated dynamically from unique values
    },
    # Blood Pressure: Medical ranges
    "BloodPressure": {
        "type": "continuous",
        "bins": [80, 90, 120, float('inf')],
        "no_info_condition": lambda x: (x == 0),
        "cluster_names": {
            0: "No Info",
            1: "Low (<80)",
            2: "Normal (80-89)",
            3: "Elevated (90-119)",
            4: "High (>=120)"
        }
    },
    # Insulin: Medical ranges
    "Insulin": {
        "type": "continuous",
        "bins": [16, 166, float('inf')],
        "no_info_condition": lambda x: (x == 0),
        "cluster_names": {
            0: "No Info",
            1: "Low (<16)",
            2: "Normal (16-165)",
            3: "High (>=166)"
        }
    },
}

# Store GP-FL results during experiments
gp_fl_results = {}
gp_fl_local_results = {}
all_results_for_table = {}  # For paper-style table: {feature: {cluster_name: {"GP-FL": metrics}}}


# =====================================================
# CLUSTERING FUNCTIONS
# =====================================================
def get_medical_clusters(df, feature_name):
    """
    Assign clusters based on predefined medical thresholds (matching paper methodology).
    Handles various threshold types: continuous, binary, categorical, ordinal.
    """
    # Normalize feature name for threshold lookup
    normalized_name = normalize_feature_name(feature_name)
    
    if normalized_name not in MEDICAL_THRESHOLDS:
        return None, None, None  # Signal to use K-means instead
    
    thresholds = MEDICAL_THRESHOLDS[normalized_name]
    threshold_type = thresholds["type"]
    df = df.copy()
    
    # Initialize cluster_id column
    df['cluster_id'] = -1
    
    if threshold_type == "continuous_with_underweight":
        # BMI-style: No Info, Underweight, then bins
        no_info_mask = thresholds["no_info_condition"](df[feature_name])
        underweight_mask = (df[feature_name] > 0) & (df[feature_name] < thresholds["underweight_threshold"])
        
        df.loc[no_info_mask, 'cluster_id'] = 0
        df.loc[underweight_mask, 'cluster_id'] = 1
        
        valid_mask = ~no_info_mask & ~underweight_mask
        if valid_mask.any():
            bins = thresholds["bins"]
            cut_result = pd.cut(df.loc[valid_mask, feature_name], bins=bins, 
                              labels=range(2, 2 + len(bins) - 1), right=False)
            df.loc[valid_mask, 'cluster_id'] = cut_result.astype(int)
        
        cluster_names = thresholds["cluster_names"]
        num_clusters = len(cluster_names)
        
    elif threshold_type == "continuous":
        # Glucose/HbA1c-style: No Info, then bins
        # Bins define thresholds: [70, 100, 126, inf] creates bins [<100], [100-125], [126+]
        no_info_mask = thresholds["no_info_condition"](df[feature_name])
        df.loc[no_info_mask, 'cluster_id'] = 0
        
        valid_mask = ~no_info_mask
        if valid_mask.any():
            # Create bins starting from -inf to capture all valid values
            bins_config = list(thresholds["bins"])
            # First bin edge should be lower than minimum valid value
            first_edge = bins_config[0]
            min_val = df.loc[valid_mask, feature_name].min()
            start_bin = min(min_val - 1, first_edge - 1) if first_edge != -float('inf') else -float('inf')
            bins = [start_bin] + bins_config
            
            # Labels start at 1 (0 is reserved for No Info)
            num_labels = len(bins_config)
            cut_result = pd.cut(df.loc[valid_mask, feature_name], bins=bins, 
                              labels=range(1, num_labels + 1), right=False)
            df.loc[valid_mask, 'cluster_id'] = cut_result.astype(int)
        
        cluster_names = thresholds["cluster_names"]
        num_clusters = len(cluster_names)
        
    elif threshold_type == "binary":
        # Binary features: 0 or 1
        df['cluster_id'] = df[feature_name].astype(int)
        cluster_names = thresholds["cluster_names"]
        num_clusters = 2
        
    elif threshold_type == "ordinal":
        # Ordinal features like GenHealth (1-5)
        df['cluster_id'] = df[feature_name].astype(int)
        cluster_names = thresholds["cluster_names"]
        num_clusters = len(cluster_names)
        
    elif threshold_type == "categorical":
        # Categorical with predefined mappings (Gender)
        categories = thresholds["categories"]
        for cluster_id, (cat_name, cat_values) in enumerate(categories.items()):
            mask = df[feature_name].isin(cat_values)
            df.loc[mask, 'cluster_id'] = cluster_id
        # Assign remaining to "Other"
        df.loc[df['cluster_id'] == -1, 'cluster_id'] = len(categories) - 1
        cluster_names = thresholds["cluster_names"]
        num_clusters = len(cluster_names)
        
    elif threshold_type == "categorical_string":
        # Categorical with dynamic mapping from unique values (Smoking)
        unique_vals = df[feature_name].unique()
        cluster_names = {i: str(val) for i, val in enumerate(unique_vals)}
        val_to_cluster = {val: i for i, val in enumerate(unique_vals)}
        df['cluster_id'] = df[feature_name].map(val_to_cluster)
        num_clusters = len(unique_vals)
    
    # Remove invalid clusters (-1) and reindex to consecutive IDs
    # IMPORTANT: Preserve the correct cluster_names mapping after reindexing
    unique_clusters = sorted(df['cluster_id'].unique())
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
        df = df[df['cluster_id'] != -1]
    
    # Create mapping from old cluster IDs to new consecutive IDs
    # Also map the cluster names correctly
    final_cluster_names = {}
    cluster_mapping = {}
    for new_id, old_id in enumerate(unique_clusters):
        cluster_mapping[old_id] = new_id
        # Use the cluster name for the OLD id, map it to the NEW id
        if old_id in cluster_names:
            final_cluster_names[new_id] = cluster_names[old_id]
        else:
            final_cluster_names[new_id] = f"Cluster {old_id}"
    
    df['cluster_id'] = df['cluster_id'].map(cluster_mapping)
    
    return df, len(unique_clusters), final_cluster_names


def get_kmeans_clusters(df, feature_name):
    """Assign clusters using K-means with silhouette score optimization."""
    X_cluster = df[[feature_name]].values
    
    best_score, best_k, best_labels = -1, 2, None
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_cluster)
        score = silhouette_score(X_cluster, labels)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
    
    df['cluster_id'] = best_labels
    
    cluster_names = {}
    for i in range(best_k):
        cluster_vals = df[df['cluster_id'] == i][feature_name]
        cluster_names[i] = f"Cluster {i} ({cluster_vals.min():.1f}-{cluster_vals.max():.1f})"
    
    return df, best_k, cluster_names


# =====================================================
# EXPERIMENT PIPELINE
# =====================================================
def run_experiment(feature_name):
    """Run federated learning experiment for a given feature."""
    df_full = pd.read_csv(csv_file)
    
    # Check if feature has medical thresholds (using normalized name)
    normalized_name = normalize_feature_name(feature_name)
    
    if normalized_name in MEDICAL_THRESHOLDS:
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
    
    clients = []
    client_sizes = []
    valid_cluster_names = {}
    
    # Create clients for each cluster
    for i in range(num_clusters):
        cluster_df = df_full[df_full['cluster_id'] == i].drop(columns=['cluster_id'])
        if len(cluster_df) < MIN_CLUSTER_SIZE:
            continue
        temp_csv = f"temp_cluster_{i}.csv"
        cluster_df.to_csv(temp_csv, index=False)
        
        client = Client(client_id=i+1, csv_path=temp_csv, gp_type=GP_TYPE, num_inducing_points=NUM_INDUCING_POINTS)
        if client.has_data:
            clients.append(client)
            client_sizes.append(len(client.train_x))
            valid_cluster_names[client.id] = cluster_names[i]
    
    if len(clients) < 2:
        for i in range(num_clusters):
            temp_csv = f"temp_cluster_{i}.csv"
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
        return f"\n{feature_name}: Not enough valid clients (need >= 2)\n"
    
    # Federated Learning: Single round - train locally, then aggregate
    client_updates = []
    for client in clients:
        client.train_local(training_iter=50)
        params = client.send_params()
        if params is not None:
            client_updates.append(params)
    
    global_params = weighted_average_aggregation(client_updates, client_sizes)
    
    for client in clients:
        client.set_params(global_params)
    
    # Clean up temp files
    for i in range(num_clusters):
        temp_csv = f"temp_cluster_{i}.csv"
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
    
    # Generate report
    report = []
    report.append(f"\n{'='*60}\n")
    report.append(f"FEATURE: {feature_name.upper()} | CLUSTERS: {len(clients)} | METHOD: {clustering_method}\n")
    report.append(f"{'='*60}\n")
    
    for client in clients:
        m_global = get_metrics(client)
        cluster_label = valid_cluster_names.get(client.id, f"Cluster {client.id}")
        
        report.append(f"Client {client.id} - {cluster_label} (n={len(client.train_x)+len(client.test_x)}):\n")
        report.append(f"  Metric      | Value    \n")
        report.append(f"  ------------|----------\n")
        for key in m_global.keys():
            report.append(f"  {key.capitalize():<11} | {m_global[key]:.4f}\n")
        report.append(f"{'-'*40}\n")
    
    visualize_clients(clients, feature_name, plots_dir)
    
    # Store results for comparison
    gp_fl_results[feature_name] = []
    gp_fl_local_results[feature_name] = {}
    all_results_for_table[feature_name] = {}
    
    for client in clients:
        m = get_metrics(client)
        cluster_label = valid_cluster_names.get(client.id, f"Cluster {client.id}")
        
        gp_fl_results[feature_name].append([m['accuracy'], m['precision'], m['recall'], m['f1']])
        gp_fl_local_results[feature_name][client.id] = [m['accuracy'], m['precision'], m['recall'], m['f1']]
        
        # Store for paper-style table
        all_results_for_table[feature_name][cluster_label] = {
            "GP-FL": [m['accuracy'], m['precision'], m['recall'], m['f1']]
        }
        
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
                print(f"{feature} completed.")
            except Exception as e:
                print(f" Error {feature} : {e}")

    # Generate paper-style table
    generate_paper_style_table(all_results_for_table, paper_table_file)
    
    # Generate comparison tables against paper baselines
    generate_all_comparisons(gp_fl_results, gp_fl_local_results, plots_dir)
