"""
Clustering module for medical feature-based clustering.
Contains medical thresholds and clustering functions.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =====================================================
# MEDICAL THRESHOLDS (based on paper methodology)
# =====================================================
MEDICAL_THRESHOLDS = {
    # ===== DATASET 3 (BRFSS) =====
    "HighBP": {"type": "binary", "cluster_names": {0: "No High BP", 1: "High BP"}},
    "HighChol": {"type": "binary", "cluster_names": {0: "Normal Cholesterol", 1: "High Cholesterol"}},
    "HeartDiseaseorAttack": {"type": "binary", "cluster_names": {0: "No Heart Disease", 1: "Heart Disease"}},
    "DiffWalk": {"type": "binary", "cluster_names": {0: "No Difficulty Walking", 1: "Difficulty Walking"}},
    "GenHlth": {"type": "ordinal", "cluster_names": {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}},
    "PhysHlth": {"type": "binary_threshold", "threshold": 14, "cluster_names": {0: "Good (0-13 days)", 1: "Poor (14+ days)"}},
    
    # ===== DATASET 1 =====
    "hypertension": {"type": "binary", "cluster_names": {0: "No Hypertension", 1: "Hypertension"}},
    "heart_disease": {"type": "binary", "cluster_names": {0: "No Heart Disease", 1: "Heart Disease"}},
    "bmi": {
        "type": "continuous_with_underweight",
        "bins": [18.5, 25, 30, 35, 40, float('inf')],
        "underweight_threshold": 18.5,
        "no_info_condition": lambda x: (x == 0),
        "cluster_names": {0: "No Info", 1: "Underweight", 2: "Healthy", 3: "Overweight", 4: "Obesity I", 5: "Obesity II", 6: "Obesity III"}
    },
    "age": {"type": "binary_median", "cluster_names": {0: "Younger", 1: "Older"}},
    "blood_glucose_level": {
        "type": "continuous",
        "bins": [100, 126, float('inf')],
        "no_info_condition": lambda x: (x < 70) | (x == 0),
        "cluster_names": {0: "No Info", 1: "Normal", 2: "Pre-diabetes", 3: "Diabetes"}
    },
    "HbA1c_level": {
        "type": "continuous",
        "bins": [5.7, 6.5, float('inf')],
        "no_info_condition": lambda x: (x == 0) | (x < 3.0),
        "cluster_names": {0: "No Info", 1: "Normal", 2: "Pre-diabetes", 3: "Diabetes"}
    },
    "gender": {"type": "binary", "cluster_names": {0: "Female", 1: "Male"}},
    "smoking_history": {
        "type": "categorical",
        "categories": {
            "Never": ["never", "Never"],
            "Former": ["former", "Former", "ever", "not current"],
            "Current": ["current", "Current"],
            "No Info": ["No Info", "No info", "no info", "Unknown"]
        },
        "cluster_names": {0: "Never", 1: "Former", 2: "Current", 3: "No Info"}
    },
    
    # ===== DATASET 2 (Pima) =====
    "Glucose": {
        "type": "continuous",
        "bins": [100, 126, float('inf')],
        "no_info_condition": lambda x: (x < 70) | (x == 0),
        "cluster_names": {0: "No Info", 1: "Normal", 2: "Pre-diabetes", 3: "Diabetes"}
    },
    "BloodPressure": {
        "type": "continuous",
        "bins": [80, 90, 120, float('inf')],
        "no_info_condition": lambda x: (x == 0),
        "cluster_names": {0: "No Info", 1: "Low", 2: "Normal", 3: "Elevated", 4: "High"}
    },
    "Insulin": {
        "type": "continuous",
        "bins": [16, 166, float('inf')],
        "no_info_condition": lambda x: (x == 0),
        "cluster_names": {0: "No Info", 1: "Low", 2: "Normal", 3: "High"}
    },
    
    # ===== SHARED (case variations) =====
    "BMI": {
        "type": "continuous_with_underweight",
        "bins": [18.5, 25, 30, 35, 40, float('inf')],
        "underweight_threshold": 18.5,
        "no_info_condition": lambda x: (x == 0),
        "cluster_names": {0: "No Info", 1: "Underweight", 2: "Healthy", 3: "Overweight", 4: "Obesity I", 5: "Obesity II", 6: "Obesity III"}
    },
    "Age": {"type": "binary_median", "cluster_names": {0: "Younger", 1: "Older"}},
}


# =====================================================
# CLUSTERING FUNCTIONS
# =====================================================
def get_medical_clusters(df, feature_name):
    """Assign clusters based on predefined medical thresholds."""
    if feature_name not in MEDICAL_THRESHOLDS:
        return None, None, None
    
    thresholds = MEDICAL_THRESHOLDS[feature_name]
    threshold_type = thresholds["type"]
    df = df.copy()
    df['cluster_id'] = -1
    
    if threshold_type == "continuous_with_underweight":
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
        no_info_mask = thresholds["no_info_condition"](df[feature_name])
        df.loc[no_info_mask, 'cluster_id'] = 0
        valid_mask = ~no_info_mask
        if valid_mask.any():
            bins_config = list(thresholds["bins"])
            first_edge = bins_config[0]
            min_val = df.loc[valid_mask, feature_name].min()
            start_bin = min(min_val - 1, first_edge - 1) if first_edge != -float('inf') else -float('inf')
            bins = [start_bin] + bins_config
            num_labels = len(bins_config)
            cut_result = pd.cut(df.loc[valid_mask, feature_name], bins=bins, 
                              labels=range(1, num_labels + 1), right=False)
            df.loc[valid_mask, 'cluster_id'] = cut_result.astype(int)
        cluster_names = thresholds["cluster_names"]
        num_clusters = len(cluster_names)
        
    elif threshold_type == "binary":
        # Handle both numeric (0/1) and string values (e.g., 'Female'/'Male')
        col_values = df[feature_name]
        # Check for string types using dtype.kind (O = object/string)
        is_string_type = (col_values.dtype.kind == 'O' or 
                          str(col_values.dtype) in ['object', 'str', 'string', 'category'])
        if is_string_type:
            # Map string values to 0/1 based on cluster_names
            value_mapping = {v: k for k, v in thresholds["cluster_names"].items()}
            df['cluster_id'] = col_values.map(value_mapping).fillna(-1).astype(int)
        else:
            df['cluster_id'] = col_values.astype(int)
        cluster_names = thresholds["cluster_names"]
        num_clusters = 2
    
    elif threshold_type == "binary_threshold":
        threshold_val = thresholds["threshold"]
        df['cluster_id'] = (df[feature_name] >= threshold_val).astype(int)
        cluster_names = thresholds["cluster_names"]
        num_clusters = 2
    
    elif threshold_type == "binary_median":
        median_val = df[feature_name].median()
        df['cluster_id'] = (df[feature_name] >= median_val).astype(int)
        cluster_names = thresholds["cluster_names"]
        num_clusters = 2
        
    elif threshold_type == "ordinal":
        df['cluster_id'] = df[feature_name].astype(int)
        cluster_names = thresholds["cluster_names"]
        num_clusters = len(cluster_names)
        
    elif threshold_type == "categorical":
        categories = thresholds["categories"]
        for cluster_id, (cat_name, cat_values) in enumerate(categories.items()):
            mask = df[feature_name].isin(cat_values)
            df.loc[mask, 'cluster_id'] = cluster_id
        df.loc[df['cluster_id'] == -1, 'cluster_id'] = len(categories) - 1
        cluster_names = thresholds["cluster_names"]
        num_clusters = len(cluster_names)
        
    elif threshold_type == "categorical_string":
        unique_vals = df[feature_name].unique()
        cluster_names = {i: str(val) for i, val in enumerate(unique_vals)}
        val_to_cluster = {val: i for i, val in enumerate(unique_vals)}
        df['cluster_id'] = df[feature_name].map(val_to_cluster)
        num_clusters = len(unique_vals)
    
    # Remove invalid clusters and reindex
    unique_clusters = sorted(df['cluster_id'].unique())
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
        df = df[df['cluster_id'] != -1]
    
    final_cluster_names = {}
    cluster_mapping = {}
    for new_id, old_id in enumerate(unique_clusters):
        cluster_mapping[old_id] = new_id
        if old_id in cluster_names:
            final_cluster_names[new_id] = cluster_names[old_id]
        else:
            final_cluster_names[new_id] = f"Cluster {old_id}"
    
    df['cluster_id'] = df['cluster_id'].map(cluster_mapping)
    return df, len(unique_clusters), final_cluster_names


def get_kmeans_clusters(df, feature_name):
    """Assign clusters using K-means with silhouette score optimization."""
    # Handle string/categorical columns by encoding them
    col_data = df[feature_name]
    if col_data.dtype == 'object' or str(col_data.dtype) == 'category':
        # For categorical, use unique values as clusters directly
        unique_vals = col_data.unique()
        val_to_cluster = {val: i for i, val in enumerate(unique_vals)}
        df = df.copy()
        df['cluster_id'] = col_data.map(val_to_cluster)
        cluster_names = {i: str(val) for i, val in enumerate(unique_vals)}
        return df, len(unique_vals), cluster_names
    
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
