import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "dataset_2/plots"
os.makedirs(output_dir, exist_ok=True)

paper_local_results = {
    "Pregnancies": {
        0: {"LR": [0.7766, 0.6071, 0.6296, 0.6182], "RFC": [0.7553, 0.5667, 0.6296, 0.5965]},
        1: {"LR": [0.6667, 0.6250, 0.7143, 0.6667], "RFC": [0.7000, 0.6667, 0.7143, 0.6897]}
    },
    "Glucose": {
        "Normal (70-99)": {"LR": [0.9048, 0.0000, 0.0000, 0.0000], "RFC": [0.9048, 0.0000, 0.0000, 0.0000]},
        "Pre-diabetes (100-125)": {"LR": [0.7000, 0.2500, 0.0769, 0.1176], "RFC": [0.7600, 1.0000, 0.0769, 0.1429]},
        "Diabetes (>=126)": {"LR": [0.7167, 0.8000, 0.7368, 0.7671], "RFC": [0.4667, 0.8750, 0.1842, 0.3043]}
    },
    "BMI": {
        "Healthy (18.5-24.9)": {"LR": [0.8000, 0.0000, 0.0000, 0.0000], "RFC": [0.8500, 0.5000, 0.2500, 0.3333]},
        "Overweight (25-29.9)": {"LR": [0.7500, 0.5000, 0.2857, 0.3636], "RFC": [0.7917, 0.6667, 0.2857, 0.4000]},
        "Obesity I (30-34.9)": {"LR": [0.6667, 0.5652, 0.7647, 0.6500], "RFC": [0.6889, 0.5714, 0.7059, 0.6316]},
        "Obesity II (35-39.9)": {"LR": [0.6333, 0.5714, 0.8000, 0.6667], "RFC": [0.7333, 0.7500, 0.6000, 0.6667]},
        "Obesity III (>=40)": {"LR": [0.6500, 0.6364, 0.7778, 0.7000], "RFC": [0.6500, 0.7143, 0.5556, 0.6250]}
    },
    "Insulin": {
        0: {"LR": [0.7857, 0.7273, 1.0000, 0.8421], "RFC": [0.7143, 0.8333, 0.6250, 0.7143]},
        1: {"LR": [0.7846, 0.6154, 0.4706, 0.5333], "RFC": [0.7692, 0.5833, 0.4118, 0.4828]}
    },
    "BloodPressure": {
        0: {"LR": [0.7832, 0.7368, 0.5714, 0.6437], "RFC": [0.7762, 0.6441, 0.7755, 0.7037]},
        1: {"LR": [0.7273, 0.7143, 0.8333, 0.7692], "RFC": [0.6364, 0.6667, 0.6667, 0.6667]}
    }
}

# Updated GP-FL local results with paper's methodology (Medical Thresholds for Glucose/BMI)
gp_fl_local_results = {
    "Pregnancies": {
        1: [0.8687, 0.7407, 0.7692, 0.7547],
        2: [0.6786, 0.6774, 0.7241, 0.7000]
    },
    "Glucose": {
        2: [0.8108, 0.0000, 0.0000, 0.0000],       # Normal (70-99)
        3: [0.7273, 0.4400, 0.9167, 0.5946],       # Pre-diabetes (100-125)
        4: [0.7167, 0.8519, 0.6389, 0.7302]        # Diabetes (>=126)
    },
    "BMI": {
        2: [0.9048, 0.6667, 0.6667, 0.6667],       # Healthy (18.5-24.9)
        3: [0.7778, 1.0000, 0.2000, 0.3333],       # Overweight (25-29.9)
        4: [0.7778, 0.6400, 0.9412, 0.7619],       # Obesity I (30-34.9)
        5: [0.8000, 0.8000, 0.6667, 0.7273],       # Obesity II (35-39.9)
        6: [0.7000, 0.7143, 0.8333, 0.7692]        # Obesity III (>=40)
    },
    "BloodPressure": {
        1: [0.7808, 0.6515, 0.8269, 0.7288],
        2: [0.7500, 0.6667, 0.6667, 0.6667]
    },
    "Insulin": {
        1: [0.5926, 0.4737, 0.9000, 0.6207],
        2: [0.7342, 0.6897, 0.6250, 0.6557],
        4: [0.8649, 0.7500, 0.6667, 0.7059],
        5: [0.8889, 0.7500, 1.0000, 0.8571]
    }
}

cluster_mapping = {
    "Pregnancies": {1: 0, 2: 1},
    "BloodPressure": {1: 0, 2: 1},
    "Insulin": {1: 0, 2: 1},
    "Glucose": {2: "Normal (70-99)", 3: "Pre-diabetes (100-125)", 4: "Diabetes (>=126)"},
    "BMI": {2: "Healthy (18.5-24.9)", 3: "Overweight (25-29.9)", 4: "Obesity I (30-34.9)", 5: "Obesity II (35-39.9)", 6: "Obesity III (>=40)"}
}

def create_local_comparison_table(feature):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    table_data = [["Cluster", "Model", "Accuracy", "Precision", "Recall", "F1", "Best F1?"]]
    
    if feature in ["Glucose", "BMI"]:
        clusters_to_compare = list(paper_local_results[feature].keys())
        gp_mapping = cluster_mapping[feature]
    else:
        clusters_to_compare = list(paper_local_results[feature].keys())
        gp_mapping = cluster_mapping[feature]
    
    for cluster in clusters_to_compare:
        paper_data = paper_local_results[feature][cluster]
        
        gp_client = None
        for client_id, mapped_cluster in cluster_mapping[feature].items():
            if mapped_cluster == cluster:
                gp_client = client_id
                break
        
        lr_data = paper_data["LR"]
        rfc_data = paper_data["RFC"]
        gp_data = gp_fl_local_results[feature].get(gp_client, [0, 0, 0, 0]) if gp_client else [0, 0, 0, 0]
        
        f1_values = [lr_data[3], rfc_data[3], gp_data[3]]
        best_idx = np.argmax(f1_values)
        best_markers = ["", "", ""]
        best_markers[best_idx] = "✓"
        
        cluster_label = f"Cluster {cluster}" if isinstance(cluster, int) else cluster
        
        table_data.append([cluster_label, "LR", f"{lr_data[0]:.4f}", f"{lr_data[1]:.4f}", 
                          f"{lr_data[2]:.4f}", f"{lr_data[3]:.4f}", best_markers[0]])
        table_data.append(["", "RFC", f"{rfc_data[0]:.4f}", f"{rfc_data[1]:.4f}", 
                          f"{rfc_data[2]:.4f}", f"{rfc_data[3]:.4f}", best_markers[1]])
        table_data.append(["", "GP-FL", f"{gp_data[0]:.4f}", f"{gp_data[1]:.4f}", 
                          f"{gp_data[2]:.4f}", f"{gp_data[3]:.4f}", best_markers[2]])
        table_data.append(["---", "---", "---", "---", "---", "---", "---"])
    
    table_data = table_data[:-1]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(table_data)):
        row = table_data[i]
        if row[1] == "GP-FL":
            for j in range(len(row)):
                if row[6] == "✓":
                    table[(i, j)].set_facecolor('#C6EFCE')
                else:
                    table[(i, j)].set_facecolor('#FFE6E6')
                table[(i, j)].set_text_props(fontweight='bold')
        elif row[0] == "---":
            for j in range(len(row)):
                table[(i, j)].set_facecolor('#FFFFFF')
                table[(i, j)].set_text_props(color='#CCCCCC')
    
    plt.title(f"Local Model Comparison: {feature}\n(GP-FL vs Paper's LR/RFC per Cluster)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"local_comparison_{feature}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")

for feature in ["Pregnancies", "BloodPressure", "Insulin", "Glucose", "BMI"]:
    create_local_comparison_table(feature)

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

summary_data = [["Feature", "Cluster", "LR F1", "RFC F1", "GP-FL F1", "Winner"]]

for feature in ["Pregnancies", "BloodPressure", "Insulin"]:
    for cluster, paper_data in paper_local_results[feature].items():
        gp_client = [k for k, v in cluster_mapping[feature].items() if v == cluster]
        gp_client = gp_client[0] if gp_client else None
        
        lr_f1 = paper_data["LR"][3]
        rfc_f1 = paper_data["RFC"][3]
        gp_f1 = gp_fl_local_results[feature].get(gp_client, [0,0,0,0])[3] if gp_client else 0
        
        if gp_f1 >= max(lr_f1, rfc_f1):
            winner = "GP-FL ✓"
        elif lr_f1 >= rfc_f1:
            winner = "LR"
        else:
            winner = "RFC"
        
        summary_data.append([feature, f"Cluster {cluster}", f"{lr_f1:.4f}", f"{rfc_f1:.4f}", f"{gp_f1:.4f}", winner])

table = ax.table(cellText=summary_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

for j in range(len(summary_data[0])):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

for i in range(1, len(summary_data)):
    if "GP-FL" in summary_data[i][5]:
        table[(i, 4)].set_facecolor('#C6EFCE')
        table[(i, 5)].set_facecolor('#C6EFCE')
        table[(i, 5)].set_text_props(fontweight='bold')

plt.title("Local Model Summary: GP-FL vs LR/RFC (Per Cluster)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

filename = os.path.join(output_dir, "local_comparison_summary.png")
plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {filename}")

print("\nLocal comparison tables saved!")
