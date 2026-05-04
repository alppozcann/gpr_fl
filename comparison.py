"""
Comparison module for GP-FL results against paper baselines (LR, RFC).
Contains paper results data and visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Paper's aggregated results for comparison
# Format: [Accuracy, Precision, Recall, F1]
PAPER_RESULTS = {
    "Pregnancies": {"LR": [0.7338, 0.6167, 0.6727, 0.6435], "RFC": [0.7338, 0.6167, 0.6727, 0.6435]},
    "Glucose": {"LR": [0.7662, 0.7436, 0.5273, 0.6170], "RFC": [0.6883, 0.8889, 0.1455, 0.2500]},
    "BloodPressure": {"LR": [0.7792, 0.7333, 0.6000, 0.6600], "RFC": [0.7662, 0.6462, 0.7636, 0.7000]},
    "SkinThickness": {"LR": [0.7615, 0.6591, 0.7250, 0.6905], "RFC": [0.7615, 0.7188, 0.5750, 0.6389]},
    "Insulin": {"LR": [0.7848, 0.6667, 0.6400, 0.6531], "RFC": [0.7595, 0.6667, 0.4800, 0.5581]},
    "BMI": {"LR": [0.6818, 0.5417, 0.7091, 0.6142], "RFC": [0.7078, 0.7500, 0.2727, 0.4000]},
    "DiabetesPedigreeFunction": {"LR": [0.7468, 0.6379, 0.6727, 0.6549], "RFC": [0.7273, 0.6000, 0.7091, 0.6500]},
    "Age": {"LR": [0.7143, 0.5821, 0.7091, 0.6393], "RFC": [0.7273, 0.6102, 0.6545, 0.6316]}
}

# Paper's local (per-cluster) results for comparison
PAPER_LOCAL_RESULTS = {
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

# Cluster mapping for GP-FL client IDs to paper cluster labels
CLUSTER_MAPPING = {
    "Pregnancies": {1: 0, 2: 1},
    "BloodPressure": {1: 0, 2: 1},
    "Insulin": {1: 0, 2: 1},
    "Glucose": {2: "Normal (70-99)", 3: "Pre-diabetes (100-125)", 4: "Diabetes (>=126)"},
    "BMI": {2: "Healthy (18.5-24.9)", 3: "Overweight (25-29.9)", 4: "Obesity I (30-34.9)", 5: "Obesity II (35-39.9)", 6: "Obesity III (>=40)"}
}


def create_comparison_table(feature, gp_fl_results, plots_dir):
    """Create comparison table for a single feature."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    clients = gp_fl_results[feature]
    gp_avg = [np.mean([c[i] for c in clients]) for i in range(4)]
    
    table_data = [
        ["Model", "Accuracy", "Precision", "Recall", "F1", "Best?"],
        ["LR (Paper)", f"{PAPER_RESULTS[feature]['LR'][0]:.4f}", f"{PAPER_RESULTS[feature]['LR'][1]:.4f}", 
         f"{PAPER_RESULTS[feature]['LR'][2]:.4f}", f"{PAPER_RESULTS[feature]['LR'][3]:.4f}", ""],
        ["RFC (Paper)", f"{PAPER_RESULTS[feature]['RFC'][0]:.4f}", f"{PAPER_RESULTS[feature]['RFC'][1]:.4f}", 
         f"{PAPER_RESULTS[feature]['RFC'][2]:.4f}", f"{PAPER_RESULTS[feature]['RFC'][3]:.4f}", ""],
        ["GP-FL (Ours)", f"{gp_avg[0]:.4f}", f"{gp_avg[1]:.4f}", f"{gp_avg[2]:.4f}", f"{gp_avg[3]:.4f}", ""]
    ]
    
    for metric_idx in range(4):
        values = [PAPER_RESULTS[feature]['LR'][metric_idx], PAPER_RESULTS[feature]['RFC'][metric_idx], gp_avg[metric_idx]]
        if np.argmax(values) == 2:
            table_data[3][5] = "✓" if table_data[3][5] == "" else table_data[3][5] + "✓"
    
    colors = [['#4472C4']*6, ['#D6DCE4']*6, ['#D6DCE4']*6]
    gp_row_color = ['#E2EFDA']
    for i in range(4):
        lr_val, rfc_val, gp_val = PAPER_RESULTS[feature]['LR'][i], PAPER_RESULTS[feature]['RFC'][i], gp_avg[i]
        gp_row_color.append('#C6EFCE' if gp_val >= max(lr_val, rfc_val) else '#FFC7CE')
    gp_row_color.append('#E2EFDA')
    colors.append(gp_row_color)
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(colors[i][j])
            if i == 0:
                table[(i, j)].set_text_props(color='white', fontweight='bold')
            if i == 3:
                table[(i, j)].set_text_props(fontweight='bold')
    
    plt.title(f"Comparison: {feature}", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"comparison_{feature}.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: comparison_{feature}.png")


def create_summary_table(gp_fl_results, plots_dir):
    """Create summary comparison table for all features."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    summary_data = [["Feature", "LR Acc", "RFC Acc", "GP-FL Acc", "LR F1", "RFC F1", "GP-FL F1", "Winner (F1)"]]
    
    for feature in PAPER_RESULTS.keys():
        if feature not in gp_fl_results:
            continue
        clients = gp_fl_results[feature]
        gp_avg = [np.mean([c[i] for c in clients]) for i in range(4)]
        lr_f1, rfc_f1, gp_f1 = PAPER_RESULTS[feature]['LR'][3], PAPER_RESULTS[feature]['RFC'][3], gp_avg[3]
        
        if gp_f1 >= max(lr_f1, rfc_f1):
            winner = "GP-FL ✓"
        elif lr_f1 >= rfc_f1:
            winner = "LR"
        else:
            winner = "RFC"
        
        summary_data.append([feature, f"{PAPER_RESULTS[feature]['LR'][0]:.4f}", f"{PAPER_RESULTS[feature]['RFC'][0]:.4f}",
                            f"{gp_avg[0]:.4f}", f"{lr_f1:.4f}", f"{rfc_f1:.4f}", f"{gp_f1:.4f}", winner])
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    for j in range(len(summary_data[0])):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(summary_data)):
        for j in range(len(summary_data[0])):
            if j == 7 and "GP-FL" in summary_data[i][j]:
                table[(i, j)].set_facecolor('#C6EFCE')
                table[(i, j)].set_text_props(fontweight='bold')
            elif j == 3 or j == 6:
                gp_val = float(summary_data[i][j])
                lr_val = float(summary_data[i][j-2])
                rfc_val = float(summary_data[i][j-1])
                if gp_val >= max(lr_val, rfc_val):
                    table[(i, j)].set_facecolor('#C6EFCE')
    
    plt.title("Summary: GP-FL vs LR vs RFC (Paper Results)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparison_summary.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: comparison_summary.png")


def create_local_comparison_table(feature, gp_fl_local_results, plots_dir):
    """Create local (per-cluster) comparison table."""
    if feature not in PAPER_LOCAL_RESULTS or feature not in gp_fl_local_results:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    table_data = [["Cluster", "Model", "Accuracy", "Precision", "Recall", "F1", "Best F1?"]]
    clusters_to_compare = list(PAPER_LOCAL_RESULTS[feature].keys())
    
    for cluster in clusters_to_compare:
        paper_data = PAPER_LOCAL_RESULTS[feature][cluster]
        
        gp_client = None
        if feature in CLUSTER_MAPPING:
            for client_id, mapped_cluster in CLUSTER_MAPPING[feature].items():
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
                table[(i, j)].set_facecolor('#C6EFCE' if row[6] == "✓" else '#FFE6E6')
                table[(i, j)].set_text_props(fontweight='bold')
        elif row[0] == "---":
            for j in range(len(row)):
                table[(i, j)].set_facecolor('#FFFFFF')
                table[(i, j)].set_text_props(color='#CCCCCC')
    
    plt.title(f"Local Model Comparison: {feature}\n(GP-FL vs Paper's LR/RFC per Cluster)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"local_comparison_{feature}.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: local_comparison_{feature}.png")


def create_local_summary_table(gp_fl_local_results, plots_dir):
    """Create summary table for local model comparisons."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    summary_data = [["Feature", "Cluster", "LR F1", "RFC F1", "GP-FL F1", "Winner"]]
    
    for feature in ["Pregnancies", "BloodPressure", "Insulin"]:
        if feature not in PAPER_LOCAL_RESULTS or feature not in gp_fl_local_results:
            continue
        for cluster, paper_data in PAPER_LOCAL_RESULTS[feature].items():
            gp_client = [k for k, v in CLUSTER_MAPPING[feature].items() if v == cluster]
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
    plt.savefig(os.path.join(plots_dir, "local_comparison_summary.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: local_comparison_summary.png")


def generate_all_comparisons(gp_fl_results, gp_fl_local_results, plots_dir):
    """Generate all comparison tables."""
    print("\n📊 Generating comparison tables...")
    
    # Aggregated comparison tables
    for feature in PAPER_RESULTS.keys():
        if feature in gp_fl_results:
            create_comparison_table(feature, gp_fl_results, plots_dir)
    create_summary_table(gp_fl_results, plots_dir)
    
    # Local comparison tables
    for feature in ["Pregnancies", "BloodPressure", "Insulin", "Glucose", "BMI"]:
        create_local_comparison_table(feature, gp_fl_local_results, plots_dir)
    create_local_summary_table(gp_fl_local_results, plots_dir)
    
    print("✅ All comparison tables generated!")


def generate_paper_style_table(all_results, output_file, global_results=None):
    """
    Generate a formatted table like the paper shows.
    
    Args:
        all_results: dict with structure:
            {feature: {cluster_name: {"GP-FL": [acc, prec, rec, f1], "LR": [...], "RFC": [...]}}}
        output_file: path to save the table
        global_results: dict with structure:
            {feature: {'accuracy': x, 'precision': x, 'recall': x, 'f1': x, ...}}
    """
    lines = []
    lines.append("=" * 100)
    lines.append(f"{'Feature':<30} {'Cluster':<25} {'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    lines.append("=" * 100)
    
    for feature, clusters in all_results.items():
        first_feature = True
        
        # Add GLOBAL row first if available
        if global_results and feature in global_results:
            g = global_results[feature]
            lines.append(f"{feature:<30} {'** GLOBAL **':<25} {'GP-FL':<15} {g['accuracy']:>10.4f} {g['precision']:>10.4f} {g['recall']:>10.4f} {g['f1']:>10.4f}")
            first_feature = False
            lines.append("-" * 100)
        
        for cluster_name, models in clusters.items():
            first_cluster = True
            for model_name, metrics in models.items():
                feature_col = feature if first_feature else ""
                cluster_col = cluster_name if first_cluster else ""
                
                acc, prec, rec, f1 = metrics
                lines.append(f"{feature_col:<30} {cluster_col:<25} {model_name:<15} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")
                
                first_feature = False
                first_cluster = False
            lines.append("-" * 100)
    
    lines.append("=" * 100)
    
    # Add global summary table
    if global_results:
        lines.append("\n")
        lines.append("=" * 80)
        lines.append("GLOBAL MODEL SUMMARY (All Features)")
        lines.append("=" * 80)
        lines.append(f"{'Feature':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Samples':>10}")
        lines.append("-" * 80)
        for feature, g in global_results.items():
            lines.append(f"{feature:<30} {g['accuracy']:>10.4f} {g['precision']:>10.4f} {g['recall']:>10.4f} {g['f1']:>10.4f} {g['total_samples']:>10}")
        lines.append("=" * 80)
    
    table_text = "\n".join(lines)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(table_text)
    
    print(f"📄 Paper-style table saved to: {output_file}")
    return table_text