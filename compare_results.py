import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "dataset_2/plots"
os.makedirs(output_dir, exist_ok=True)

paper_results = {
    "Pregnancies": {"LR": [0.7338, 0.6167, 0.6727, 0.6435], "RFC": [0.7338, 0.6167, 0.6727, 0.6435]},
    "Glucose": {"LR": [0.7662, 0.7436, 0.5273, 0.6170], "RFC": [0.6883, 0.8889, 0.1455, 0.2500]},
    "BloodPressure": {"LR": [0.7792, 0.7333, 0.6000, 0.6600], "RFC": [0.7662, 0.6462, 0.7636, 0.7000]},
    "SkinThickness": {"LR": [0.7615, 0.6591, 0.7250, 0.6905], "RFC": [0.7615, 0.7188, 0.5750, 0.6389]},
    "Insulin": {"LR": [0.7848, 0.6667, 0.6400, 0.6531], "RFC": [0.7595, 0.6667, 0.4800, 0.5581]},
    "BMI": {"LR": [0.6818, 0.5417, 0.7091, 0.6142], "RFC": [0.7078, 0.7500, 0.2727, 0.4000]},
    "DiabetesPedigreeFunction": {"LR": [0.7468, 0.6379, 0.6727, 0.6549], "RFC": [0.7273, 0.6000, 0.7091, 0.6500]},
    "Age": {"LR": [0.7143, 0.5821, 0.7091, 0.6393], "RFC": [0.7273, 0.6102, 0.6545, 0.6316]}
}

# Updated GP-FL results with paper's methodology (Medical Thresholds for Glucose/BMI)
gp_fl_results = {
    "Pregnancies": [[0.8687, 0.7407, 0.7692, 0.7547], [0.6786, 0.6774, 0.7241, 0.7000]],
    "Glucose": [[0.8108, 0.0000, 0.0000, 0.0000], [0.7273, 0.4400, 0.9167, 0.5946], [0.7167, 0.8519, 0.6389, 0.7302]],  # Normal, Pre-diabetes, Diabetes
    "BloodPressure": [[0.7808, 0.6515, 0.8269, 0.7288], [0.7500, 0.6667, 0.6667, 0.6667]],
    "SkinThickness": [[0.7778, 0.7059, 0.9231, 0.8000], [0.7234, 0.6190, 0.7222, 0.6667], [1.0000, 1.0000, 1.0000, 1.0000], [0.7391, 0.6316, 0.7059, 0.6667]],
    "Insulin": [[0.5926, 0.4737, 0.9000, 0.6207], [0.7342, 0.6897, 0.6250, 0.6557], [0.8649, 0.7500, 0.6667, 0.7059], [0.8889, 0.7500, 1.0000, 0.8571]],
    "BMI": [[0.9048, 0.6667, 0.6667, 0.6667], [0.7778, 1.0000, 0.2000, 0.3333], [0.7778, 0.6400, 0.9412, 0.7619], [0.8000, 0.8000, 0.6667, 0.7273], [0.7000, 0.7143, 0.8333, 0.7692]],  # Healthy, Overweight, Obesity I/II/III
    "DiabetesPedigreeFunction": [[0.6923, 0.6333, 0.9500, 0.7600], [0.8000, 0.6667, 0.7895, 0.7229]],
    "Age": [[0.8505, 0.6667, 0.8148, 0.7333], [0.6383, 0.6957, 0.6154, 0.6531]]
}

metrics = ["Accuracy", "Precision", "Recall", "F1"]

def get_gp_avg(feature):
    clients = gp_fl_results[feature]
    return [np.mean([c[i] for c in clients]) for i in range(4)]

for feature in paper_results.keys():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    gp_avg = get_gp_avg(feature)
    
    table_data = [
        ["Model", "Accuracy", "Precision", "Recall", "F1", "Best?"],
        ["LR (Paper)", f"{paper_results[feature]['LR'][0]:.4f}", f"{paper_results[feature]['LR'][1]:.4f}", 
         f"{paper_results[feature]['LR'][2]:.4f}", f"{paper_results[feature]['LR'][3]:.4f}", ""],
        ["RFC (Paper)", f"{paper_results[feature]['RFC'][0]:.4f}", f"{paper_results[feature]['RFC'][1]:.4f}", 
         f"{paper_results[feature]['RFC'][2]:.4f}", f"{paper_results[feature]['RFC'][3]:.4f}", ""],
        ["GP-FL (Ours)", f"{gp_avg[0]:.4f}", f"{gp_avg[1]:.4f}", f"{gp_avg[2]:.4f}", f"{gp_avg[3]:.4f}", ""]
    ]
    
    for i, metric_idx in enumerate(range(4)):
        values = [
            paper_results[feature]['LR'][metric_idx],
            paper_results[feature]['RFC'][metric_idx],
            gp_avg[metric_idx]
        ]
        best_idx = np.argmax(values)
        if best_idx == 2:
            table_data[3][5] = "✓" if table_data[3][5] == "" else table_data[3][5] + "✓"
    
    colors = [['#4472C4']*6]
    colors.append(['#D6DCE4']*6)
    colors.append(['#D6DCE4']*6)
    
    gp_row_color = []
    for i in range(6):
        if i == 0:
            gp_row_color.append('#E2EFDA')
        elif i <= 4:
            lr_val = paper_results[feature]['LR'][i-1]
            rfc_val = paper_results[feature]['RFC'][i-1]
            gp_val = gp_avg[i-1]
            if gp_val >= max(lr_val, rfc_val):
                gp_row_color.append('#C6EFCE')
            else:
                gp_row_color.append('#FFC7CE')
        else:
            gp_row_color.append('#E2EFDA')
    colors.append(gp_row_color)
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i][j])
            if i == 0:
                cell.set_text_props(color='white', fontweight='bold')
            if i == 3:
                cell.set_text_props(fontweight='bold')
    
    plt.title(f"Comparison: {feature}", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"comparison_{feature}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")

fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('off')

summary_data = [["Feature", "LR Acc", "RFC Acc", "GP-FL Acc", "LR F1", "RFC F1", "GP-FL F1", "Winner (F1)"]]

for feature in paper_results.keys():
    gp_avg = get_gp_avg(feature)
    lr_f1 = paper_results[feature]['LR'][3]
    rfc_f1 = paper_results[feature]['RFC'][3]
    gp_f1 = gp_avg[3]
    
    if gp_f1 >= max(lr_f1, rfc_f1):
        winner = "GP-FL ✓"
    elif lr_f1 >= rfc_f1:
        winner = "LR"
    else:
        winner = "RFC"
    
    summary_data.append([
        feature,
        f"{paper_results[feature]['LR'][0]:.4f}",
        f"{paper_results[feature]['RFC'][0]:.4f}",
        f"{gp_avg[0]:.4f}",
        f"{lr_f1:.4f}",
        f"{rfc_f1:.4f}",
        f"{gp_f1:.4f}",
        winner
    ])

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

filename = os.path.join(output_dir, "comparison_summary.png")
plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {filename}")

print("\nAll comparison tables saved!")
