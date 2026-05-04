import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_3/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

corr_matrix = df.corr(numeric_only=True)

target_corr = corr_matrix["Diabetes_binary"].drop("Diabetes_binary")

# Filter features with correlation > 0.15 with target
selected_features = target_corr[target_corr > 0.20]

print("\nFeatures with correlation > 0.20 with Diabetes_binary:")
print("="*55)
for feature, corr_value in selected_features.sort_values(ascending=False).items():
    print(f"{feature}: {corr_value:.3f}")

print(f"\nTotal selected features: {len(selected_features)}")
print(f"Selected feature names: {list(selected_features.index)}")

plt.figure(figsize=(12,10))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of Dataset 3")
plt.savefig("dataset_3/correlation_matrix.png")
