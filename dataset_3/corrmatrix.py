import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_3/diabetes_3.csv")

corr_matrix = df.corr(numeric_only=True)

# Get correlation with target variable (Diabetes_binary)
target_corr = corr_matrix["Diabetes_binary"].drop("Diabetes_binary")

# Filter features with correlation > 0.15 with target
selected_features = target_corr[abs(target_corr) > 0.15]

print("\nFeatures with correlation > 0.15 with Diabetes_binary:")
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
plt.savefig("correlation_matrix.png")
