import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_3/diabetes_3.csv")

# Convert Diabetes_012 to binary: 0 -> 0 (no diabetes), 1-2 -> 1 (diabetes)
df["Diabetes_binary"] = df["Diabetes_012"].apply(lambda x: 0 if x == 0 else 1)
df = df.drop("Diabetes_012", axis=1)

# Remove duplicates and show count
rows_before = len(df)
df = df.drop_duplicates()
rows_after = len(df)
print(f"Duplicate rows removed: {rows_before - rows_after}")
print(f"Rows before: {rows_before}, Rows after: {rows_after}")

df.to_csv("dataset_3/diabetes_3.csv", index=False)