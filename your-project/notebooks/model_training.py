# model_training.py — Phase B: EDA

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# 1. Basic dataset info
print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe())

# 2. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 3. Save first rows to CSV (for reference)
df.to_csv("../data/iris_sample.csv", index=False)

# 4. Visualizations
sns.pairplot(df, hue="target")
plt.savefig("../data/pairplot.png")
plt.close()

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.savefig("../data/corr_heatmap.png")
plt.close()

print("\nEDA completed. Charts saved in data/ folder.")
