
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "ai4i2020_project_dataset.csv")
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# Missing values
df.isna().sum().to_csv(OUT / "missing_values.csv")

# Class balance
plt.figure(figsize=(6,4))
df["Machine failure"].value_counts().sort_index().plot(kind="bar")
plt.title("Machine Failure Class Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=160)
plt.close()

# Numeric distributions
for col in ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]:
    plt.figure(figsize=(7,4))
    sns.histplot(data=df, x=col, hue="Machine failure", kde=True, element="step")
    plt.title(f"{col} by Failure Status")
    plt.tight_layout()
    plt.savefig(OUT / (col.replace(" ", "_").replace("[","").replace("]","").replace("/","_") + ".png"), dpi=160)
    plt.close()

# Correlation heatmap
num = df.select_dtypes(include="number")
plt.figure(figsize=(10,7))
sns.heatmap(num.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=160)
plt.close()
