
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ai4i2020_project_dataset.csv"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# Remove identifiers and failure-mode labels to avoid leakage.
target = "Machine failure"
drop_cols = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
X = df.drop(columns=[target] + drop_cols)
y = df[target]

cat_cols = ["Type"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
    "SVM": SVC(probability=True, class_weight="balanced", random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

rows = []
fitted = {}

for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba)
    })
    fitted[name] = pipe

metrics = pd.DataFrame(rows).sort_values("F1", ascending=False)
metrics.to_csv(RESULTS / "classification_model_comparison.csv", index=False)

best_name = metrics.iloc[0]["Model"]
joblib.dump(fitted[best_name], MODELS / "best_failure_model.pkl")

pred = fitted[best_name].predict(X_test)
proba = fitted[best_name].predict_proba(X_test)[:, 1]
pd.DataFrame(confusion_matrix(y_test, pred)).to_csv(RESULTS / "confusion_matrix.csv", index=False)
with open(RESULTS / "classification_report.txt", "w") as f:
    f.write(classification_report(y_test, pred, digits=4))

# Auxiliary regression task: predict continuous tool wear from operating variables.
# This satisfies the course's classification + regression requirement without using
# the machine-failure label as a regression target.
reg_target = "Tool wear [min]"
reg_drop = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF", reg_target]
Xr = df.drop(columns=reg_drop)
yr = df[reg_target]
cat_r = ["Type"]
num_r = [c for c in Xr.columns if c not in cat_r]

reg_pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_r),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_r)
])

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.20, random_state=42)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
}

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
reg_rows = []
for name, model in reg_models.items():
    pipe = Pipeline([("preprocessor", reg_pre), ("model", model)])
    pipe.fit(Xr_train, yr_train)
    rp = pipe.predict(Xr_test)
    reg_rows.append({
        "Model": name,
        "MAE": mean_absolute_error(yr_test, rp),
        "RMSE": mean_squared_error(yr_test, rp) ** 0.5,
        "R2": r2_score(yr_test, rp)
    })

reg_metrics = pd.DataFrame(reg_rows).sort_values("R2", ascending=False)
reg_metrics.to_csv(RESULTS / "regression_model_comparison.csv", index=False)

print("Best classification model:", best_name)
print(metrics.to_string(index=False))
print("\nRegression results:")
print(reg_metrics.to_string(index=False))
