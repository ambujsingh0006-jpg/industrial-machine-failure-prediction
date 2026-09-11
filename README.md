# Industrial Machine Failure Prediction for Predictive Maintenance

## Project Overview
This project predicts whether an industrial machine is likely to fail from process and sensor variables. It compares Logistic Regression, Decision Tree, Random Forest and SVM classifiers, and includes an auxiliary regression task for continuous tool-wear estimation.

## Dataset
The project uses an **AI4I 2020-style predictive-maintenance dataset** with 10,000 records and 14 columns based on the public UCI AI4I 2020 Predictive Maintenance Dataset structure.

**Important:** The included CSV is a project-generated dataset following the published AI4I structure and documented failure rules. It is not a byte-for-byte copy of UCI's downloadable CSV. For a submission that requires the exact UCI file, replace `data/ai4i2020_project_dataset.csv` with the official `ai4i2020.csv`.

Official source: https://archive.ics.uci.edu/dataset/601/ai4i

## Models
### Classification
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Metrics:
- Precision
- Recall
- F1-score
- ROC-AUC

### Regression
- Linear Regression
- Random Forest Regressor

Regression metrics:
- MAE
- RMSE
- R²

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 src/eda.py
python3 src/train_models.py
streamlit run app/app.py
```

## Project Structure

```text
Industrial_Machine_Failure_Prediction/
├── data/
├── src/
├── models/
├── results/
├── app/
├── notebooks/
├── docs/
├── requirements.txt
└── README.md
```

## Career value
The project demonstrates an end-to-end applied ML workflow: preprocessing, class imbalance handling, model comparison, evaluation, feature analysis, prediction and deployment.
