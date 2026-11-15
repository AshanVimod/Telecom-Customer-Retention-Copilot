"""Train an XGBoost churn model, log metrics and a SHAP plot to MLflow,
and save the files the scoring endpoint needs.

Local test:
    python src/train/train.py --data data/telco_churn.csv --model_dir outputs/model
In Azure ML the same script runs unchanged; see azureml/job.yml.
"""
import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # no display on a training cluster
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw customer rows into the numeric table the model trains on."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID", "Churn"], errors="ignore")
    return pd.get_dummies(df, drop_first=False, dtype=int)


def recall_at_top(y_true: pd.Series, prob: np.ndarray, fraction: float = 0.1) -> float:
    """If the team can only call the riskiest `fraction` of customers,
    what share of all churners do they reach?"""
    top = np.argsort(-prob)[: int(fraction * len(prob))]
    return float(y_true.iloc[top].sum() / y_true.sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--n_estimators", type=int, default=400)
    a = p.parse_args()

    raw = pd.read_csv(a.data)
    y = (raw["Churn"] == "Yes").astype(int)
    X = prepare(raw)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    with mlflow.start_run():
        mlflow.log_params(
            {"max_depth": a.max_depth, "learning_rate": a.learning_rate, "n_estimators": a.n_estimators}
        )
        model = XGBClassifier(
            max_depth=a.max_depth,
            learning_rate=a.learning_rate,
            n_estimators=a.n_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(),  # handle 26% churn imbalance
            eval_metric="aucpr",
            random_state=42,
        )
        model.fit(Xtr, ytr)

        prob = model.predict_proba(Xte)[:, 1]
        metrics = {
            "test_auc": roc_auc_score(yte, prob),
            "test_pr_auc": average_precision_score(yte, prob),
            "recall_at_top10pct": recall_at_top(yte, prob, 0.1),
        }
        mlflow.log_metrics(metrics)
        print(json.dumps(metrics, indent=2))
        print(classification_report(yte, (prob > 0.5).astype(int)))

        # global explanation: which features push churn risk up or down
        shap_values = shap.TreeExplainer(model).shap_values(Xte)
        shap.summary_plot(shap_values, Xte, show=False, max_display=15)
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "shap_summary.png")

        # files the scoring endpoint loads in Phase 3
        os.makedirs(a.model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(a.model_dir, "model.pkl"))
        with open(os.path.join(a.model_dir, "columns.json"), "w") as f:
            json.dump(list(X.columns), f)


if __name__ == "__main__":
    main()
