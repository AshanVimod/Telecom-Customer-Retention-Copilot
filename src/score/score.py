import glob
import json
import os

import joblib
import pandas as pd
import shap


def _find(name):
    root = os.environ["AZUREML_MODEL_DIR"]
    return glob.glob(os.path.join(root, "**", name), recursive=True)[0]


def init():
    """Runs once when the container starts: load the model into memory."""
    global model, columns, explainer
    model = joblib.load(_find("model.pkl"))
    with open(_find("columns.json")) as f:
        columns = json.load(f)
    explainer = shap.TreeExplainer(model)


def prepare(df):
    """Same preparation as training, then force the exact training columns."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", 0), errors="coerce").fillna(0)
    df = df.drop(columns=["customerID", "Churn"], errors="ignore")
    return pd.get_dummies(df, dtype=int).reindex(columns=columns, fill_value=0)


def _label(feature, value):
    # "Contract_Month-to-month" -> "Contract = Month-to-month"; "tenure" -> "tenure = 2"
    if "_" in feature:
        return feature.replace("_", " = ", 1)
    return f"{feature} = {value:g}"


def run(raw_data):
    """Runs on every request."""
    rows = json.loads(raw_data)["data"]
    X = prepare(pd.DataFrame(rows))
    prob = model.predict_proba(X)[:, 1]
    sv = explainer.shap_values(X)
    results = []
    for i in range(len(X)):
        contrib = sorted(zip(columns, sv[i]), key=lambda t: -t[1])
        reasons = [_label(f, X.iloc[i][f]) for f, v in contrib if v > 0 and X.iloc[i][f] != 0][:3]
        p = float(prob[i])
        results.append({
            "customerID": rows[i].get("customerID"),
            "churn_probability": round(p, 3),
            "risk_level": "high" if p >= 0.6 else "medium" if p >= 0.3 else "low",
            "top_reasons": reasons,
        })
    return results
