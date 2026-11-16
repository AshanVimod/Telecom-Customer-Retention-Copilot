"""Extra checks on the churn model: calibration, fairness by segment, business estimate.

Retrains locally with the registered model's settings and the same train/test
split as src/train/train.py, then reports on the held-out 20%.

Run from the project root:  python eval/model_report.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "train"))

from train import prepare, recall_at_top  # noqa: E402

raw = pd.read_csv(ROOT / "data" / "telco_churn.csv")
y = (raw["Churn"] == "Yes").astype(int)
X = prepare(raw)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
raw_te = raw.loc[Xte.index]

model = XGBClassifier(
    max_depth=3, learning_rate=0.03, n_estimators=600, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(), eval_metric="aucpr", random_state=42,
).fit(Xtr, ytr)
prob = pd.Series(model.predict_proba(Xte)[:, 1], index=Xte.index)
pred = prob >= 0.5

print("## Calibration (is a predicted 70% really about 70%?)")
frac_pos, mean_pred = calibration_curve(yte, prob, n_bins=10)
print(pd.DataFrame({"mean predicted": mean_pred.round(2), "actual churn rate": frac_pos.round(2)}).to_string(index=False))
print(f"Brier score: {brier_score_loss(yte, prob):.3f} (lower is better)")
plt.figure(figsize=(4.5, 4.5))
plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
plt.plot(mean_pred, frac_pos, marker="o", label="model")
plt.xlabel("Predicted churn probability")
plt.ylabel("Actual churn rate")
plt.title("Calibration on the test set")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "docs" / "calibration.png", dpi=120)

print("\n## Fairness by segment (threshold 0.5)")
rows = []
for col in ["gender", "SeniorCitizen"]:
    for value, idx in raw_te.groupby(col).groups.items():
        yt, pr, pd_ = yte.loc[idx], prob.loc[idx], pred.loc[idx]
        rows.append({
            "segment": f"{col}={value}", "customers": len(idx), "churn rate": round(yt.mean(), 3),
            "AUC": round(roc_auc_score(yt, pr), 3),
            "recall": round((pd_ & (yt == 1)).sum() / (yt == 1).sum(), 3),
            "false positive rate": round((pd_ & (yt == 0)).sum() / (yt == 0).sum(), 3),
        })
print(pd.DataFrame(rows).to_string(index=False))

print("\n## Business estimate (illustrative assumptions)")
top = prob.sort_values(ascending=False).index[: int(0.1 * len(prob))]
churners_reached = int(yte.loc[top].sum())
save_rate = 0.25  # assumption: 1 in 4 contacted churners accepts an offer and stays
avg_charge = raw_te.loc[top, "MonthlyCharges"].mean()
saved = churners_reached * save_rate
print(f"Test customers: {len(prob)}; contacting the riskiest 10% = {len(top)} customers")
print(f"Churners among them: {churners_reached} of {int(yte.sum())} ({recall_at_top(yte, prob.values, 0.1):.0%} of all churners)")
print(f"Assumed save rate {save_rate:.0%} -> {saved:.0f} customers kept")
print(f"Average monthly charge of that group: {avg_charge:.2f}")
print(f"Revenue protected per year: {saved:.0f} x {avg_charge:.2f} x 12 = {saved * avg_charge * 12:,.0f}")
