import numpy as np
import pandas as pd

from train import prepare, recall_at_top


def sample():
    return pd.DataFrame({
        "customerID": ["A", "B"],
        "Contract": ["Month-to-month", "Two year"],
        "tenure": [1, 60],
        "TotalCharges": ["29.85", " "],
        "Churn": ["Yes", "No"],
    })


def test_prepare_drops_id_and_label():
    out = prepare(sample())
    assert "customerID" not in out.columns
    assert "Churn" not in out.columns


def test_prepare_blank_total_charges_becomes_zero():
    out = prepare(sample())
    assert out["TotalCharges"].tolist() == [29.85, 0.0]


def test_prepare_turns_text_into_columns():
    out = prepare(sample())
    assert out["Contract_Month-to-month"].tolist() == [1, 0]
    assert out["Contract_Two year"].tolist() == [0, 1]


def test_recall_at_top():
    y = pd.Series([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    prob = np.array([0.9, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # top 10% is one customer, a churner, out of two churners in total
    assert recall_at_top(y, prob, 0.1) == 0.5
