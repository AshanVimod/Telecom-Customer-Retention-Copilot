import pandas as pd

import score


def test_label_for_one_hot_feature():
    assert score._label("Contract_Month-to-month", 1) == "Contract = Month-to-month"


def test_label_for_numeric_feature():
    assert score._label("tenure", 2) == "tenure = 2"


def test_prepare_forces_training_columns():
    # the model was trained with these columns, in this order
    score.columns = ["tenure", "TotalCharges", "Contract_Month-to-month", "Contract_Two year"]
    one_customer = pd.DataFrame([{
        "customerID": "A", "tenure": 2, "TotalCharges": " ", "Contract": "Month-to-month",
    }])
    out = score.prepare(one_customer)
    assert list(out.columns) == score.columns
    # a single customer has one contract type; the missing one must appear as 0
    assert out.iloc[0].tolist() == [2, 0.0, 1, 0]
