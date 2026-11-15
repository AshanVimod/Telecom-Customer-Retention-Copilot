"""Convert the IBM 'Telco_customer_churn.xlsx' (33 columns) into the classic
21-column CSV layout that the rest of this project expects.

Usage:  python scripts/convert_ibm_xlsx.py data/telco_churn.xlsx data/telco_churn.csv
"""
import sys

import pandas as pd

RENAME = {
    "CustomerID": "customerID",
    "Gender": "gender",
    "Senior Citizen": "SeniorCitizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "Tenure Months": "tenure",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Contract": "Contract",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Churn Label": "Churn",
}

# Dropped on purpose:
#   Churn Value, Churn Score, CLTV, Churn Reason -> known only AFTER the customer
#     churned (or derived from it); training on them would be label leakage.
#   Count, Country, State, City, Zip Code, Lat Long, Latitude, Longitude -> geography
#     that the retention policies do not use.


def main(src: str, dst: str) -> None:
    df = pd.read_excel(src)
    out = df[list(RENAME)].rename(columns=RENAME)
    out["SeniorCitizen"] = (out["SeniorCitizen"] == "Yes").astype(int)
    out.to_csv(dst, index=False)
    print(f"Wrote {dst}: {out.shape[0]} rows x {out.shape[1]} columns")
    print(out["Churn"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
