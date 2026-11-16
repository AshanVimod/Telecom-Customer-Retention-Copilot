"""The three tools the agent can call. Each is a plain Python function."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "rag"))

from rag_search import search_policies  # noqa: E402  (tool 2, defined in src/rag)

load_dotenv()

# The app ships a small sample; locally the full dataset is used if present.
_DATA = next(p for p in [ROOT / "data" / "telco_churn.csv", ROOT / "data" / "sample_customers.csv"] if p.exists())
CUSTOMERS = pd.read_csv(_DATA).drop(columns=["Churn"], errors="ignore").set_index("customerID")

PROFILE_FIELDS = [
    "tenure", "Contract", "InternetService", "TechSupport", "OnlineSecurity",
    "MonthlyCharges", "PaymentMethod", "PaperlessBilling", "SeniorCitizen",
]


def get_churn_risk(customer_id: str) -> dict:
    """Tool 1: call the Azure ML endpoint for one customer."""
    if customer_id not in CUSTOMERS.index:
        raise ValueError(f"Customer '{customer_id}' was not found.")
    row = json.loads(CUSTOMERS.loc[customer_id].to_json())  # plain Python types
    row["customerID"] = customer_id
    r = requests.post(
        os.environ["CHURN_ENDPOINT_URL"],
        headers={"Authorization": f"Bearer {os.environ['CHURN_ENDPOINT_KEY']}"},
        json={"data": [row]},
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    if isinstance(body, str):  # the scoring server may return JSON as a string
        body = json.loads(body)
    result = body[0]
    result["profile"] = {k: row[k] for k in PROFILE_FIELDS}
    return result


def translate(text: str, to: str) -> str:
    """Tool 3: translate text into Sinhala ('si') or Tamil ('ta')."""
    r = requests.post(
        "https://api.cognitive.microsofttranslator.com/translate",
        params={"api-version": "3.0", "to": to},
        json=[{"text": text}],
        headers={
            "Ocp-Apim-Subscription-Key": os.environ["TRANSLATOR_KEY"],
            "Ocp-Apim-Subscription-Region": os.environ["TRANSLATOR_REGION"],
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()[0]["translations"][0]["text"]


FUNCS = {"get_churn_risk": get_churn_risk, "search_policies": search_policies, "translate": translate}
