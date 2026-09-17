"""Inference demo: churn probability for one customer."""

import pandas as pd

from data import generate
from train import build_pipeline

SAMPLE = {
    "tenure_months": 3,
    "monthly_charges": 89.5,
    "total_charges": 268.5,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "payment_method": "electronic-check",
    "support_calls": 4,
}


def main():
    df = generate(5000)
    X, y = df.drop(columns=["churn"]), df["churn"]
    pipe = build_pipeline().fit(X, y)
    proba = pipe.predict_proba(pd.DataFrame([SAMPLE]))[0, 1]
    print("sample customer:", SAMPLE)
    print(f"predicted churn probability: {proba:.4f}")


if __name__ == "__main__":
    main()
