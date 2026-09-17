"""Seeded synthetic telecom churn dataset.

The data is synthetic but realistic: churn probability depends on tenure,
charges, contract type, internet service, payment method and support calls,
plus noise. Seeded RNG => the exact same dataset every run.
"""

import numpy as np
import pandas as pd

SEED = 42


def generate(n_customers=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n_customers)
    monthly = np.round(rng.normal(65, 20, n_customers).clip(20, 120), 2)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        n_customers, p=[0.5, 0.3, 0.2],
    )
    internet = rng.choice(
        ["fiber", "dsl", "none"], n_customers, p=[0.45, 0.35, 0.2]
    )
    payment = rng.choice(
        ["electronic-check", "mailed-check", "bank-transfer", "credit-card"],
        n_customers, p=[0.35, 0.2, 0.25, 0.2],
    )
    support_calls = rng.poisson(1.2, n_customers)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.1, n_customers), 2)

    logit = (
        -2.4
        + 0.028 * monthly
        - 0.045 * tenure
        + 0.9 * (contract == "month-to-month")
        - 0.5 * (contract == "two-year")
        + 0.45 * (internet == "fiber")
        + 0.30 * support_calls
        + 0.25 * (payment == "electronic-check")
        + rng.normal(0, 0.6, n_customers)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n_customers) < prob).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "support_calls": support_calls,
            "churn": churn,
        }
    )


if __name__ == "__main__":
    df = generate()
    print(df.shape)
    print("churn rate:", round(df["churn"].mean(), 4))
    print(df.head())
