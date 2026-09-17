"""Train and evaluate the churn classifier. Prints real metrics."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import generate

NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
CAT = ["contract", "internet_service", "payment_method"]


def build_pipeline():
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ]
    )
    clf = HistGradientBoostingClassifier(random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    df = generate(5000)
    X, y = df.drop(columns=["churn"]), df["churn"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="roc_auc")
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)[:, 1]
    print(f"rows={len(df)} churn_rate={y.mean():.4f}")
    print(f"cv_roc_auc={cv.mean():.4f} +- {cv.std():.4f}")
    print(f"accuracy={accuracy_score(yte, pred):.4f}")
    print(f"precision={precision_score(yte, pred):.4f}")
    print(f"recall={recall_score(yte, pred):.4f}")
    print(f"f1={f1_score(yte, pred):.4f}")
    print(f"roc_auc={roc_auc_score(yte, proba):.4f}")
    print(classification_report(yte, pred, digits=4))


if __name__ == "__main__":
    main()
