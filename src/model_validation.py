#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/04_aggregated/aggregated_dataframe.csv"
TARGET_COL = "TenYearCHD"
RANDOM_STATE = 42


def main():
    print(">>> Starting model validation on aggregated data")

    # Load aggregated dataset
    df = pd.read_csv(DATA_PATH)

    # Split features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Preprocessing (all features already numeric)
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    models = {
        "LogisticRegression": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ]),
        "RandomForest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample"
            ))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n=== Cross-validation results ===")
    results = {}

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=["roc_auc", "recall", "accuracy"],
            n_jobs=-1
        )

        results[name] = {
            "roc_auc": scores["test_roc_auc"].mean(),
            "recall": scores["test_recall"].mean(),
            "accuracy": scores["test_accuracy"].mean()
        }

        print(
            f"{name}: "
            f"ROC-AUC={results[name]['roc_auc']:.3f}, "
            f"Recall={results[name]['recall']:.3f}, "
            f"Accuracy={results[name]['accuracy']:.3f}"
        )

    # Select best model by ROC-AUC
    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = models[best_model_name]

    print(f"\nSelected best model: {best_model_name}")

    # Final evaluation on test set
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n=== Holdout test results ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # Save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")

    print("\nModel saved to models/best_model.joblib")


if __name__ == "__main__":
    main()
