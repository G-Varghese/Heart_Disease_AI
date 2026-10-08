from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from preprocessing import split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = PROJECT_ROOT / "models" / "trained_models.joblib"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_subgroup_metrics(
    y_true,
    y_pred,
    y_prob,
):
    """Calculate classification metrics for one subgroup."""

    metrics = {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1-score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    # ROC-AUC requires both classes to be present.
    try:
        metrics["ROC-AUC"] = roc_auc_score(
            y_true,
            y_prob,
        )
    except ValueError:
        metrics["ROC-AUC"] = float("nan")

    return metrics


def main():

    # --------------------------------------------------
    # Load trained Logistic Regression model
    # --------------------------------------------------

    model_bundle = joblib.load(MODEL_FILE)

    logistic_model = model_bundle["logistic_model"]
    preprocessor = model_bundle["preprocessor"]
    feature_names = model_bundle["feature_names"]

    # --------------------------------------------------
    # Recreate the same train-test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = split_data()

    # --------------------------------------------------
    # Apply the fitted preprocessing pipeline
    # --------------------------------------------------

    X_test_processed = preprocessor.transform(X_test)

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    # --------------------------------------------------
    # Generate Logistic Regression predictions
    # --------------------------------------------------

    y_pred = logistic_model.predict(
        X_test_processed
    )

    y_prob = logistic_model.predict_proba(
        X_test_processed
    )[:, 1]

    # --------------------------------------------------
    # 1. SEX SUBGROUP ANALYSIS
    # --------------------------------------------------

    sex_results = []

    for sex_value in sorted(
        X_test["sex"].unique()
    ):

        mask = X_test["sex"] == sex_value

        y_true_group = y_test.loc[mask]
        y_pred_group = y_pred[mask.to_numpy()]
        y_prob_group = y_prob[mask.to_numpy()]

        metrics = calculate_subgroup_metrics(
            y_true_group,
            y_pred_group,
            y_prob_group,
        )

        metrics["Sex"] = int(sex_value)
        metrics["Sample Size"] = int(mask.sum())

        sex_results.append(metrics)

    sex_results_df = pd.DataFrame(
        sex_results
    )[
        [
            "Sex",
            "Sample Size",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
        ]
    ]

    print("\nSex Subgroup Results")
    print("--------------------")
    print(
        sex_results_df.to_string(
            index=False
        )
    )

    sex_results_df.to_csv(
        TABLES_DIR / "sex_subgroup_results.csv",
        index=False,
    )

    # --------------------------------------------------
    # 2. AGE SUBGROUP ANALYSIS
    # --------------------------------------------------

    X_test_age = X_test.copy()

    X_test_age["age_group"] = pd.cut(
        X_test_age["age"],
        bins=[0, 49, 59, 100],
        labels=["<50", "50-59", "60+"],
    )

    age_results = []

    for age_group in (
        X_test_age["age_group"]
        .dropna()
        .unique()
    ):

        mask = (
            X_test_age["age_group"]
            == age_group
        )

        y_true_group = y_test.loc[mask]
        y_pred_group = y_pred[mask.to_numpy()]
        y_prob_group = y_prob[mask.to_numpy()]

        metrics = calculate_subgroup_metrics(
            y_true_group,
            y_pred_group,
            y_prob_group,
        )

        metrics["Age Group"] = str(age_group)
        metrics["Sample Size"] = int(mask.sum())

        age_results.append(metrics)

    age_results_df = pd.DataFrame(
        age_results
    )[
        [
            "Age Group",
            "Sample Size",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
        ]
    ]

    print("\nAge Subgroup Results")
    print("--------------------")
    print(
        age_results_df.to_string(
            index=False
        )
    )

    age_results_df.to_csv(
        TABLES_DIR / "age_subgroup_results.csv",
        index=False,
    )

    # --------------------------------------------------
    # 3. FALSE-NEGATIVE ANALYSIS
    # --------------------------------------------------

    false_negative_mask = (
        (y_test == 1)
        & (y_pred == 0)
    )

    false_negatives = int(
        false_negative_mask.sum()
    )

    actual_positive = int(
        (y_test == 1).sum()
    )

    false_negative_rate = (
        false_negatives / actual_positive
    )

    false_negative_results = pd.DataFrame(
        [
            {
                "False Negatives": false_negatives,
                "Actual Positive Cases": actual_positive,
                "False Negative Rate": round(
                    false_negative_rate,
                    4,
                ),
            }
        ]
    )

    print("\nFalse-Negative Analysis")
    print("-----------------------")
    print(
        "False negatives:",
        false_negatives,
    )
    print(
        "Actual positive cases:",
        actual_positive,
    )
    print(
        "False-negative rate:",
        round(
            false_negative_rate,
            4,
        ),
    )

    false_negative_results.to_csv(
        TABLES_DIR / "false_negative_analysis.csv",
        index=False,
    )

    print(
        "\nResponsible AI analysis completed successfully."
    )

    print(
        f"Results saved to: {TABLES_DIR}"
    )


if __name__ == "__main__":
    main()