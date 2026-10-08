from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from preprocessing import split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = PROJECT_ROOT / "models" / "trained_models.joblib"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification performance metrics."""

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
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
        "ROC-AUC": roc_auc_score(
            y_true,
            y_prob,
        ),
    }


def main():
    # Load trained models and preprocessing
    model_bundle = joblib.load(MODEL_FILE)

    logistic_model = model_bundle["logistic_model"]
    random_forest_model = model_bundle["random_forest_model"]
    preprocessor = model_bundle["preprocessor"]

    # Recreate the same train-test split
    X_train, X_test, y_train, y_test = split_data()

    # Apply the already-fitted preprocessing pipeline
    feature_names = model_bundle["feature_names"]

    X_test_processed = preprocessor.transform(X_test)

    X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index,
)

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    y_pred_logistic = logistic_model.predict(
        X_test_processed
    )

    y_prob_logistic = logistic_model.predict_proba(
        X_test_processed
    )[:, 1]

    logistic_metrics = calculate_metrics(
        y_test,
        y_pred_logistic,
        y_prob_logistic,
    )

    # -----------------------------
    # Random Forest
    # -----------------------------
    y_pred_rf = random_forest_model.predict(
        X_test_processed
    )

    y_prob_rf = random_forest_model.predict_proba(
        X_test_processed
    )[:, 1]

    rf_metrics = calculate_metrics(
        y_test,
        y_pred_rf,
        y_prob_rf,
    )

    # -----------------------------
    # Model comparison table
    # -----------------------------
    results = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                **logistic_metrics,
            },
            {
                "Model": "Random Forest",
                **rf_metrics,
            },
        ]
    )

    results_rounded = results.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    results_rounded[metric_columns] = (
        results_rounded[metric_columns].round(4)
    )

    print("\nModel Performance Comparison")
    print(results_rounded.to_string(index=False))

    results_rounded.to_csv(
        TABLES_DIR / "model_results.csv",
        index=False,
    )

    # -----------------------------
    # Confusion matrices
    # -----------------------------
    logistic_cm = confusion_matrix(
        y_test,
        y_pred_logistic,
    )

    rf_cm = confusion_matrix(
        y_test,
        y_pred_rf,
    )

    print("\nLogistic Regression Confusion Matrix:")
    print(logistic_cm)

    print("\nRandom Forest Confusion Matrix:")
    print(rf_cm)

    pd.DataFrame(
        logistic_cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(
        TABLES_DIR / "logistic_confusion_matrix.csv"
    )

    pd.DataFrame(
        rf_cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(
        TABLES_DIR / "random_forest_confusion_matrix.csv"
    )

    # -----------------------------
    # ROC curve
    # -----------------------------
    fpr_logistic, tpr_logistic, _ = roc_curve(
        y_test,
        y_prob_logistic,
    )

    fpr_rf, tpr_rf, _ = roc_curve(
        y_test,
        y_prob_rf,
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr_logistic,
        tpr_logistic,
        label=(
            f"Logistic Regression "
            f"(AUC = {logistic_metrics['ROC-AUC']:.3f})"
        ),
    )

    plt.plot(
        fpr_rf,
        tpr_rf,
        label=(
            f"Random Forest "
            f"(AUC = {rf_metrics['ROC-AUC']:.3f})"
        ),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "roc_curve_comparison.png",
        dpi=300,
    )

    plt.close()

    # -----------------------------
    # Model comparison chart
    # -----------------------------
    plot_data = results.set_index("Model")[
        metric_columns
    ]

    ax = plot_data.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_ylim(0, 1)

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "model_performance_comparison.png",
        dpi=300,
    )

    plt.close()

    print("\nEvaluation completed successfully.")
    print(f"Results saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()