from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap

from preprocessing import split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = PROJECT_ROOT / "models" / "trained_models.joblib"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load trained models and preprocessing information
    model_bundle = joblib.load(MODEL_FILE)

    logistic_model = model_bundle["logistic_model"]
    preprocessor = model_bundle["preprocessor"]
    feature_names = model_bundle["feature_names"]

    # Recreate the same train-test split
    X_train, X_test, y_train, y_test = split_data()

    # Transform the data using the fitted preprocessing pipeline
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert to DataFrames so feature names are preserved
    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    # --------------------------------------------------
    # SHAP Explainer
    # --------------------------------------------------

    explainer = shap.Explainer(
        logistic_model,
        X_train_processed,
    )

    shap_values = explainer(
        X_test_processed
    )

    print("SHAP analysis completed.")
    print("SHAP values shape:", shap_values.values.shape)
    print("Test data shape:", X_test_processed.shape)

    # --------------------------------------------------
    # Global SHAP feature importance
    # --------------------------------------------------

    shap_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean Absolute SHAP": (
                abs(shap_values.values).mean(axis=0)
            ),
        }
    ).sort_values(
        by="Mean Absolute SHAP",
        ascending=False,
    )

    print("\nTop 10 SHAP features:")
    print(
        shap_importance.head(10).to_string(
            index=False
        )
    )

    shap_importance.to_csv(
        TABLES_DIR / "shap_feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------
    # SHAP global bar plot
    # --------------------------------------------------

    shap.plots.bar(
        shap_values,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "shap_bar.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # SHAP beeswarm plot
    # --------------------------------------------------

    shap.plots.beeswarm(
        shap_values,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Individual patient explanation
    # --------------------------------------------------

    patient_index = 0

    patient_prediction = logistic_model.predict(
        X_test_processed.iloc[[patient_index]]
    )[0]

    patient_probability = logistic_model.predict_proba(
        X_test_processed.iloc[[patient_index]]
    )[0, 1]

    actual_class = y_test.iloc[patient_index]

    print("\nIndividual Patient Explanation")
    print("------------------------------")
    print("Patient index:", patient_index)
    print("Actual class:", actual_class)
    print("Predicted class:", patient_prediction)
    print(
        "Probability of heart disease:",
        round(patient_probability, 4),
    )

    print("\nPatient feature values:")
    print(X_test.iloc[patient_index])

    # Save patient prediction information
    patient_results = pd.DataFrame(
        [
            {
                "Patient Index": patient_index,
                "Actual Class": actual_class,
                "Predicted Class": patient_prediction,
                "Probability of Heart Disease": patient_probability,
            }
        ]
    )

    patient_results.to_csv(
        TABLES_DIR / "individual_patient_prediction.csv",
        index=False,
    )

    # --------------------------------------------------
    # SHAP waterfall plot
    # --------------------------------------------------

    patient_shap = shap_values[patient_index]

    shap.plots.waterfall(
        patient_shap,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "shap_waterfall_patient_0.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nSHAP figures and tables saved successfully.")


if __name__ == "__main__":
    main()