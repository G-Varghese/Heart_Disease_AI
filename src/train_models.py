from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from preprocessing import split_data, preprocess_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


def train_models():
    """
    Train Logistic Regression and Random Forest models
    using the processed training data.
    """

    # Load and split the dataset
    X_train, X_test, y_train, y_test = split_data()

    # Preprocess using training data only
    (
        X_train_processed,
        X_test_processed,
        preprocessor,
        feature_names,
    ) = preprocess_training_data(
        X_train,
        X_test,
    )

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    logistic_model.fit(
        X_train_processed,
        y_train,
    )

    # -----------------------------
    # Random Forest
    # -----------------------------
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    random_forest_model.fit(
        X_train_processed,
        y_train,
    )

    # Store everything needed later
    model_bundle = {
        "logistic_model": logistic_model,
        "random_forest_model": random_forest_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }

    model_file = MODEL_DIR / "trained_models.joblib"

    joblib.dump(
        model_bundle,
        model_file,
    )

    print("Models trained successfully.")
    print(f"Saved model file: {model_file}")


if __name__ == "__main__":
    train_models()