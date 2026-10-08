from __future__ import annotations

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def load_dataset():
    """
    Load the Cleveland Heart Disease dataset from the UCI
    Machine Learning Repository.
    """
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features.copy()
    y_original = heart_disease.data.targets["num"].copy()

    return X, y_original


def create_binary_target(y_original):
    """
    Convert the original target:
        0     -> 0 (No disease)
        1-4   -> 1 (Disease)
    """
    return (y_original > 0).astype(int)


def split_data():
    """
    Load the dataset, create the binary target, and perform
    a stratified 80/20 train-test split.
    """
    X, y_original = load_dataset()

    y = create_binary_target(y_original)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor():
    """
    Create a leakage-safe preprocessing pipeline.

    Numerical features:
        Median imputation -> StandardScaler

    Categorical features:
        Most-frequent imputation -> One-hot encoding
    """

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def preprocess_training_data(X_train, X_test):
    """
    Fit preprocessing using training data only and then
    transform both training and testing data.
    """

    preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

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

    return (
        X_train_processed,
        X_test_processed,
        preprocessor,
        feature_names,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_data()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting target distribution:")
    print(y_test.value_counts().sort_index())

    (
        X_train_processed,
        X_test_processed,
        preprocessor,
        feature_names,
    ) = preprocess_training_data(
        X_train,
        X_test,
    )

    print("\nProcessed X_train shape:", X_train_processed.shape)
    print("Processed X_test shape:", X_test_processed.shape)
    print("Number of processed features:", len(feature_names))