"""Train the breast cancer classifier as an MLflow Project entry point."""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR.parent / "mlartifacts"
DATASET_PATH = PROJECT_DIR / "dataset_preprocessing.csv"
EXPERIMENT_NAME = "breast-cancer-workflow-ci"
TARGET_NAMES = {0: "malignant", 1: "benign"}
EXPECTED_ROWS = 569


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(PROJECT_DIR / "modelling.log", encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    data = pd.read_csv(DATASET_PATH)
    required_columns = {"target", "diagnosis"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if len(data) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} Breast Cancer Wisconsin rows, found {len(data)}."
        )

    feature_columns = [column for column in data.columns if column not in required_columns]
    if len(feature_columns) != 30:
        raise ValueError(
            f"Expected 30 numeric features, found {len(feature_columns)}."
        )
    y = data["target"].astype(int)
    if set(y.unique()) != set(TARGET_NAMES):
        raise ValueError(f"Unexpected target values: {sorted(y.unique())}")
    expected_diagnoses = {0: "malignant", 1: "benign"}
    diagnoses_by_target = data.assign(target=y).groupby("target")["diagnosis"].unique()
    if any(
        len(diagnoses_by_target.get(target, [])) != 1
        or diagnoses_by_target[target][0] != diagnosis
        for target, diagnosis in expected_diagnoses.items()
    ):
        raise ValueError("Dataset diagnosis labels do not match target values.")

    X = data[feature_columns].apply(pd.to_numeric, errors="raise")

    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=ARTIFACT_DIR.resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=True, log_input_examples=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    logger.info(
        "Breast Cancer Wisconsin dataset loaded from %s: %d rows, %d features.",
        DATASET_PATH.name,
        *X.shape,
    )
    logger.info(
        "Stratified split: %d training rows and %d test rows.",
        len(X_train),
        len(X_test),
    )

    with mlflow.start_run(run_name="logistic-regression-breast-cancer-ci") as run:
        mlflow.log_params(
            {
                "dataset": "Breast Cancer Wisconsin (Diagnostic)",
                "dataset_source": (
                    "https://archive.ics.uci.edu/dataset/17/"
                    "breast+cancer+wisconsin+diagnostic"
                ),
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "test_size": args.test_size,
                "random_state": args.random_state,
                "preprocessing": "median imputation + standard scaling",
                "class_weight": "balanced",
            }
        )
        mlflow.log_artifact(str(DATASET_PATH))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_precision_macro": precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "test_recall_macro": recall_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "test_f1_macro": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
        }
        mlflow.log_metrics(metrics)

        logger.info("Model training completed.")
        logger.info("Test metrics: %s", metrics)
        logger.info(
            "Classification report:\n%s",
            classification_report(
                y_test,
                predictions,
                labels=list(TARGET_NAMES),
                target_names=list(TARGET_NAMES.values()),
                zero_division=0,
            ),
        )
        logger.info("MLflow run ID: %s", run.info.run_id)


if __name__ == "__main__":
    main()
