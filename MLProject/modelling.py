"""Train an Iris classifier as an MLflow Project entry point."""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR.parent / "mlartifacts"
DATASET_PATH = PROJECT_DIR / "dataset_preprocessing.csv"
EXPERIMENT_NAME = "iris-workflow-ci"
FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
TARGET_NAMES = ["setosa", "versicolor", "virginica"]


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

    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=ARTIFACT_DIR.resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=True, log_input_examples=False)

    data = pd.read_csv(DATASET_PATH)
    required_columns = {*FEATURE_COLUMNS, "target", "species"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    X = data[FEATURE_COLUMNS]
    y = data["target"].astype(int)
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
                LogisticRegression(max_iter=1000, random_state=args.random_state),
            ),
        ]
    )

    logger.info(
        "Dataset Iris dimuat dari %s: %d baris, %d fitur.",
        DATASET_PATH.name,
        *X.shape,
    )
    logger.info(
        "Data dibagi: %d data latih dan %d data uji.",
        len(X_train),
        len(X_test),
    )

    with mlflow.start_run(run_name="logistic-regression-iris-ci") as run:
        mlflow.log_params(
            {
                "dataset": "Iris",
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "test_size": args.test_size,
                "random_state": args.random_state,
                "preprocessing": "median imputation + standard scaling",
            }
        )
        mlflow.log_artifact(str(DATASET_PATH))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("test_accuracy", test_accuracy)

        logger.info("Model selesai dilatih.")
        logger.info("Akurasi data uji: %.4f", test_accuracy)
        logger.info(
            "Classification report:\n%s",
            classification_report(
                y_test,
                predictions,
                target_names=TARGET_NAMES,
                zero_division=0,
            ),
        )
        logger.info("MLflow run ID: %s", run.info.run_id)


if __name__ == "__main__":
    main()
