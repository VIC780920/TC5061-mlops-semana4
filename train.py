import json
import mlflow
import mlflow.sklearn
import argparse
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(path):
    """Carga el dataset Wine Quality."""
    return pd.read_csv(path, sep=";")


def preprocess_data(df):
    """Elimina registros duplicados y separa variables."""
    df = df.drop_duplicates().copy()

    X = df.drop(columns=["quality"])
    y = df["quality"]

    return X, y


def train_model(X_train, y_train, args):
    """Entrena el modelo con los hiperparámetros indicados."""
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
        n_jobs=1
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Calcula las métricas de evaluación."""
    predictions = model.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento reproducible - Wine Quality"
    )

    parser.add_argument("--data", type=str, default="winequality-red.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    df = load_data(args.data)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.random_state
    )

    mlflow.set_tracking_uri(
        f"sqlite:///{Path(__file__).resolve().parent / 'mlflow.db'}"
    )
    mlflow.set_experiment("Wine_Quality_RF_Experiments")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", 0.20)

        model = train_model(X_train, y_train, args)
        metrics = evaluate_model(model, X_test, y_test)

        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        artifact_dir = Path(__file__).resolve().parent / "outputs"
        artifact_dir.mkdir(exist_ok=True)

        metrics_file = artifact_dir / "metrics.json"
        metrics_file.write_text(
            json.dumps(
                {k: float(v) for k, v in metrics.items()},
                indent=4
            ),
            encoding="utf-8"
        )

        mlflow.log_artifact(str(metrics_file))

        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

    print("Modelo entrenado correctamente")
    print("Registros utilizados:", len(X))
    print("Hiperparámetros:", {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "random_state": args.random_state
    })

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
