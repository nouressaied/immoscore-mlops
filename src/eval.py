from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


ROOT = Path(__file__).resolve().parents[1]


def calculate_metrics(model, X, y):

    predictions = model.predict(X)

    return {
        "rmse": float(mean_squared_error(y, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions))
    }


def evaluate_model():

    # Charger le modèle et les variables sélectionnées
    model_path = ROOT / "models" / "model.pkl"
    artifact = joblib.load(model_path)

    model = artifact["model"]
    features = artifact["features"]

    # Charger les données Train
    X_train = pd.read_csv(
        ROOT / "data" / "split" / "X_train.csv"
    )

    y_train = pd.read_csv(
        ROOT / "data" / "split" / "y_train.csv"
    )["SalePrice"]

    # Charger les données Test
    X_test = pd.read_csv(
        ROOT / "data" / "split" / "X_test.csv"
    )

    y_test = pd.read_csv(
        ROOT / "data" / "split" / "y_test.csv"
    )["SalePrice"]

    # Respecter l'ordre des variables du modèle
    X_train = X_train[features]
    X_test = X_test[features]

    # Évaluer le modèle
    metrics = {
        "train": calculate_metrics(model, X_train, y_train),
        "test": calculate_metrics(model, X_test, y_test)
    }

    # Enregistrer les métriques
    output_path = ROOT / "reports" / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("===== EVALUATION DU MODELE =====")

    for dataset, values in metrics.items():
        print(f"\n=== {dataset.upper()} ===")
        print(f"RMSE : {values['rmse']:.2f}")
        print(f"MAE  : {values['mae']:.2f}")
        print(f"R2   : {values['r2']:.4f}")

    print("\nMétriques enregistrées :", output_path)


def main():
    evaluate_model()


if __name__ == "__main__":
    main()