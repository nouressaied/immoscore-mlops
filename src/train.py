from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]


def train_model():

    X_train_path = ROOT / "data" / "split" / "X_train.csv"
    y_train_path = ROOT / "data" / "split" / "y_train.csv"

    model_path = ROOT / "models" / "model.pkl"

    # Charger les données d'entraînement
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)["SalePrice"]

    # Pipeline de prédiction avec imputation
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regression", LinearRegression())
    ])

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Enregistrer le modèle et la liste des variables
    artifact = {
        "model": model,
        "features": X_train.columns.tolist()
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, model_path)

    print("Entraînement terminé.")
    print("Nombre d'observations :", len(X_train))
    print("Variables utilisées :", X_train.columns.tolist())
    print("Nombre de variables :", X_train.shape[1])
    print("Modèle enregistré :", model_path)


def main():
    train_model()


if __name__ == "__main__":
    main()