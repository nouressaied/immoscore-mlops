
from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]


def load_params():
    with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_model():
    params = load_params()
    threshold = params["process"]["corr_threshold"]

    train_path = ROOT / "data" / "processed" / "train.csv"
    model_path = ROOT / "models" / "model.joblib"

    df = pd.read_csv(train_path)

    X_train = df.drop(columns=["SalePrice"])
    y_train = df["SalePrice"]

    # Imputation apprise uniquement sur Train
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    # Corrélations calculées uniquement sur Train
    correlations = X_imputed.corrwith(y_train).abs()

    selected_features = correlations[
        correlations >= threshold
    ].index.tolist()

    if not selected_features:
        raise ValueError("Aucune variable sélectionnée.")

    # Pipeline de prédiction avec imputation
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regression", LinearRegression())
    ])

    model.fit(X_train[selected_features], y_train)

    # Enregistrer le modèle et la liste des variables
    artifact = {
        "model": model,
        "features": selected_features
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    print("Entraînement terminé.")
    print("Nombre d'observations :", len(X_train))
    print("Variables sélectionnées :", selected_features)
    print("Nombre de variables :", len(selected_features))
    print("Modèle enregistré :", model_path)


def main():
    train_model()


if __name__ == "__main__":
    main()
