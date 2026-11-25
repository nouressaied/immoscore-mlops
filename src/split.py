from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def load_params():
    with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_data():
    params = load_params()

    test_size = params["split"]["test_size"]
    n_bins = params["split"]["n_bins"]
    seed = params["split"]["seed"]

    input_path = ROOT / "data" / "processed" / "dataset.csv"
    output_dir = ROOT / "data" / "split"

    # Charger les données préparées
    df = pd.read_csv(input_path)

    # Séparer les variables explicatives et la cible
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    # Créer les catégories de prix pour la stratification
    price_bins = pd.qcut(
        y,
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    # Séparer les données
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=price_bins
    )

    # Enregistrer les quatre ensembles
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print("Dimensions initiales :", df.shape)
    print("X_train :", X_train.shape)
    print("y_train :", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)
    print("Seed :", seed)
    print("Découpage stratifié terminé.")


def main():
    split_data()


if __name__ == "__main__":
    main()