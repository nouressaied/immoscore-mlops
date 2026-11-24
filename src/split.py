
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

    input_path = ROOT / "data" / "processed" / "data.csv"
    output_dir = ROOT / "data" / "processed"

    # Charger les données préparées
    df = pd.read_csv(input_path)

    # Créer les catégories de prix pour la stratification
    price_bins = pd.qcut(
        df["SalePrice"],
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    # Séparer les données
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=price_bins
    )

    # Enregistrer les ensembles
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Dimensions initiales :", df.shape)
    print("Train :", train_df.shape)
    print("Test :", test_df.shape)
    print("Seed :", seed)
    print("Découpage stratifié terminé.")


def main():
    split_data()


if __name__ == "__main__":
    main()
