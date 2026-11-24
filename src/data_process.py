
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def process_data():
    input_path = ROOT / "data" / "raw" / "data.csv"
    output_path = ROOT / "data" / "processed" / "data.csv"

    df = pd.read_csv(input_path)

    print("Dimensions initiales :", df.shape)

    # Conserver les variables numériques et la cible
    df = df.select_dtypes(include="number")

    if "SalePrice" not in df.columns:
        raise ValueError("La colonne SalePrice est absente.")

    # Supprimer uniquement les colonnes entièrement vides
    df = df.dropna(axis=1, how="all")

    # Ne pas calculer de médianes ou de corrélations ici
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Dimensions finales :", df.shape)
    print("Fichier enregistré :", output_path)


def main():
    process_data()


if __name__ == "__main__":
    main()
