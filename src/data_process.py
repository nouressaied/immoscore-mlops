from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_params():
    with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_data():
    params = load_params()
    threshold = params["process"]["corr_threshold"]

    input_path = ROOT / "data" / "raw" / "data.csv"
    output_path = ROOT / "data" / "processed" / "dataset.csv"

    df = pd.read_csv(input_path)

    print("Dimensions initiales :", df.shape)

    # Conserver les variables numériques
    df = df.select_dtypes(include="number")

    if "SalePrice" not in df.columns:
        raise ValueError("La colonne SalePrice est absente.")

    # Supprimer les colonnes entièrement vides
    df = df.dropna(axis=1, how="all")

    # Calculer les corrélations avec la variable cible
    correlations = df.corr()["SalePrice"].abs()

    # Sélectionner les variables dépassant le seuil
    selected_features = correlations[
        correlations > threshold
    ].index.tolist()

    selected_features = [
        col for col in selected_features
        if col != "SalePrice"
    ]

    if not selected_features:
        raise ValueError("Aucune variable sélectionnée.")

    # Conserver les variables sélectionnées et la cible
    df = df[selected_features + ["SalePrice"]]

    # Enregistrer le dataset préparé
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Seuil de corrélation :", threshold)
    print("Variables sélectionnées :", selected_features)
    print("Nombre de variables :", len(selected_features))
    print("Dimensions finales :", df.shape)
    print("Fichier enregistré :", output_path)


def main():
    process_data()


if __name__ == "__main__":
    main()