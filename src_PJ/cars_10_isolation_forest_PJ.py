# cars_10_isolation_forest_PJ.py

import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from src_PJ.config_PJ import SEED

def run_isolation_forest(
    input_path: Path,
    output_path: Path,
    target_col: str,
    contamination: float = 0.05,
    random_state: int = SEED
) -> pd.DataFrame:

    # 1. Wczytanie danych
    df = pd.read_csv(input_path)

    # 2. Usuwamy kolumny nienumeryczne
    if "name" in df.columns:
        df = df.drop(columns=["name"])

    # Konwersja TRUE/FALSE → bool → int
    df = df.replace({"TRUE": True, "FALSE": False})
    df = df.astype({col: "int32" for col in df.select_dtypes(include="bool").columns})

    # 3. Oddzielenie targetu
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 4. Isolation Forest
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    preds = iso.fit_predict(X)

    mask = preds == 1
    df_clean = df[mask].reset_index(drop=True)

    # 5. Zapis
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    return df_clean