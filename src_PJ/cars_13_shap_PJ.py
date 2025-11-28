# cars_13_shap_PJ.py
"""
SHAP analysis for the final Optuna-tuned XGBoost regression model.

Założenia:
- finalny pipeline został zapisany w:
    models_PJ/regression/optuna_final_pipeline_regression.pkl
  przez funkcję train_best_model() w cars_12_optuna_regression_PJ.py

- dane wejściowe (po EDA + FE + OHE + czyszczeniu Isolation Forest):
    data_PJ/processed/cars_feature_engineered_ohe_clean_if.csv

Ten moduł:
1) wczytuje dane X, y,
2) wczytuje wytrenowany pipeline (preprocessor + model),
3) liczy proste metryki na całym zbiorze,
4) oblicza wartości SHAP dla całego X,
5) zwraca pipeline, X, y, shap_values, metrics do użycia w notebooku.
"""

from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src_PJ.config_PJ import PROCESSED_DIR, MODELS_REGRESSION_DIR


# === ŚCIEŻKI DO PLIKÓW ===
INPUT_PATH = PROCESSED_DIR / "cars_feature_engineered_ohe_clean_if.csv"
PIPELINE_PATH = MODELS_REGRESSION_DIR / "optuna_final_pipeline_regression.pkl"



# 1. Wczytanie danych X, y


def load_data(input_path: Path = INPUT_PATH) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Wczytuje dane z CSV i rozdziela na X (features) i y (target='mpg').

    Zwraca:
    -------
    X : pd.DataFrame
    y : np.ndarray
    """
    df = pd.read_csv(input_path)

    if "mpg" not in df.columns:
        raise ValueError("Oczekuję kolumny targetu 'mpg' w DataFrame.")

    y = df["mpg"].values
    X = df.drop(columns=["mpg"])

    print(f"[INFO] Loaded data from: {input_path}")
    print(f"[INFO] Shape X: {X.shape}, y: {y.shape}")

    return X, y



# 2. Wczytanie wytrenowanego pipeline'u (Optuna + XGBRegressor)


def load_trained_pipeline(pipeline_path: Path = PIPELINE_PATH):
    """
    Wczytuje wytrenowany pipeline zapisany przez Optunę.

    Pipeline musi zawierać kroki:
    - 'prep'  → ColumnTransformer (preprocessing)
    - 'model' → XGBRegressor
    """
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono wytrenowanego pipeline'u: {pipeline_path}\n"
            f"Najpierw uruchom cars_12_optuna_regression_PJ.train_best_model(), "
            f"żeby zapisać finalny model."
        )

    pipeline = joblib.load(pipeline_path)
    print(f"[INFO] Loaded trained pipeline from: {pipeline_path}")

    # mała kontrola
    if "prep" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError(
            "Pipeline musi zawierać kroki 'prep' (preprocessor) i 'model' (XGBRegressor)."
        )

    return pipeline



# 3. Metryki dla całego zbioru (pomocniczo)


def compute_basic_metrics(pipeline, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
    """
    Liczy MAE, RMSE, R2 dla wczytanego pipeline'u na CAŁYM zbiorze X, y.

    Uwaga:
    - To NIE jest klasyczny wynik testowy (train/test split),
      tylko ocena na pełnym zbiorze (bardziej informacyjnie).
    """
    y_pred = pipeline.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    metrics = {
        "MAE_all": mae,
        "RMSE_all": rmse,
        "R2_all": r2
    }

    print("\n[INFO] Basic metrics of loaded pipeline on ENTIRE dataset:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics



# 4. Obliczanie wartości SHAP (TreeExplainer na XGBRegressor)


def compute_shap_values(pipeline, X: pd.DataFrame):
    """
    Oblicza wartości SHAP dla całego X, używając:
    - pipeline.named_steps['prep']  → transformacja cech,
    - pipeline.named_steps['model'] → XGBRegressor,
    - shap.TreeExplainer(model)     → wartości Shapleya.

    Zwraca:
    -------
    shap_values : shap._explanation.Explanation
        Wartości SHAP dla wszystkich obserwacji.
    """
    model = pipeline.named_steps.get("model", None)
    prep = pipeline.named_steps.get("prep", None)

    if model is None or prep is None:
        raise ValueError(
            "Pipeline musi zawierać kroki 'prep' (preprocessor) i 'model' (XGBRegressor)."
        )

    # Transformacja cech tak, jak widzi je model (skalowanie, passthrough OHE itd.)
    X_transformed = prep.transform(X)

    print("[INFO] Building SHAP TreeExplainer for XGBRegressor...")
    explainer = shap.TreeExplainer(model)

    print("[INFO] Computing SHAP values for all samples...")
    shap_values = explainer(X_transformed)

    print("[INFO] SHAP values computed. Shape:", np.array(shap_values.values).shape)

    return shap_values



# 5. Główna funkcja – uruchomienie całej analizy SHAP


def run_shap_for_optuna_model(
    input_path: Path = INPUT_PATH,
    pipeline_path: Path = PIPELINE_PATH,
):
    """
    Główna funkcja:
    1) Wczytuje dane X, y,
    2) Wczytuje wytrenowany pipeline z Optuny,
    3) Liczy proste metryki na całym zbiorze,
    4) Oblicza wartości SHAP dla całego X.

    Zwraca:
    -------
    pipeline  : wytrenowany pipeline (prep + model)
    X         : pd.DataFrame z cechami
    y         : np.ndarray z targetem
    shap_vals : Explanation (wartości SHAP)
    metrics   : dict z MAE_all, RMSE_all, R2_all
    """
    print("[INFO] === SHAP analysis for Optuna XGBRegressor ===")

    # 1. Dane
    X, y = load_data(input_path)

    # 2. Pipeline
    pipeline = load_trained_pipeline(pipeline_path)

    # 3. Metryki pomocnicze
    metrics = compute_basic_metrics(pipeline, X, y)

    # 4. Wartości SHAP
    shap_values = compute_shap_values(pipeline, X)

    print("\n[INFO] Done. You can now create SHAP plots, e.g.:")
    print("  import shap")
    print("  shap.summary_plot(shap_values, X)")
    print("  shap.summary_plot(shap_values, X, plot_type='bar')")

    return pipeline, X, y, shap_values, metrics



# 6. Uruchamianie jako skrypt


if __name__ == "__main__":
    # Pozwala uruchomić:
    #   python -m src_PJ.cars_13_shap_PJ
    # lub
    #   python src_PJ/cars_13_shap_PJ.py
    run_shap_for_optuna_model()
