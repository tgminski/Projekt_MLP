# cars_12_optuna_regression_PJ.py

"""
Regresja LASSO – Optuna + zapis modelu i parametrów do models_PJ/regression/.

Zawiera dwie funkcje:
1) run_optuna_lasso(input_path, n_trials=50)
      → zwraca (study, preprocessor)

2) train_best_model(input_path, best_params, preprocessor)
      → zwraca (pipeline, metrics)
"""

import json
from pathlib import Path
from typing import Dict, Any

import optuna
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src_PJ.config_PJ import MODELS_REGRESSION_DIR, SEED
from src_PJ.cars_06_preprocessing_PJ import prepare_data_for_ml


# =====================================================================
# 1. Wczytanie danych
# =====================================================================

def load_data(input_path: Path):
    df = pd.read_csv(input_path)
    X, y, preprocessor, _ = prepare_data_for_ml(df, target_col="mpg")
    return X, y, preprocessor


# =====================================================================
# 2. Funkcja celu dla Optuny – LASSO
# =====================================================================

def objective(trial, X, y, preprocessor):

    params = {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 3000, 10000)
    }

    model = Lasso(**params, random_state=SEED)

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )

    return -scores.mean()


# =====================================================================
# 3. Uruchomienie Optuny
# =====================================================================

def run_optuna_lasso(input_path: Path, n_trials=50):

    X, y, preprocessor = load_data(input_path)

    print("\n[INFO] Starting LASSO Optuna optimization…")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X, y, preprocessor),
        n_trials=n_trials
    )

    print("\n[INFO] Best params:", study.best_params)
    print("[INFO] Best RMSE:", study.best_value)

    return study, preprocessor


# =====================================================================
# 4. Trenowanie finalnego modelu + zapis
# =====================================================================

def train_best_model(input_path: Path, best_params: Dict[str, Any], preprocessor):

    df = pd.read_csv(input_path)
    X = df.drop("mpg", axis=1)
    y = df["mpg"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED
    )

    model = Lasso(
        **best_params,
        random_state=SEED
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }


    MODELS_REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_PIPELINE = MODELS_REGRESSION_DIR / "optuna_final_pipeline_regression.pkl"
    OUTPUT_PARAMS = MODELS_REGRESSION_DIR / "optuna_best_params_regression.json"

    joblib.dump(pipeline, OUTPUT_PIPELINE)

    with open(OUTPUT_PARAMS, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"[INFO] Saved final LASSO pipeline → {OUTPUT_PIPELINE}")
    print(f"[INFO] Saved Optuna params → {OUTPUT_PARAMS}")
    print("[INFO] LASSO regression model saved correctly.\n")

    return pipeline, metrics





#-----------------------

