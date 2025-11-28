
from src_PJ.config_PJ import SEED
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np
import pandas as pd


def get_regression_models():
    """
    Zwraca słownik modeli bazowych do regresji.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),  # małe alpha, żeby nie wyzerował wszystkiego
        "KNNRegressor": KNeighborsRegressor(n_neighbors=5),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            random_state=SEED,
            n_jobs=-1),
        "XGBRegressor": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror"
        )
    }
    return models


def train_and_evaluate_regression_models(X_train, X_test, y_train, y_test):
    """
    Trenuje modele regresji i zwraca tabelę metryk.

    Parametry:
    ----------
    X_train, X_test : numpy arrays
        Dane przetworzone (po preprocessorze)
    y_train, y_test : numpy arrays
        Target

    Zwraca:
    -------
    results_df : pd.DataFrame
        Tabela metryk modeli (MAE, RMSE, R2)
    """

    models = get_regression_models()
    results = []

    for name, model in models.items():
        # Uczenie
        model.fit(X_train, y_train)

        # Predykcja
        y_pred = model.predict(X_test)

        # Metryki
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    # Tabela wyników
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="RMSE")

    return results_df
