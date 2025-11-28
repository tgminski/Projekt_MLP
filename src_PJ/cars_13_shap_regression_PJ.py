# cars_13_shap_regression_PJ.py

import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src_PJ.cars_06_preprocessing_PJ import prepare_data_for_ml


# ---------------------------------------------------
# 1. Feature names (numeric + OHE)
# ---------------------------------------------------

def get_feature_names(df, target_col="mpg", origin_prefix="origin_"):
    """
    Tworzy listę nazw cech po preprocessingu.
    Kolejność: numeric_features + categorical_features.
    """
    _, _, _, feature_info = prepare_data_for_ml(df, target_col=target_col, origin_prefix=origin_prefix)

    numeric_features = feature_info["numeric_features"]
    categorical_features = feature_info["categorical_features"]

    feature_names = numeric_features + categorical_features
    return feature_names


# ---------------------------------------------------
# 2. Przygotowanie X_test i X_test_processed
# ---------------------------------------------------

def prepare_test_data(df, pipeline, target_col="mpg"):
    """
    Zwraca:
    - X_test (oryginalny)
    - y_test
    - X_test_processed (po preprocessorze)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # używamy preprocesora z pipeline
    X_test_processed = pipeline.named_steps["prep"].transform(X_test)

    return X_test, y_test, X_test_processed


# ---------------------------------------------------
# 3. SHAP explainer + SHAP values
# ---------------------------------------------------

def get_shap_values(pipeline, X_test_processed):
    """
    Zwraca:
    - explainer
    - shap_values
    """
    model = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_processed)

    return explainer, shap_values


# ---------------------------------------------------
# 4. Wykresy globalne (summary + bar)
# ---------------------------------------------------

def plot_shap_global(shap_values, X_test_processed, feature_names):
    """
    Tworzy globalne wykresy SHAP:
    - summary plot
    - bar plot
    """
    print(">> SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names)

    print(">> SHAP Bar Plot")
    shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, plot_type="bar")


# ---------------------------------------------------
# 5. Wykres lokalny (waterfall)
# ---------------------------------------------------

def plot_shap_local(explainer, shap_values, X_test_processed, feature_names, index=0):
    """
    Tworzy waterfall plot dla jednej próbki.
    """
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[index],
            base_values=explainer.expected_value,
            data=X_test_processed[index],
            feature_names=feature_names
        )
    )
