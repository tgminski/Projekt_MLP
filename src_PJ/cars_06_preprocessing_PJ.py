

"""
Moduł do przygotowania danych do dalszych kroków ML.
- funkcję rozdzielającą X i y,
- funkcję wykrywającą cechy numeryczne i kategoryczne (OHE),
- budowę preprocessora (StandardScaler + passthrough),
- główną funkcję prepare_data_for_ml().
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# 2. ----- split_features_target -----


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "mpg"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Rozdziela dane na:
    - X (wszystkie kolumny poza targetem)
    - y (kolumna targetu np. mpg)

    Parametry:
    ----------
    df : pd.DataFrame
        Dane wejściowe po EDA + FE + OHE.
    target_col : str
        Nazwa kolumny targetu.

    Zwraca:
    -------
    X : pd.DataFrame
    y : numpy array
    """

    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Kolumna targetu '{target_col}' nie istnieje w DataFrame.")

    y = df[target_col].values
    X = df.drop(columns=[target_col])

    return X, y


# 3. ----- detect_numeric_and_categorical_features -----

def detect_numeric_and_categorical_features(
    X: pd.DataFrame,
    origin_prefix: str = "origin_"
) -> Dict[str, List[str]]:
    """
    Automatycznie wykrywa:
    - cechy numeryczne (int/float)
    - cechy kategoryczne po One-Hot Encoding (kolumny zaczynające się na origin_)

    Parametry:
    ----------
    X : pd.DataFrame
        Dane wejściowe bez kolumny targetu.
    origin_prefix : str
        Prefiks dla kolumn OHE (domyślnie origin_)

    Zwraca:
    -------
    dict:
        {
            "numeric_features": [...],
            "categorical_features": [...]
        }
    """

    # wszystkie kolumny numeryczne
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    # kolumny po One-Hot Encoding origin
    categorical_features = [col for col in X.columns if col.startswith(origin_prefix)]

    # upewniamy się, że kolumny OHE nie wpadają do numeric_features
    numeric_features = [col for col in numeric_features if col not in categorical_features]

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }


# 4. ----- build_preprocessor -----


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Tworzy obiekt ColumnTransformer:
    - StandardScaler dla cech numerycznych
    - passthrough dla cech kategorycznych (OHE)

    Zwraca:
    -------
    preprocessor : ColumnTransformer
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", "passthrough", categorical_features),
        ]
    )

    return preprocessor


# 5. ----- prepare_data_for_ml -----


def prepare_data_for_ml(
    df: pd.DataFrame,
    target_col: str = "mpg",
    origin_prefix: str = "origin_"
) -> Tuple[pd.DataFrame, np.ndarray, ColumnTransformer, Dict[str, List[str]]]:
    """
    Główna funkcja łącząca:
    1. Rozdzielenie danych na X i y
    2. Wykrycie cech numerycznych i kategorycznych
    3. Budowę preprocessora (StandardScaler + passthrough)

    Zwraca:
    -------
    X : pd.DataFrame
        Dane wejściowe bez targetu.
    y : numpy array
        Target.
    preprocessor : ColumnTransformer
        Gotowy preprocessor (jeszcze bez fitowania).
    feature_info : dict
        {
            "numeric_features": [...],
            "categorical_features": [...]
        }
    """

    # 1. Rozdzielenie X i y
    X, y = split_features_target(df, target_col=target_col)

    # 2. Wykrycie cech

    feature_info = detect_numeric_and_categorical_features(
        X,
        origin_prefix=origin_prefix
    )

    # 3. Budowa preprocessora
    preprocessor = build_preprocessor(
        numeric_features=feature_info["numeric_features"],
        categorical_features=feature_info["categorical_features"]
    )

    return X, y, preprocessor, feature_info

# Funkcje preprocessingu nic nie zapisują do CSV —  przygotowują obiekty potrzebne do dalszych kroków ML.
# CSV powstaną dopiero w późniejszych etapach, takich jak Isolation Forest, PCA, train/test split czy SHAP.