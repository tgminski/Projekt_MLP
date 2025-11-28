# cars_11_pca_PJ.py

import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def run_pca_analysis(
    input_path: Path,
    target_col: str,
    n_components: int = None
):
    """
    Wykonuje PCA na zbiorze po usunięciu outliers
    i generuje wykres Explained Variance.

    Parametry:
    ----------
    input_path : Path
        Ścieżka do oczyszczonego pliku CSV.
    target_col : str
        Nazwa kolumny targetu.
    n_components : int lub None
        Ile komponentów PCA obliczać (None = wszystkie).

    Zwraca:
    -------
    pca : PCA
    X_pca : ndarray
        Przetransformowane dane PCA.
    explained_var : ndarray
        Wariancja wyjaśniona przez komponenty.
    y : pd.Series
        Target (niezmieniony).
    """

    # 1. Wczytanie oczyszczonego zbioru
    df = pd.read_csv(input_path)

    # 2. Oddzielanie targetu
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 3. PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    explained_var = pca.explained_variance_ratio_

    # 4. Wykres wyjaśnionej wariancji
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(explained_var) + 1), explained_var * 100)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.title("PCA – Explained Variance")
    plt.tight_layout()
    plt.show()

    return pca, X_pca, explained_var, y
