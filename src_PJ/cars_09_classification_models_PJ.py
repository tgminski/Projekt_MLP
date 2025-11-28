# src_PJ/cars_09_classification_models_PJ.py
from src_PJ.config_PJ import SEED
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


def get_classification_models():
    """
    Zwraca zestaw prostych modeli baseline do klasyfikacji.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, random_state=SEED
        ),
        "KNNClassifier": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(),
        "XGBClassifier": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            eval_metric="mlogloss"
        )
    }
    return models


def train_and_evaluate_classification_models(X_train, X_test, y_train, y_test):
    """
    Trenuje modele klasyfikacyjne i zwraca tabelę metryk.
    """
    models = get_classification_models()
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_macro": f1
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_macro", ascending=False)

    return results_df
