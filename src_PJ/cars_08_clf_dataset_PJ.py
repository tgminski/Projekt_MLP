

import pandas as pd
from pathlib import Path


def create_classification_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # 1. Tworzymy klasy
    df["mpg_class"] = pd.cut(
        df["mpg"],
        bins=[0, 20, 30, 100],
        labels=["low", "medium", "high"],
        right=False
    )

    # 2. Usuwamy mpg — BO TO JEST ŹRÓDŁO targetu, a jego obecność powoduje 100% accuracy
    df = df.drop(columns=["mpg"])

    # 3. Zapis
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df
