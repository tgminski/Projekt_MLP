# cars_01_load_data_PJ.py

import pandas as pd
from src_PJ.config_PJ import RAW_DIR

def load_cars_data():
    """Load cars.csv."""
    path = RAW_DIR / "cars.csv"
    df = pd.read_csv(path)

    print("\nFile loaded from:", path)

    print("\n=== Head ===")
    print(df.head())

    # return df