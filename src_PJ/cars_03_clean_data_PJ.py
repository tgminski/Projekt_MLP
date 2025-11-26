# cars_03_clean_data_PJ.py

import pandas as pd
from src_PJ.config_PJ import RAW_CARS, INTERIM_DIR, DATA_DIR


def clean_data_remove_3_cylinders():
    """Load raw data, remove cars with 3 cylinders, reset index, save cleaned file."""

    # 1. load raw dataset
    print(f"Loading raw data from: {RAW_CARS}")
    df = pd.read_csv(RAW_CARS)

    # 2. remove cars with 3 cylinders
    print("[INFO] Removing cars with 3 cylinders...")
    df_cleaned = df[df["cylinders"] != 3].reset_index(drop=True)

    # 3. prepare output path
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DIR / "cars_no_3cylinders.csv"

    # 4. save cleaned data
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")

    print("\n-----shape: cars_no_3cylinders.csv -----")
    df = pd.read_csv(INTERIM_DIR / "cars_no_3cylinders.csv")
    print(df.shape)

    # return df_cleaned

# if __name__ == "__main__":
#     clean_data_remove_3_cylinders()