import pandas as pd
from src_PJ.config_PJ import RAW_DIR

def load_cars_data():
    """Load cars.csv."""
    path = RAW_DIR / "cars.csv"
    df = pd.read_csv(path)

    print("\nfile loaded from:", path)
    print("\n----- shape -----")
    print(f"number of lines: {df.shape[0]}")
    print(f"number of columns: {df.shape[1]}")

    print("\n----- columns in the dataset -----")
    print(df.columns.tolist())

    print("\n----- data frame -----")

    summary = pd.DataFrame({
        'Data types': df.dtypes,
        'Number of unique values': df.nunique(),
        'Number of missing values': df.isna().sum(),
        'Duplicates': df.duplicated().sum(),
    })

    print(summary)

    print("\n----- statistics of numerical features -----")
    print(df.describe().T)

    print("\n----- unique values in the 'cylinders' column -----")
    print(df['cylinders'].value_counts().sort_index())

    print("\n----- cars with 3 or 5 cylinders -----")
    df_cyl_3_5 = df[df['cylinders'].isin([3, 5])]
    print(df_cyl_3_5)



    # return df
load_cars_data()