# 1. ----- Importy i przygotowanie danych -----

# --- Core data handling ---
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

# 2. ----- FEATURES THAT CREATE NEW CHARACTERISTICS -----


def add_power_to_weight(df, hp_col="horsepower", weight_col="weight"):
    df["power_to_weight"] = df[hp_col] / df[weight_col]
    return df

def add_displacement_per_cylinder(df, disp_col="displacement", cyl_col="cylinders"):
    df["displacement_per_cyl"] = df[disp_col] / df[cyl_col]
    return df

def add_acceleration_per_weight(df, acc_col="acceleration", weight_col="weight"):
    df["acceleration_per_weight"] = df[acc_col] / df[weight_col]
    return df

# -----------------------------------------------------
# # my example
# def add_log_feature(df, column):
#
#     import numpy as np
#
#     if (df[column] <= 0).any():
#         raise ValueError(f"Column '{column}' contains non-positive values. Cannot apply log.")
#
#     df[f"{column}_log"] = np.log(df[column])
#     print(f"Added log-transformed column: {column}_log")
#
#     return df
#
# # in a notebook
# from src_PJ.cars_04_exploratory_data_analysis import add_log_feature
# from src_PJ.config_PJ import INTERIM_DIR, PROCESSED_DIR
# import pandas as pd
#
# # data loading
# df = pd.read_csv(PROCESSED_DIR / "cars_no_cyl_outliers_iqr.csv")
#
# # add log horsepower
# df = add_log_feature(df, "horsepower")
#
# # save as ...
# df.to_csv(INTERIM_DIR / "cars_no_cyl_outliers_iqr_log_added.csv", index=False)
# -----------------------------------------------------

# 3. ----- one hot encoding -----

import pandas as pd


def encode_origin(df, col="origin"):
    """
    Map numeric origin codes (1, 2, 3) to readable labels (USA, Europe, Japan)
    and create One-Hot Encoded columns:
    origin_USA, origin_Europe, origin_Japan
    """
    import pandas as pd

    df_out = df.copy()

    # 1. Map numeric codes to country names
    origin_map = {
        1: "USA",
        2: "Europe",
        3: "Japan"
    }
    df_out[col] = df_out[col].map(origin_map)

    # 2. One-Hot Encoding with readable names
    df_out = pd.get_dummies(df_out, columns=[col], prefix="origin", drop_first=False)

    return df_out



