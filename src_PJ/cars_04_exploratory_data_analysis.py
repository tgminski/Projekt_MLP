
# 1. ----- Importy i przygotowanie danych -----

# --- Core data handling ---
import pandas as pd
import numpy as np

# --- Plotting & visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

import math

# --- Utility ---
# import warnings
# warnings.filterwarnings("ignore")



# 2. ----- Histogram with statistics -----


def plot_histograms_with_stats(df):
    """
    Draw histograms for all numerical columns with:
    - mean, median, +/- 1 std, skewness
    Shown in a clean 2-column layout.
    """
    print("----- Histograms with statistics -----")

    # selection of numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    n_cols = 2
    n_plots = len(numeric_cols)
    n_rows = math.ceil(n_plots / n_cols)

    plt.figure(figsize=(14, 5 * n_rows))

    for i, column in enumerate(numeric_cols, 1):
        values = df[column].dropna().astype(float)

        mean_val = values.mean()
        median_val = values.median()
        std_val = values.std()
        skew_val = values.skew()

        ax = plt.subplot(n_rows, n_cols, i)
        sns.histplot(values, kde=True, color="steelblue", bins=40, alpha=0.7, ax=ax)

        # helper lines
        ax.axvline(mean_val, linestyle="dashed", color="green", label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, linestyle="dotted", color="red", label=f"Median: {median_val:.2f}")
        ax.axvline(mean_val + std_val, linestyle="dashdot", color="orange", label=f"+1σ: {(mean_val + std_val):.2f}")
        ax.axvline(mean_val - std_val, linestyle="dashdot", color="orange", label=f"-1σ: {(mean_val - std_val):.2f}")

        ax.set_title(f"Distribution of {column} (skew={skew_val:.2f})", fontsize=13)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.show()


# 3. ----- Histogram with statistics and boxplot ---- # histogram + boxplot + stats


def plot_histogram_and_boxplot(df, exclude_columns=None):
    """
    Draw histogram and boxplot for each numerical column in df.
    Additional statistics on histogram: mean, median, std (+/-1 std), skewness.

    Parameters:
    - df : DataFrame
    - exclude_columns : list (optional) of columns to exclude
    """

    # select numeric columns automatically
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # remove excluded columns
    if exclude_columns is not None:
        numeric_cols = [col for col in numeric_cols if col not in exclude_columns]

    print("Plotting histogram + boxplot for columns:", numeric_cols)

    for col in numeric_cols:
        # compute statistics
        series = df[col].dropna()
        mean_val = series.mean()
        median_val = series.median()
        std_val = series.std()
        skew_val = series.skew()

        plt.figure(figsize=(6, 4))

        # ---------------- HISTOGRAM ----------------
        ax1 = plt.subplot(2, 1, 1)
        sns.histplot(series, kde=True, bins=30, color="steelblue", ax=ax1)

        # Lines: mean, median, +1σ, -1σ
        ax1.axvline(mean_val, linestyle='dashed', color='green', label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, linestyle='dotted', color='red', label=f'Median: {median_val:.2f}')
        ax1.axvline(mean_val + std_val, linestyle='dashdot', color='orange', label=f'+1σ: {(mean_val + std_val):.2f}')
        ax1.axvline(mean_val - std_val, linestyle='dashdot', color='orange', label=f'-1σ: {(mean_val - std_val):.2f}')

        ax1.set_title(f"Histogram of {col}  (skewness = {skew_val:.3f})", fontsize=12)
        ax1.set_xlabel(col)
        ax1.set_ylabel("Count")
        ax1.legend()

        # ---------------- BOXPLOT ----------------
        ax2 = plt.subplot(2, 1, 2)
        sns.boxplot(x=series, color="orange", ax=ax2)
        ax2.set_title(f"Boxplot of {col}", fontsize=12)
        ax2.set_xlabel(col)

        plt.tight_layout()
        plt.show()


# 4. ----- Boxplots -----


def plot_boxplots(df):

    print("----- Boxplots -----")

    # numerical columns except 'year' and 'origin' (treated as categorical)
    numeric_cols = [
        col for col in df.select_dtypes(include=["int64", "float64"]).columns
        if col.lower() not in ["year", "origin"]
    ]

    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3  # dynamic number of rows (3 per row)

    plt.figure(figsize=(14, 4 * n_rows))

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, 3, i)
        sns.boxplot(data=df, x=col, color='steelblue')
        plt.title(col)
        plt.xlabel(col)

    plt.tight_layout()
    plt.show()


# 5. ----- show_outliers_iqr_with_boxplots -----


def show_outliers_iqr_with_boxplots(df, columns=None, save_clean_path=None):
    """
    Detect outliers using the IQR method, visualize boxplots,
    and optionally save the cleaned dataset (without outliers).
    """

    # wybór kolumn numerycznych
    if columns is None:
        columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("Analyzing columns:", columns)

    df_numeric = df[columns].copy()
    mask_df = pd.DataFrame()

    # --- IQR DETEKCJA OUTLIERÓW ---
    for col in columns:
        Q1 = df_numeric[col].quantile(0.25)
        Q3 = df_numeric[col].quantile(0.75)
        IQR = Q3 - Q1

        mask_df[col] = (df_numeric[col] < Q1 - 1.5 * IQR) | \
                       (df_numeric[col] > Q3 + 1.5 * IQR)

    # flaga wierszowa
    mask_df["is_outlier"] = mask_df.any(axis=1)

    # wydzielenie outlierów
    df_reset = df.reset_index(drop=True)
    outliers_only = df_reset[mask_df["is_outlier"]].copy()

    # print("\nNumber of outliers detected:", outliers_only.shape[0])

    # --- usuwanie outlierów ---
    df_clean = df_reset[~mask_df["is_outlier"]].copy()

    # --- zapis oczyszczonego zbioru ---
    if save_clean_path is not None:
        df_clean.to_csv(save_clean_path, index=False)
        # print(f"Clean dataset saved to: {save_clean_path}")

    # --- BOX PLOTS ---
    num_cols = len(columns)
    n_rows = (num_cols + 2) // 3

    plt.figure(figsize=(14, 4 * n_rows))

    for idx, col in enumerate(columns, 1):
        plt.subplot(n_rows, 3, idx)
        sns.boxplot(x=df[col], color='steelblue')
        plt.title(f"Boxplot — {col}")

    plt.tight_layout()
    plt.show()
    print()
    print("\nNumber of outliers detected:", outliers_only.shape[0])
    print()
    print(f"Clean dataset saved to: {save_clean_path}")
    print()
    return {
        "outliers_only": outliers_only,
        "mask_df": mask_df,
        "df_clean": df_clean
    }


# 5. ----- show_outliers_iqr_with_boxplots_2 liczba outlierów dla każdej kolumny -----


def show_outliers_iqr_with_boxplots_2(df, columns=None, save_clean_path=None):
    """
    Detect outliers using the IQR method, visualize boxplots,
    and optionally save the cleaned dataset (without outliers).

    Now ALSO returns:
    - outlier_counts_per_column: liczba outlierów dla każdej kolumny
    """

    # wybór kolumn numerycznych
    if columns is None:
        columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("Analyzing columns:", columns)

    df_numeric = df[columns].copy()
    mask_df = pd.DataFrame()

    # słownik na liczbę outlierów per kolumna
    outlier_counts = {}

    # --- IQR DETEKCJA OUTLIERÓW ---
    for col in columns:
        Q1 = df_numeric[col].quantile(0.25)
        Q3 = df_numeric[col].quantile(0.75)
        IQR = Q3 - Q1

        mask_df[col] = (df_numeric[col] < Q1 - 1.5 * IQR) | \
                       (df_numeric[col] > Q3 + 1.5 * IQR)

        # liczba outlierów w kolumnie
        outlier_counts[col] = mask_df[col].sum()

    # flaga wierszowa
    mask_df["is_outlier"] = mask_df.any(axis=1)

    # wydzielenie outlierów
    df_reset = df.reset_index(drop=True)
    outliers_only = df_reset[mask_df["is_outlier"]].copy()

    # usuwanie outlierów
    df_clean = df_reset[~mask_df["is_outlier"]].copy()

    # zapis oczyszczonego zbioru
    if save_clean_path is not None:
        df_clean.to_csv(save_clean_path, index=False)

    # --- BOX PLOTS ---
    num_cols = len(columns)
    n_rows = (num_cols + 2) // 3

    plt.figure(figsize=(14, 4 * n_rows))

    for idx, col in enumerate(columns, 1):
        plt.subplot(n_rows, 3, idx)
        sns.boxplot(x=df[col], color='steelblue')
        plt.title(f"Boxplot — {col}")

    plt.tight_layout()
    plt.show()

    # --- PODSUMOWANIE ---
    print("\n=== OUTLIER SUMMARY ===")
    print(f"Total rows with any outlier: {outliers_only.shape[0]}\n")

    print("Outliers per column:")
    for col, count in outlier_counts.items():
        print(f"  {col}: {count}")

    print(f"\nClean dataset saved to: {save_clean_path}\n")

    return {
        "outliers_only": outliers_only,
        "mask_df": mask_df,
        "df_clean": df_clean,
        "outlier_counts_per_column": outlier_counts
    }



# 6. ----- Skewness -----


def compute_skew(df):
    """
    Compute skewness for numeric columns,
    excluding 'name', 'year', and 'origin'.
    """

    exclude_cols = ["name", "year", "origin"]

    # usuń wskazane kolumny
    df_temp = df.drop(columns=exclude_cols, errors="ignore")

    # wybierz tylko numeryczne kolumny
    df_num = df_temp.select_dtypes(include=["int64", "float64"])

    # oblicz skewness
    skew_values = df_num.skew()

    print("Skewness values:")
    print(skew_values)

    # return skew_values


# 7.1 ----- log transformation -----


def log_transform_and_replace(df, columns):
    """
    Apply log transformation to selected columns.
    Replace original columns by their log versions.

    Creates new columns <col>_log, then removes original <col>.
    """
    import numpy as np

    df = df.copy()

    for col in columns:
        if (df[col] <= 0).any():
            raise ValueError(f"Column '{col}' contains non-positive values. Cannot apply log transform.")

        new_col = f"{col}_log"
        df[new_col] = np.log(df[col])

        # remove original column
        df.drop(columns=col, inplace=True)
        print(f"Replaced column: {col} -> {new_col}")

    return df

# # cars_04_exploratory_data_analysis.py
#
# from src_PJ.cars_04_exploratory_data_analysis import log_transform_and_replace
# from src_PJ.config_PJ import INTERIM_DIR, PROCESSED_DIR
#
# # 1. load dataset without 3-cylinder cars
# df = pd.read_csv(INTERIM_DIR / "cars_no_cyl_outliers_iqr.csv")
#
# # 2. choose columns to transform
# cols_to_transform = ["horsepower"]   # add more if needed
#
# # 3. apply log transform and replace original columns
# df_log = log_transform_and_replace(df, cols_to_transform)
#
# # 4. save dataset
# save_path = PROCESSED_DIR / "cars_no_cyl_outliers_iqr_log_eda.csv"
# df_log.to_csv(save_path, index=False)
#
# print("Saved transformed dataset to:", save_path)
# print(f'shape:{df_log.shape}')



# 7.2 ----- add_log_transforms -----

def add_log_transforms(df, columns):
    """
    Add log-transformed versions of selected columns.
    Does NOT remove original columns.

    Example:
    horsepower -> horsepower_log
    """
    import numpy as np

    df_out = df.copy()

    for col in columns:
        if col not in df_out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        if (df_out[col] <= 0).any():
            raise ValueError(f"Column '{col}' contains non-positive values; cannot apply log.")

        df_out[f"{col}_log"] = np.log(df_out[col])
        print(f"Added column: {col}_log")


    return df_out

# from src_PJ.cars_04_exploratory_data_analysis import add_log_transforms
# from src_PJ.config_PJ import INTERIM_DIR, PROCESSED_DIR
# import pandas as pd
#
# df = pd.read_csv(INTERIM_DIR / "cars_no_cyl_outliers_iqr.csv")
#
# df_log = add_log_transforms(df, ["horsepower"])
#
# df_log.to_csv(PROCESSED_DIR / "cars_no_cyl_outliers_iqr_log_added.csv", index=False)



# 8. ----- Basic info -----


def show_basic_info(df):
    print("=== Basic Info ===")
    print("Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nUnique Values:\n", df.nunique())
    print("\nMissing Values:\n", df.isna().sum())

    summary = pd.DataFrame({
        'Data type': df.dtypes,
        'Unique values': df.nunique(),
        'Missing values': df.isna().sum()
    })
    return summary


# 9. ----- Statistics -----


def show_statistics(df):
    print("=== Descriptive Statistics ===")
    return df.describe().T


# 10. ----- Pairplot -----


def plot_pairplot(df):
    print("=== Pairplot ===")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    sns.pairplot(df[numeric_cols], diag_kind="hist")
    plt.show()


# 11. ----- Correlation heatmap -----


def plot_correlation_heatmap(df):
    print("----- Correlation Heatmap -----")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='RdBu', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    return corr


# 12. ----- Regplots with trendline -----


def plot_regplots(df, columns_to_plot=None, exclude_columns=None, target="mpg"):
    """
    Draw regression plots (regplot) for multiple columns in a 2-column layout.
    Parameters:
    - df : DataFrame
    - columns_to_plot : list of columns to include (optional)
    - exclude_columns : list of columns to exclude (optional)
    - target : dependent variable (default = 'mpg')
    """

    # 1. Automatycznie wybierz numeryczne, jeśli nie podano
    if columns_to_plot is None:
        columns_to_plot = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # 2. Usuń kolumny wykluczone
    if exclude_columns is not None:
        columns_to_plot = [col for col in columns_to_plot if col not in exclude_columns]

    # 3. Usuń target, żeby nie rysować mpg vs mpg
    columns_to_plot = [col for col in columns_to_plot if col != target]

    print("Plotting regplots for:", columns_to_plot)

    # --- USTAWIENIE SIATKI 2-KOLUMNOWEJ ---
    n_cols = 2
    n_rows = math.ceil(len(columns_to_plot) / n_cols)

    plt.figure(figsize=(14, 5 * n_rows))

    for idx, col in enumerate(columns_to_plot, 1):
        ax = plt.subplot(n_rows, n_cols, idx)

        sns.regplot(
            data=df,
            x=col,
            y=target,
            scatter_kws={'alpha': 0.6},
            line_kws={'color': 'red'},
            ax=ax
        )

        ax.set_title(f"{target} vs {col}", fontsize=13)
        ax.set_xlabel(col)
        ax.set_ylabel(target)

    plt.tight_layout()
    plt.show()


# 13. ----- Strong correlations -----


def find_strong_correlations(df, threshold=0.7):
    print("----- Strong Correlations (>|0.7|) -----")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[numeric_cols].corr()

    high_corr = {}

    for col in corr.columns:
        strong = corr[col][corr[col].abs() > threshold]
        strong = strong[strong.index != col]  # exclude 1.0 diagonal
        if not strong.empty:
            high_corr[col] = strong

    return high_corr


# 14. ----- categorical variable vs. numerical variable -----

def categorical_boxplot(df: pd.DataFrame, cat_col: str, num_col: str):
    """
    Simple box plot: categorical variable vs. numerical variable.

    Parametry:
    ----------
    df : pd.DataFrame
        DataFrame z danymi.
    cat_col : str
        Nazwa kolumny kategorycznej (np. 'origin').
    num_col : str
        Nazwa kolumny numerycznej (np. 'mpg').
    """

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=cat_col, y=num_col, palette='Paired')
    plt.title(f'{num_col} depending on {cat_col}')
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.tight_layout()
    plt.show()