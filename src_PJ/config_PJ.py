# config_PJ.py

from pathlib import Path

# === ROOT PROJECT DIR ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# === DATA FOLDERS ===
DATA_DIR = PROJECT_ROOT / "data_PJ"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# === MODELS FOLDERS ===
MODELS_DIR = PROJECT_ROOT / "models_PJ"
MODELS_REGRESSION_DIR = MODELS_DIR / "regression"
MODELS_CLASSIFICATION_DIR = MODELS_DIR / "classification"

# === FILES ===
RAW_CARS = RAW_DIR / "cars.csv"

# === GLOBAL SETTINGS ===
SEED = 42
