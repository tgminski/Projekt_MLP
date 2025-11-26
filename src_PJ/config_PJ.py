# config_PJ.py

from pathlib import Path

# root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# data folders
DATA_DIR = PROJECT_ROOT / 'data_PJ'
RAW_DIR = DATA_DIR / 'raw'
INTERIM_DIR = DATA_DIR / 'interim'
PROCESSED_DIR = DATA_DIR / 'processed'

# files
RAW_CARS = RAW_DIR / 'cars.csv'

# misc
SEED = 102