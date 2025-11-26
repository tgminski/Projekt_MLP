# cars_00_download_data_PJ.py

import gdown
from src_PJ.config_PJ import RAW_DIR

# Google Drive file source
# https://drive.google.com/file/d/1QrhkJvc42A5rZ_u4wQu0qwODj5UMZlLe/view?usp=sharing

# File IDs on Google Drive
FILES = {
    "cars.csv": "1QrhkJvc42A5rZ_u4wQu0qwODj5UMZlLe",
}

def download_raw_files():
    """Download raw csv dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, file_id in FILES.items():
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = RAW_DIR / filename
        print(f"Downloading {filename} to {output_path}")
        gdown.download(url, str(output_path), quiet=False)

    print("Download complete.")


# download_raw_files()