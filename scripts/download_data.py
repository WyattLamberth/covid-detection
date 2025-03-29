import kagglehub
import shutil
from utils.paths import RAW_DATA_DIR
from pathlib import Path

def download_covid_xray_dataset():
    print("Downloading dataset from KaggleHub...")
    dataset_path = kagglehub.dataset_download("praveengovi/coronahack-chest-xraydataset")

    dataset_path = Path(dataset_path)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for item in dataset_path.iterdir():
        target_path = RAW_DATA_DIR / item.name
        if not target_path.exists():
            shutil.move(str(item), str(target_path))
            print(f"Moved {item.name} to {RAW_DATA_DIR}")
        else:
            print(f"{item.name} already exists in {RAW_DATA_DIR}")

    print("✅ Download complete.")

if __name__ == "__main__":
    download_covid_xray_dataset()