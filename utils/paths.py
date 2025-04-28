from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Dataset folder
CORONAHACK_DIR = RAW_DATA_DIR / "Coronahack-Chest-XRay-Dataset" / \
    "Coronahack-Chest-XRay-Dataset"

# Metadata and Split CSVs
IMAGE_CAPTIONS_CSV = CORONAHACK_DIR / "image_captions.csv"
TRAIN_SPLIT_CSV = PROJECT_ROOT / "train_split.csv"
VAL_SPLIT_CSV = PROJECT_ROOT / "val_split.csv"
TEST_SPLIT_CSV = PROJECT_ROOT / "test_split.csv"
