import pandas as pd
from pathlib import Path
import numpy as np

# Path to your existing captions CSV
CSV_PATH = Path(
    "data/raw/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/image_captions.csv")

# Load the CSV
df = pd.read_csv(CSV_PATH)

# Shuffle the dataframe
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split sizes
n_total = len(df)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.15)
n_test = n_total - n_train - n_val  # Remaining

# Assign dataset types
df.loc[:n_train - 1, "Dataset_type"] = "train"
df.loc[n_train:n_train + n_val - 1, "Dataset_type"] = "val"
df.loc[n_train + n_val:, "Dataset_type"] = "test"

# Save back to CSV
df.to_csv(CSV_PATH, index=False)

print(
    f"✅ Dataset split complete: {n_train} train, {n_val} val, {n_test} test samples.")
