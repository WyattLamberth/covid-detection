"""
Preprocess the COVID X-ray dataset and generate image captions.
"""
import pandas as pd
import sys
import os
from pathlib import Path

# Add the project root to Python path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import RAW_DATA_DIR

def generate_caption(row):
    """
    Generate a descriptive caption for an X-ray image based on its labels.
    
    Args:
        row: Pandas DataFrame row with Label, Label_1_Virus_category, and Label_2_Virus_category
        
    Returns:
        str: A descriptive caption for the X-ray image
    """
    label = row["Label"]
    virus_1 = row.get("Label_1_Virus_category")
    virus_2 = row.get("Label_2_Virus_category")

    if label == "Normal":
        return "No signs of pneumonia."

    elif label == "Pnemonia":
        if virus_2 == "COVID-19":
            return "Lung opacity consistent with COVID-19."
        elif virus_2 in {"SARS", "ARDS", "Streptococcus"}:
            return "Signs of pneumonia, likely viral origin."
        elif virus_1 == "bacteria":
            return "Pneumonia likely due to bacterial infection."
        elif virus_1 == "Virus":
            return "Pneumonia likely due to viral infection."
        else:
            return "Pneumonia detected."

    return "Unspecified condition."

def process_dataset(csv_path=None, out_path=None):
    """
    Process the dataset by cleaning data, generating captions, and validating image paths.
    
    Args:
        csv_path: Path to the input CSV. If None, uses default metadata path.
        out_path: Path to save the processed CSV. If None, uses default output path.
    
    Returns:
        Processed pandas DataFrame
    """
    # Use default paths if not specified
    if csv_path is None:
        csv_path = RAW_DATA_DIR / "Chest_xray_Corona_Metadata.csv"
    
    if out_path is None:
        out_path = RAW_DATA_DIR / "Coronahack-Chest-XRay-Dataset" / "Coronahack-Chest-XRay-Dataset" / "image_captions.csv"
    
    # Load and clean data
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape}")
    
    # Drop rows with missing image names or labels
    df = df.dropna(subset=["X_ray_image_name", "Label"])
    df = df.drop_duplicates(subset=["X_ray_image_name"])
    print(f"Cleaned shape: {df.shape}")
    
    # Generate captions
    print("Generating captions...")
    df["caption"] = df.apply(generate_caption, axis=1)
    
    # Add full image path using Dataset_type (train/test) and image name
    print("Adding image paths...")
    df["image_path"] = df.apply(
        lambda row: RAW_DATA_DIR / "Coronahack-Chest-XRay-Dataset" / "Coronahack-Chest-XRay-Dataset" / 
                   row["Dataset_type"] / row["X_ray_image_name"],
        axis=1
    )
    
    # Filter out entries with missing files
    valid = df["image_path"].apply(lambda p: p.exists())
    print(f"Valid image paths: {valid.sum()} / {len(df)}")
    df = df[valid]
    
    # Save results
    print(f"Caption distribution:\n{df['caption'].value_counts()}")
    
    # Create output directory if it doesn't exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV for later training
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    
    return df

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process X-ray dataset and generate captions")
    parser.add_argument("--input", help="Path to input CSV file")
    parser.add_argument("--output", help="Path to save the processed CSV") 
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)

if __name__ == "__main__":
    main()