import pandas as pd
df = pd.read_csv(
    "data/raw/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/image_captions.csv")
print(df["caption"].unique()[:20])  # print some example captions
