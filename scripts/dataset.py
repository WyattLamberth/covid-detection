from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class ChestXrayCaptionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=20, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Image ---
        img_path = Path(row["image_path"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # --- Caption ---
        caption = row["caption"]
        seq = self.tokenizer.text_to_sequence(caption)
        seq = self.tokenizer.pad_sequence(seq, self.max_len)

        return image, seq