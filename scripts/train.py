import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

from utils.tokenizer import Tokenizer
from scripts.dataset import ChestXrayCaptionDataset
from scripts.model import ImageCaptioningModel
from scripts.collate import collate_fn

# -------------------------------
# 1. Config and Device
# -------------------------------
EPOCHS = 5
BATCH_SIZE = 32
EMBED_SIZE = 256
HIDDEN_SIZE = 512
MAX_SEQ_LEN = 20
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 2. Load Data + Tokenizer
# -------------------------------
CSV_PATH = Path("data/raw/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/image_captions.csv")
df = pd.read_csv(CSV_PATH)
df["image_path"] = df["image_path"].map(Path)

# Tokenizer
tokenizer = Tokenizer()
tokenizer.build_vocab(df["caption"])
vocab_size = len(tokenizer)

# Dataset and DataLoader
dataset = ChestXrayCaptionDataset(df, tokenizer, max_len=MAX_SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# -------------------------------
# 3. Model, Loss, Optimizer
# -------------------------------
model = ImageCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<pad>'])  # mask pad tokens
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------------
# 4. Training Loop
# -------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, captions in dataloader:
        images, captions = images.to(device), captions.to(device)

        outputs = model(images, captions)  # [B, T, vocab_size]
        targets = captions[:, 1:]  # shift targets: [B, T-1]
        outputs = outputs[:, :-1, :].contiguous()  # match target length

        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# -------------------------------
# 5. Save Checkpoint
# -------------------------------
torch.save(model.state_dict(), "model_caption.pth")
print("Model saved to model_caption.pth")