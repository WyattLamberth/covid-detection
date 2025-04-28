import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from utils.tokenizer import Tokenizer
from utils.paths import IMAGE_CAPTIONS_CSV
from scripts.dataset import ChestXrayCaptionDataset
from scripts.model import ImageCaptioningModel
from scripts.collate import collate_fn


# -------------------------------
# 1. Config and Device
# -------------------------------
EPOCHS = 50
BATCH_SIZE = 32
EMBED_SIZE = 256
HIDDEN_SIZE = 512
MAX_SEQ_LEN = 20
LEARNING_RATE = 1e-3
PATIENCE = 5  # Early stopping patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# -------------------------------
# 2. Load Data + Tokenizer
# -------------------------------
df = pd.read_csv(IMAGE_CAPTIONS_CSV)

df["image_path"] = df["image_path"].map(Path)

tokenizer = Tokenizer()
tokenizer.build_vocab(df["caption"])
vocab_size = len(tokenizer)

dataset = ChestXrayCaptionDataset(df, tokenizer, max_len=MAX_SEQ_LEN)

# Dataset splitting (70% train, 15% val, 15% test)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size])

train_df = df.iloc[train_set.indices]
val_df = df.iloc[val_set.indices]
test_df = df.iloc[test_set.indices]

train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)

print(f"📂 Dataset Split:")
print(f"Training: {train_size} images")
print(f"Validation: {val_size} images")
print(f"Test: {test_size} images")
print("✅ Saved train_split.csv, val_split.csv, and test_split.csv.")


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)

# -------------------------------
# 3. Model, Loss, Optimizer
# -------------------------------
model = ImageCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<pad>'])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------------
# 4. Training Loop with Validation & Early Stopping
# -------------------------------
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for images, captions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
        images, captions = images.to(device), captions.to(device)
        outputs = model(images, captions)
        targets = captions[:, 1:]
        outputs = outputs[:, :-1, :].contiguous()

        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, captions in val_loader:
            images, captions = images.to(device), captions.to(device)
            outputs = model(images, captions)
            targets = captions[:, 1:]
            outputs = outputs[:, :-1, :].contiguous()

            val_loss = criterion(
                outputs.view(-1, vocab_size), targets.reshape(-1))
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "model_caption.pth")
        print(
            f"✅ Model saved at Epoch {epoch+1} (Best Val Loss: {best_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

# -------------------------------
# 5. Plot Training and Validation Loss
# -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses)+1), train_losses,
         label="Training Loss", marker='o')
plt.plot(range(1, len(val_losses)+1), val_losses,
         label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()
