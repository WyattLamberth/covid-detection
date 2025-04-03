import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from utils.tokenizer import Tokenizer
from scripts.model import ImageCaptioningModel

# -------------------------------
# 1. Config and Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_SIZE = 256
HIDDEN_SIZE = 512
MAX_LEN = 20

MODEL_PATH = "model_caption.pth"
VOCAB_PATH = "data/raw/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/image_captions.csv"

# -------------------------------
# 2. Load Tokenizer and Vocab
# -------------------------------
import pandas as pd
df = pd.read_csv(VOCAB_PATH)
tokenizer = Tokenizer()
tokenizer.build_vocab(df["caption"])

# -------------------------------
# 3. Load Model
# -------------------------------
vocab_size = len(tokenizer)
model = ImageCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, vocab_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------------------------------
# 4. Image Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0).to(device)  # shape [1, 3, 224, 224]

# -------------------------------
# 5. Caption Generation
# -------------------------------
def generate_caption(image_tensor):
    with torch.no_grad():
        features = model.encoder(image_tensor)
        states = None
        input_token = torch.tensor([[tokenizer.word2idx["<start>"]]], device=device)

        generated = []

        for _ in range(MAX_LEN):
            embeddings = model.decoder.embed(input_token)
            if states is None:
                inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
            else:
                inputs = embeddings

            output, states = model.decoder.lstm(inputs, states)
            logits = model.decoder.linear(output.squeeze(1))
            predicted = logits.argmax(1)
            predicted_word = tokenizer.idx2word[predicted.item()]

            if predicted_word == "<end>":
                break
            generated.append(predicted_word)
            input_token = predicted.unsqueeze(0)

    return " ".join(generated)

# -------------------------------
# 6. Run Prediction
# -------------------------------
def predict(image_path):
    image_tensor = preprocess_image(image_path)
    caption = generate_caption(image_tensor)
    return caption

# -------------------------------
# 7. Entry Point
# -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to chest X-ray image")
    args = parser.parse_args()

    caption = predict(args.image)
    print(f"🩺 Predicted Caption: {caption}")