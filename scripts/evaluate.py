import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from utils.tokenizer import Tokenizer
from scripts.model import ImageCaptioningModel
from scripts.predict import generate_caption
from utils.paths import TEST_SPLIT_CSV


def caption_to_label(caption: str) -> str:
    caption = caption.lower()
    if "no signs" in caption or "clear" in caption or "normal" in caption:
        return "normal"
    elif "covid" in caption or "corona" in caption:
        return "covid"
    elif "bacterial" in caption or "bacteria" in caption:
        return "bacterial"
    elif "viral" in caption or "virus" in caption:
        return "viral"
    elif "pneumonia" in caption:
        return "unspecified"
    else:
        return "unknown"


def main():
    # -----------------------------
    # Config and device setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # -----------------------------
    # Load metadata and test set
    # -----------------------------
    test_df = pd.read_csv(TEST_SPLIT_CSV)
    test_df["image_path"] = test_df["image_path"].map(Path)
    print(f"🧪 Loaded {len(test_df)} test images from test_split.csv.")

    # -----------------------------
    # Tokenizer and model
    # -----------------------------
    tokenizer = Tokenizer()
    tokenizer.build_vocab(test_df["caption"])
    vocab_size = len(tokenizer)

    model = ImageCaptioningModel(
        embed_size=256, hidden_size=512, vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load("model_caption.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # -----------------------------
    # Predict captions
    # -----------------------------
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="🔮 Predicting"):
        try:
            image = Image.open(row["image_path"]).convert("RGB")
            image_tensor = transform(image).unsqueeze(
                0).to(device)
            prediction = generate_caption(
                model, image_tensor, tokenizer, device=device)
        except Exception as e:
            prediction = "error"
            print(f"Error on image {row['image_path']}: {e}")

        results.append({
            "image_name": row["X_ray_image_name"],
            "true_caption": row["caption"],
            "predicted_caption": prediction
        })

    # -----------------------------
    # Convert to labels and evaluate
    # -----------------------------
    out_df = pd.DataFrame(results)
    out_df["true_label"] = out_df["true_caption"].map(caption_to_label)
    out_df["predicted_label"] = out_df["predicted_caption"].map(
        caption_to_label)

    print("\n📊 Classification Report:")
    print(classification_report(
        out_df["true_label"], out_df["predicted_label"], zero_division=0))

    print("🧾 Confusion Matrix:")
    print(confusion_matrix(out_df["true_label"], out_df["predicted_label"]))

    # -----------------------------
    # Save results
    # -----------------------------
    out_df.to_csv("test_predictions.csv", index=False)
    print("✅ Results saved to test_predictions.csv")


if __name__ == "__main__":
    main()
