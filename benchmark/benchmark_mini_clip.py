"""
Benchmark Mini-CLIP (ResNet-18 + DistilBERT).
Model tự train trên dataset AIO.
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer

from Data import MiniCLIPDataset
from models import ImageEncoder, TextEncoder, MiniCLIP
from benchmark.utils import print_recalls, get_device, DEFAULT_CSV, DEFAULT_IMG_DIRS


def benchmark_mini_clip(device=None, csv_path=None, img_dirs=None):
    if device is None:
        device = get_device()
    if csv_path is None:
        csv_path = DEFAULT_CSV
    if img_dirs is None:
        img_dirs = DEFAULT_IMG_DIRS

    print(f"\n[Mini-CLIP] Device: {device.type.upper()}")

    # Data
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = MiniCLIPDataset(csv_file=csv_path, img_dir=img_dirs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # Load model
    image_encoder = ImageEncoder(embed_dim=512)
    text_encoder = TextEncoder(embed_dim=512)
    model = MiniCLIP(image_encoder, text_encoder).to(device)

    checkpoint_path = "best_mini_clip.pth"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Cannot find '{checkpoint_path}'. Train the model first.")
        sys.exit(1)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[Mini-CLIP] Loaded weights: {checkpoint_path}")

    # Encode
    all_image_features = []
    all_text_features = []

    print(f"[Mini-CLIP] Encoding {len(dataset)} samples...")
    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            text_tokens = tokenizer(
                list(captions), padding=True, truncation=True,
                max_length=77, return_tensors="pt"
            )
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}

            img_feat = F.normalize(model.image_encoder(images), p=2, dim=-1)
            txt_feat = F.normalize(
                model.text_encoder(text_tokens["input_ids"], text_tokens["attention_mask"]),
                p=2, dim=-1,
            )

            all_image_features.append(img_feat.cpu())
            all_text_features.append(txt_feat.cpu())

    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    similarity_matrix = all_text_features @ all_image_features.T

    results = print_recalls(similarity_matrix, name="Mini-CLIP (ResNet-18 + DistilBERT)")
    return results


if __name__ == "__main__":
    benchmark_mini_clip()
