"""
Benchmark OpenAI CLIP (ViT-B/32).
Sử dụng model pretrained gốc từ OpenAI qua HuggingFace transformers.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from data import MiniCLIPDataset
from benchmark.utils import (
    print_recalls, collate_fn_pil, get_device,
    DEFAULT_CSV, DEFAULT_IMG_DIRS,
)


def benchmark_clip(device=None, csv_path=None, img_dirs=None):
    if device is None:
        device = get_device()
    if csv_path is None:
        csv_path = DEFAULT_CSV
    if img_dirs is None:
        img_dirs = DEFAULT_IMG_DIRS

    print(f"\n[CLIP] Device: {device.type.upper()}")

    # Dataset trả về PIL images (không transform)
    dataset = MiniCLIPDataset(csv_file=csv_path, img_dir=img_dirs, transform=None)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=0, collate_fn=collate_fn_pil,
    )

    # Load CLIP
    model_name = "openai/clip-vit-base-patch32"
    print(f"[CLIP] Loading {model_name}...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    all_image_features = []
    all_text_features = []

    print(f"[CLIP] Encoding {len(dataset)} samples...")
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="CLIP"):
            inputs = processor(
                text=captions, images=images,
                padding=True, truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            img_feat = F.normalize(outputs.image_embeds, p=2, dim=-1)
            txt_feat = F.normalize(outputs.text_embeds, p=2, dim=-1)

            all_image_features.append(img_feat.cpu())
            all_text_features.append(txt_feat.cpu())

    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    similarity_matrix = all_text_features @ all_image_features.T
    results = print_recalls(similarity_matrix, name="CLIP (ViT-B/32, OpenAI)")

    # Giải phóng bộ nhớ
    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    benchmark_clip()
