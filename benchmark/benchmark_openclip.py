"""
Benchmark OpenCLIP (ViT-B-32, laion2b_s34b_b79k).
Pretrained trên LAION-2B, mạnh hơn CLIP gốc.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

from data import MiniCLIPDataset
from benchmark.utils import (
    print_recalls, collate_fn_pil, get_device,
    DEFAULT_CSV, DEFAULT_IMG_DIRS,
)


def benchmark_openclip(device=None, csv_path=None, img_dirs=None):
    if device is None:
        device = get_device()
    if csv_path is None:
        csv_path = DEFAULT_CSV
    if img_dirs is None:
        img_dirs = DEFAULT_IMG_DIRS

    print(f"\n[OpenCLIP] Device: {device.type.upper()}")

    # Dataset trả về PIL images (không transform)
    dataset = MiniCLIPDataset(csv_file=csv_path, img_dir=img_dirs, transform=None)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=0, collate_fn=collate_fn_pil,
    )

    # Load OpenCLIP
    print("[OpenCLIP] Loading ViT-B-32 (laion2b_s34b_b79k)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    all_image_features = []
    all_text_features = []

    print(f"[OpenCLIP] Encoding {len(dataset)} samples...")
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="OpenCLIP"):
            image_tensors = torch.stack([preprocess(img) for img in images]).to(device)
            text_tokens = tokenizer(captions).to(device)

            img_feat = F.normalize(model.encode_image(image_tensors), p=2, dim=-1)
            txt_feat = F.normalize(model.encode_text(text_tokens), p=2, dim=-1)

            all_image_features.append(img_feat.cpu())
            all_text_features.append(txt_feat.cpu())

    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    similarity_matrix = all_text_features @ all_image_features.T
    results = print_recalls(similarity_matrix, name="OpenCLIP (ViT-B-32, LAION-2B)")

    # Giải phóng bộ nhớ
    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    benchmark_openclip()
