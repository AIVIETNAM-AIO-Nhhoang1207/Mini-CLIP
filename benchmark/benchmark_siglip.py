"""
Benchmark SigLIP (google/siglip-base-patch16-224).
SigLIP dùng Sigmoid loss thay vì Softmax (InfoNCE) như CLIP.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

from Data import MiniCLIPDataset
from benchmark.utils import (
    print_recalls, collate_fn_pil, get_device,
    DEFAULT_CSV, DEFAULT_IMG_DIRS,
)


def benchmark_siglip(device=None, csv_path=None, img_dirs=None):
    if device is None:
        device = get_device()
    if csv_path is None:
        csv_path = DEFAULT_CSV
    if img_dirs is None:
        img_dirs = DEFAULT_IMG_DIRS

    print(f"\n[SigLIP] Device: {device.type.upper()}")

    # Dataset trả về PIL images (không transform)
    dataset = MiniCLIPDataset(csv_file=csv_path, img_dir=img_dirs, transform=None)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=0, collate_fn=collate_fn_pil,
    )

    # Load SigLIP
    model_name = "google/siglip-base-patch16-224"
    print(f"[SigLIP] Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_image_features = []
    all_text_features = []

    print(f"[SigLIP] Encoding {len(dataset)} samples...")
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="SigLIP"):
            inputs = processor(
                text=captions, images=images,
                padding="max_length", truncation=True,
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
    results = print_recalls(similarity_matrix, name="SigLIP (siglip-base-patch16-224)")

    # Giải phóng bộ nhớ
    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    benchmark_siglip()
