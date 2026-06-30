"""
Vẽ Cosine Similarity Matrix giữa ảnh và text.
Hiển thị 1 batch gồm 8 cặp (image, caption).
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

from Data import MiniCLIPDataset
from models import ImageEncoder, TextEncoder, MiniCLIP


def main():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for visualization: {device}")

    csv_path = "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/ML_test/test.csv"
    img_dirs = [
        "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-002/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-003/AIO_conquer/images",
    ]

    dataset = MiniCLIPDataset(csv_file=csv_path, img_dir=img_dirs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    image_encoder = ImageEncoder(embed_dim=512)
    text_encoder = TextEncoder(embed_dim=512)
    model = MiniCLIP(image_encoder, text_encoder)

    model.load_state_dict(torch.load("best_mini_clip.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    images, captions = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        text_tokens = tokenizer(
            list(captions), padding=True, truncation=True,
            max_length=77, return_tensors="pt",
        )
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}

        image_features, text_features, logit_scale = model(
            images, text_tokens["input_ids"], text_tokens["attention_mask"],
        )

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        similarity_matrix = (image_features @ text_features.T).cpu().numpy()

    images = images.cpu()
    short_captions = [cap[:20] + "..." if len(cap) > 20 else cap for cap in captions]

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(8, 2, width_ratios=[1, 4])

    for i in range(8):
        ax_img = fig.add_subplot(gs[i, 0])

        img_tensor = images[i].clone().permute(1, 2, 0)
        std = torch.tensor([0.229, 0.224, 0.225])
        mean = torch.tensor([0.485, 0.456, 0.406])
        img_np = (img_tensor * std + mean).clamp(0, 1).numpy()

        ax_img.imshow(img_np)
        ax_img.axis("off")
        ax_img.set_title(f"Image {i}", fontsize=10)

    ax_heat = fig.add_subplot(gs[:, 1])
    sns.heatmap(
        similarity_matrix, annot=True, cmap="coolwarm", fmt=".2f",
        xticklabels=short_captions, yticklabels=False, ax=ax_heat,
    )

    ax_heat.set_title("Cosine Similarity Matrix", fontsize=14)
    ax_heat.set_xlabel("Text", fontsize=12)
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
