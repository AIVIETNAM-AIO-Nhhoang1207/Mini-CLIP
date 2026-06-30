"""
Train Mini-CLIP: ResNet-18 (Image) + DistilBERT (Text) với Contrastive Loss.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer

from data import MiniCLIPDataset
from models import ImageEncoder, TextEncoder, MiniCLIP, contrastive_loss


def main():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Paths
    base = "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/ML_test"
    train_csv_path = f"{base}/train.csv"
    val_csv_path = f"{base}/val.csv"
    test_csv_path = f"{base}/test.csv"

    img_dirs = [
        "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-002/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-003/AIO_conquer/images",
    ]

    train_dataset = MiniCLIPDataset(csv_file=train_csv_path, img_dir=img_dirs, transform=transform)
    val_dataset = MiniCLIPDataset(csv_file=val_csv_path, img_dir=img_dirs, transform=transform)
    test_dataset = MiniCLIPDataset(csv_file=test_csv_path, img_dir=img_dirs, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    image_encoder = ImageEncoder(embed_dim=512).to(device)
    text_encoder = TextEncoder(embed_dim=512).to(device)
    model = MiniCLIP(image_encoder, text_encoder).to(device)

    optimizer = optim.Adam([
        {"params": model.image_encoder.backbone.parameters(), "lr": 1e-5},
        {"params": model.text_encoder.model.parameters(), "lr": 1e-5},
        {"params": model.image_encoder.projection.parameters(), "lr": 1e-3},
        {"params": model.text_encoder.projection.parameters(), "lr": 1e-3},
        {"params": [model.logit_scale], "lr": 1e-3},
    ])

    model.train()

    # Training config
    epochs = 100
    patience = 5
    min_delta = 0.005
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(epochs):
        print(f"\n{epoch + 1}/{epochs}")

        # --- Train ---
        for batch_idx, (images, captions) in enumerate(train_dataloader):
            images = images.to(device)
            text_tokens = tokenizer(
                list(captions), padding=True, truncation=True,
                max_length=77, return_tensors="pt",
            )

            input_ids = text_tokens["input_ids"].to(device)
            attention_mask = text_tokens["attention_mask"].to(device)

            optimizer.zero_grad()
            image_features, text_features, logit_scale = model(images, input_ids, attention_mask)
            loss = contrastive_loss(image_features, text_features, logit_scale)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1} | Batch {batch_idx + 1:02d} | Train Loss: {loss.item():.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for val_images, val_captions in val_dataloader:
                val_images = val_images.to(device)
                val_tokens = tokenizer(
                    list(val_captions), padding=True, truncation=True,
                    max_length=77, return_tensors="pt",
                )

                val_img_feat, val_txt_feat, val_scale = model(
                    val_images,
                    val_tokens["input_ids"].to(device),
                    val_tokens["attention_mask"].to(device),
                )
                batch_loss = contrastive_loss(val_img_feat, val_txt_feat, val_scale)
                val_loss += batch_loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"\n(VAL): Loss = {avg_val_loss:.4f}\n" + "=" * 40)

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_mini_clip.pth")
            print("Validation loss improved. Saved best_mini_clip.pth")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early Stopping: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("Early Stopping triggered. Halting training.")
            break

        model.train()

    # --- Test ---
    print("\n" + "=" * 50)
    model.load_state_dict(torch.load("best_mini_clip.pth"))
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for test_images, test_captions in test_dataloader:
            test_images = test_images.to(device)
            test_tokens = tokenizer(
                list(test_captions), padding=True, truncation=True,
                max_length=77, return_tensors="pt",
            )

            test_img_feat, test_txt_feat, test_scale = model(
                test_images,
                test_tokens["input_ids"].to(device),
                test_tokens["attention_mask"].to(device),
            )
            batch_loss = contrastive_loss(test_img_feat, test_txt_feat, test_scale)
            test_loss += batch_loss.item()

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"TEST: Loss = {avg_test_loss:.4f}")
    print("*" * 50)


if __name__ == "__main__":
    main()
