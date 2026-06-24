import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import DistilBertTokenizer, BertTokenizer, AutoTokenizer

from Data import MiniCLIPDataset
from models import ImageEncoder, TextEncoder, get_encoders, get_checkpoint_name

class MiniCLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        
    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        return image_features, text_features, self.logit_scale.exp()

def contrastive_loss(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    
    labels = torch.arange(image_features.shape[0], device=image_features.device)
    
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)
    
    return (loss_img + loss_txt) / 2

if __name__ == "__main__":
    # ============================================================
    #  Argparse — chọn encoder config từ command line
    # ============================================================
    parser = argparse.ArgumentParser(description="Train Mini-CLIP với các encoder khác nhau")
    parser.add_argument("--image", type=str, default="resnet18", choices=["resnet18", "vit"],
                        help="Image encoder: resnet18 (default) hoặc vit")
    parser.add_argument("--text", type=str, default="distilbert", choices=["distilbert", "bert"],
                        help="Text encoder: distilbert (default) hoặc bert")
    parser.add_argument("--epochs", type=int, default=100, help="Số epoch (default: 100)")
    args = parser.parse_args()

    print(f"{'='*50}")
    print(f"  Config: Image={args.image}, Text={args.text}")
    print(f"  Epochs: {args.epochs}")
    print(f"{'='*50}")

    # ============================================================
    #  Tạo encoder + tokenizer dựa trên config
    # ============================================================
    image_encoder, text_encoder, tokenizer_name = get_encoders(args.image, args.text, embed_dim=512)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    checkpoint_name = get_checkpoint_name(args.image, args.text)

    print(f"Tokenizer: {tokenizer_name}")
    print(f"Checkpoint sẽ lưu: {checkpoint_name}")
    
    # ============================================================
    #  Data — thêm Resize(224, 224)
    # ============================================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_csv_path = "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/ML_test/train.csv"
    val_csv_path = "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/ML_test/val.csv"
    test_csv_path = "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/ML_test/test.csv"
    
    img_dirs = [
        "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-002/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-003/AIO_conquer/images"
    ]
    
    train_dataset = MiniCLIPDataset(csv_file=train_csv_path, img_dir=img_dirs, transform=transform)
    val_dataset = MiniCLIPDataset(csv_file=val_csv_path, img_dir=img_dirs, transform=transform)
    test_dataset = MiniCLIPDataset(csv_file=test_csv_path, img_dir=img_dirs, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    
    # ============================================================
    #  Model + Optimizer
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MiniCLIP(image_encoder, text_encoder).to(device)

    # Optimizer — phân biệt lr cho backbone vs projection
    optimizer = optim.Adam([
        {'params': model.image_encoder.backbone.parameters(), 'lr': 1e-5},
        {'params': model.text_encoder.model.parameters(), 'lr': 1e-5},
        {'params': model.image_encoder.projection.parameters(), 'lr': 1e-3},
        {'params': model.text_encoder.projection.parameters(), 'lr': 1e-3},
        {'params': [model.logit_scale], 'lr': 1e-3}
    ])

    model.train()
    
    epochs = args.epochs
    
    # Các biến dùng để Dừng sớm (tránh lố)
    patience = 5
    min_delta = 0.005
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(epochs):
        print(f"\n{epoch + 1}/{epochs}")
        
        for batch_idx, (images, captions) in enumerate(train_dataloader):
            images = images.to(device)
            text_tokens = tokenizer(
                list(captions),  
                padding=True, 
                truncation=True, 
                max_length=77, 
                return_tensors="pt" 
            )

            input_ids = text_tokens["input_ids"].to(device)
            attention_mask = text_tokens["attention_mask"].to(device)
            
            optimizer.zero_grad()
            
            image_features, text_features, logit_scale = model(images, input_ids, attention_mask)
            
            loss = contrastive_loss(image_features, text_features, logit_scale)
            
            loss.backward()
            
            optimizer.step()
            print(f"Epoch {epoch + 1} | Batch {batch_idx + 1:02d} | Train Loss: {loss.item():.4f}")
            
        model.eval() 

        val_loss = 0.0
        
        with torch.no_grad(): 
            for val_images, val_captions in val_dataloader:
                val_images = val_images.to(device)
                val_tokens = tokenizer(
                    list(val_captions), padding=True, truncation=True, 
                    max_length=77, return_tensors="pt"
                )
                
                val_img_feat, val_txt_feat, val_scale = model(
                    val_images, val_tokens["input_ids"].to(device), val_tokens["attention_mask"].to(device)
                )
                
                batch_loss = contrastive_loss(val_img_feat, val_txt_feat, val_scale)
                val_loss += batch_loss.item()
                
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"\n(VAL): Loss = {avg_val_loss:.4f}\n" + "="*40)
        
        # Logic của Dừng sớm
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), checkpoint_name)
            print(f"Validation loss improved. Saved {checkpoint_name}")
        else:
            early_stop_counter += 1
            print(f"Validation loss did not improve enough. Early Stopping Counter: {early_stop_counter}/{patience}")
            
        if early_stop_counter >= patience:
            print("Early Stopping triggered. Halting training.")
            break
        
        model.train() 

    print("end eval\n")

    # Chạy thử test xem chất lượng ntn
    print("="*50)
    model.load_state_dict(torch.load(checkpoint_name))
    model.eval() 
    
    test_loss = 0.0
    with torch.no_grad(): 
        for test_images, test_captions in test_dataloader:
            test_images = test_images.to(device)
            test_tokens = tokenizer(
                list(test_captions), padding=True, truncation=True, 
                max_length=77, return_tensors="pt"
            )
            
            test_img_feat, test_txt_feat, test_scale = model(
                test_images, test_tokens["input_ids"].to(device), test_tokens["attention_mask"].to(device)
            )
            
            batch_loss = contrastive_loss(test_img_feat, test_txt_feat, test_scale)
            test_loss += batch_loss.item()
            
    avg_test_loss = test_loss / len(test_dataloader)
    print(f"TEST: Loss = {avg_test_loss:.4f}")
    print("*"*50)
