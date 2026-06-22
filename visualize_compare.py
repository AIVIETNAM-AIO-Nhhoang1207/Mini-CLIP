import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Data import MiniCLIPDataset
from models import ImageEncoder, TextEncoder
from train import MiniCLIP

if __name__ == "__main__":
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for visualization: {device}")
    
    csv_path = "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/ML_test/test.csv"
    img_dirs = [
        "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-002/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-003/AIO_conquer/images"
    ]
    
    dataset = MiniCLIPDataset(csv_file=csv_path, img_dir=img_dirs, transform=transform)
    # Ép batch size nhỏ lại còn 6 để vẽ hình cho to, dễ nhìn
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
    
    # 1. Khởi tạo mô hình (Lúc này não nó còn trắng tinh, tạ random)
    image_encoder = ImageEncoder(embed_dim=512)
    text_encoder = TextEncoder(embed_dim=512)
    model = MiniCLIP(image_encoder, text_encoder).to(device)
    model.eval()
    
    images, captions = next(iter(dataloader))
    images = images.to(device)
    short_captions = [cap[:20] + "..." if len(cap) > 20 else cap for cap in captions]
    
    with torch.no_grad():
        text_tokens = tokenizer(list(captions), padding=True, truncation=True, max_length=77, return_tensors="pt")
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
        # B1: Trước khi train
        # Cho mô hình dự đoán khi chưa huấn luyện
        img_feat_before, txt_feat_before, _ = model(images, text_tokens["input_ids"], text_tokens["attention_mask"])
        sim_matrix_before = (F.normalize(img_feat_before, p=2, dim=-1) @ F.normalize(txt_feat_before, p=2, dim=-1).T).cpu().numpy()
        
        # B2: Tải trọng số đã huấn luyện vào mô hình
        print("Loading trained weights...")
        model.load_state_dict(torch.load("best_mini_clip.pth", map_location=device, weights_only=True))
        
        # B3: Sau khi train
        # Đoán lại với bộ não xịn
        img_feat_after, txt_feat_after, _ = model(images, text_tokens["input_ids"], text_tokens["attention_mask"])
        sim_matrix_after = (F.normalize(img_feat_after, p=2, dim=-1) @ F.normalize(txt_feat_after, p=2, dim=-1).T).cpu().numpy()

    images = images.cpu()
    
    # visualize
    # Chia làm 3 cột: 1 Cột Ảnh - 1 Biểu đồ Ngu - 1 Biểu đồ Khôn
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 3.5, 4], wspace=0.1)
    
    # Dán 6 tấm ảnh lại thành một dải dọc để ốp cho vừa với 6 cái hàng bên biểu đồ
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.485, 0.456, 0.406])
    
    img_list = []
    for i in range(6):
        img_tensor = images[i].clone().permute(1, 2, 0) 
        img_np = (img_tensor * std + mean).clamp(0, 1).numpy()
        img_list.append(img_np)
        
    stacked_images = np.vstack(img_list) # Nối dọc các ảnh lại
    
    # Cột Ảnh
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(stacked_images)
    ax_img.axis('off') 
        
    # Vẽ biểu đồ trước khi train
    ax_before = fig.add_subplot(gs[0, 1])
    sns.heatmap(sim_matrix_before, annot=True, cmap="coolwarm", fmt=".2f", 
                xticklabels=short_captions, yticklabels=False, ax=ax_before, cbar=False)
    ax_before.set_title("BEFORE Training (Random Weights)", fontsize=13, color="red", pad=15, fontweight='bold')
    ax_before.set_xticklabels(ax_before.get_xticklabels(), rotation=45, ha='right')

    # Vẽ biểu đồ sau khi train
    ax_after = fig.add_subplot(gs[0, 2])
    sns.heatmap(sim_matrix_after, annot=True, cmap="coolwarm", fmt=".2f", 
                xticklabels=short_captions, yticklabels=False, ax=ax_after)
    ax_after.set_title("AFTER Training (Trained Weights)", fontsize=13, color="green", pad=15, fontweight='bold')
    ax_after.set_xticklabels(ax_after.get_xticklabels(), rotation=45, ha='right')

    # Điều chỉnh layout tự động
    fig.set_layout_engine('tight')
    plt.show()
