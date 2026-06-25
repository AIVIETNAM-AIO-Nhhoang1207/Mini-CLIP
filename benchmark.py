import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from tqdm import tqdm

from Data import MiniCLIPDataset
from models import ImageEncoderViT, TextEncoderBERT
from train import MiniCLIP


def calculate_recall_at_k(similarity_matrix, k):
    """
    Tính điểm Recall@K.
    Cơ chế: Lấy từng câu truy vấn, kiếm top K thằng giống nhất. 
    Nếu đáp án đúng lọt top K thì coi như là đoán trúng.
    """
    num_queries = similarity_matrix.shape[0]
    
    # Lấy index của top K thằng giống nhất
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
    
    # Đáp án đúng (do không đảo lộn data nên vị trí thứ i sẽ khớp với thứ i)
    ground_truth = torch.arange(num_queries, device=similarity_matrix.device).view(-1, 1)
    
    # Đếm tổng số câu đoán trúng đáp án
    correct = (top_k_indices == ground_truth).any(dim=1).sum().item()
    
    return (correct / num_queries) * 100


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check cẩn thận xem máy có nhận card GPU không
    if device.type == "cuda":
        try:
            torch.zeros(1).to(device)
        except Exception:
            device = torch.device("cpu")
    
    print(f"Device: {device.type.upper()}")

    csv_path = "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/ML_test/test.csv"
    img_dirs = [
        "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-002/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-003/AIO_conquer/images"
    ]
    
    # Data
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MiniCLIPDataset(csv_file=csv_path, img_dir=img_dirs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Load model
    image_encoder = ImageEncoderViT(embed_dim=512)
    text_encoder = TextEncoderBERT(embed_dim=512)
    model = MiniCLIP(image_encoder, text_encoder).to(device)
    
    checkpoint_path = "best_vit_bert.pth"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Cannot find '{checkpoint_path}'. Please train the model first.")
        sys.exit(1)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded weights: {checkpoint_path}")
    
    # B1: Cho ảnh và chữ thành vector
    all_image_features = []
    all_text_features = []
    
    print(f"Encoding {len(dataset)} samples...")
    
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            
            text_tokens = tokenizer(list(captions), padding=True, truncation=True, max_length=77, return_tensors="pt")
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            
            img_feat = F.normalize(model.image_encoder(images), p=2, dim=-1)
            txt_feat = F.normalize(model.text_encoder(text_tokens["input_ids"], text_tokens["attention_mask"]), p=2, dim=-1)
            
            all_image_features.append(img_feat.cpu())
            all_text_features.append(txt_feat.cpu())
    
    all_image_features = torch.cat(all_image_features, dim=0) 
    all_text_features = torch.cat(all_text_features, dim=0)    
    
    # B2: tính cosine similarity 
    # Đã normalize nên giờ chỉ việc nhân 2 ma trận với nhau là ra cosine
    similarity_matrix = all_text_features @ all_image_features.T
    
    # B3: Tính Recall
    print(f"\n{'='*40}")
    print(f"  BENCHMARK RESULTS ({len(dataset)} queries)")
    print(f"{'='*40}")
    
    # Nhập chữ tìm ảnh
    print("--- Text-to-Image (T2I) ---")
    r1_t2i = calculate_recall_at_k(similarity_matrix, k=1)
    r5_t2i = calculate_recall_at_k(similarity_matrix, k=5)
    r10_t2i = calculate_recall_at_k(similarity_matrix, k=10)
    print(f"  Recall@1  : {r1_t2i:.2f}%")
    print(f"  Recall@5  : {r5_t2i:.2f}%")
    print(f"  Recall@10 : {r10_t2i:.2f}%")
    
    # Nhét ảnh vào tìm chữ
    print("\n--- Image-to-Text (I2T) ---")
    r1_i2t = calculate_recall_at_k(similarity_matrix.T, k=1)
    r5_i2t = calculate_recall_at_k(similarity_matrix.T, k=5)
    r10_i2t = calculate_recall_at_k(similarity_matrix.T, k=10)
    print(f"  Recall@1  : {r1_i2t:.2f}%")
    print(f"  Recall@5  : {r5_i2t:.2f}%")
    print(f"  Recall@10 : {r10_i2t:.2f}%")
    print(f"{'='*40}")
