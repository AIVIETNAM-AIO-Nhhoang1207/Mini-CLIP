import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm

from Data import MiniCLIPDataset
from models import get_encoders, get_checkpoint_name, ALL_CONFIGS
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


def run_benchmark(image_type, text_type, dataloader, dataset_size, device):
    """
    Chạy benchmark cho 1 config encoder.
    
    Returns:
        dict với các key: t2i_r1, t2i_r5, t2i_r10, i2t_r1, i2t_r5, i2t_r10
        Hoặc None nếu không tìm thấy checkpoint.
    """
    checkpoint_path = get_checkpoint_name(image_type, text_type)
    
    # Hỗ trợ checkpoint cũ best_mini_clip.pth cho baseline
    if not os.path.exists(checkpoint_path) and image_type == "resnet18" and text_type == "distilbert":
        if os.path.exists("best_mini_clip.pth"):
            checkpoint_path = "best_mini_clip.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"  SKIP: Checkpoint '{checkpoint_path}' not found.")
        return None
    
    # Tạo encoder + tokenizer
    image_encoder, text_encoder, tokenizer_name = get_encoders(image_type, text_type, embed_dim=512)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = MiniCLIP(image_encoder, text_encoder).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded: {checkpoint_path}")
    
    # Encode tất cả ảnh + text thành vector
    all_image_features = []
    all_text_features = []
    
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc=f"  [{image_type}+{text_type}]"):
            images = images.to(device)
            
            text_tokens = tokenizer(list(captions), padding=True, truncation=True, max_length=77, return_tensors="pt")
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            
            img_feat = F.normalize(model.image_encoder(images), p=2, dim=-1)
            txt_feat = F.normalize(model.text_encoder(text_tokens["input_ids"], text_tokens["attention_mask"]), p=2, dim=-1)
            
            all_image_features.append(img_feat.cpu())
            all_text_features.append(txt_feat.cpu())
    
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    
    # Cosine similarity
    similarity_matrix = all_text_features @ all_image_features.T
    
    # Recall@K
    results = {
        "t2i_r1":  calculate_recall_at_k(similarity_matrix, k=1),
        "t2i_r5":  calculate_recall_at_k(similarity_matrix, k=5),
        "t2i_r10": calculate_recall_at_k(similarity_matrix, k=10),
        "i2t_r1":  calculate_recall_at_k(similarity_matrix.T, k=1),
        "i2t_r5":  calculate_recall_at_k(similarity_matrix.T, k=5),
        "i2t_r10": calculate_recall_at_k(similarity_matrix.T, k=10),
    }
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Mini-CLIP với các encoder khác nhau")
    parser.add_argument("--image", type=str, default="resnet18", choices=["resnet18", "vit"],
                        help="Image encoder: resnet18 (default) hoặc vit")
    parser.add_argument("--text", type=str, default="distilbert", choices=["distilbert", "bert"],
                        help="Text encoder: distilbert (default) hoặc bert")
    parser.add_argument("--all", action="store_true",
                        help="Chạy benchmark tất cả config có checkpoint, in bảng so sánh")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check GPU
    if device.type == "cuda":
        try:
            torch.zeros(1).to(device)
        except Exception:
            device = torch.device("cpu")
    
    print(f"Device: {device.type.upper()}")

    # Data
    csv_path = "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/ML_test/test.csv"
    img_dirs = [
        "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-002/AIO_conquer/images",
        "Full_Data/AIO_conquer-20260619T153349Z-3-003/AIO_conquer/images"
    ]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MiniCLIPDataset(csv_file=csv_path, img_dir=img_dirs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    num_samples = len(dataset)

    if args.all:
        # ============================================================
        #  Mode: Benchmark TẤT CẢ config, in bảng so sánh
        # ============================================================
        print(f"\n{'='*80}")
        print(f"  BENCHMARK ALL CONFIGS ({num_samples} queries)")
        print(f"{'='*80}\n")
        
        all_results = {}
        
        for img_type, txt_type in ALL_CONFIGS:
            config_name = f"{img_type}+{txt_type}"
            print(f"\n--- {config_name} ---")
            result = run_benchmark(img_type, txt_type, dataloader, num_samples, device)
            if result is not None:
                all_results[config_name] = result
        
        # In bảng so sánh
        if all_results:
            print(f"\n\n{'='*90}")
            print(f"  COMPARISON TABLE ({num_samples} queries)")
            print(f"{'='*90}")
            header = f"{'Config':<25} | {'T2I R@1':>7} | {'T2I R@5':>7} | {'T2I R@10':>8} | {'I2T R@1':>7} | {'I2T R@5':>7} | {'I2T R@10':>8}"
            print(header)
            print("-" * len(header))
            
            for config_name, r in all_results.items():
                print(f"{config_name:<25} | {r['t2i_r1']:>6.2f}% | {r['t2i_r5']:>6.2f}% | {r['t2i_r10']:>7.2f}% | {r['i2t_r1']:>6.2f}% | {r['i2t_r5']:>6.2f}% | {r['i2t_r10']:>7.2f}%")
            
            print(f"{'='*90}")
        else:
            print("\nKhông tìm thấy checkpoint nào. Hãy train trước!")
    
    else:
        # ============================================================
        #  Mode: Benchmark 1 config
        # ============================================================
        config_name = f"{args.image}+{args.text}"
        print(f"\nBenchmark: {config_name}")
        print(f"Encoding {num_samples} samples...\n")
        
        result = run_benchmark(args.image, args.text, dataloader, num_samples, device)
        
        if result is not None:
            print(f"\n{'='*40}")
            print(f"  BENCHMARK RESULTS ({num_samples} queries)")
            print(f"  Config: {config_name}")
            print(f"{'='*40}")
            
            print("--- Text-to-Image (T2I) ---")
            print(f"  Recall@1  : {result['t2i_r1']:.2f}%")
            print(f"  Recall@5  : {result['t2i_r5']:.2f}%")
            print(f"  Recall@10 : {result['t2i_r10']:.2f}%")
            
            print("\n--- Image-to-Text (I2T) ---")
            print(f"  Recall@1  : {result['i2t_r1']:.2f}%")
            print(f"  Recall@5  : {result['i2t_r5']:.2f}%")
            print(f"  Recall@10 : {result['i2t_r10']:.2f}%")
            print(f"{'='*40}")
        else:
            print(f"\nERROR: Không tìm thấy checkpoint cho config '{config_name}'.")
            print(f"Hãy train trước: python train.py --image {args.image} --text {args.text}")
            sys.exit(1)
