import torch


def calculate_recall_at_k(similarity_matrix, k):
    """
    Tính Recall@K.
    Lấy từng câu truy vấn, tìm top K kết quả giống nhất.
    Nếu đáp án đúng nằm trong top K thì coi là đoán trúng.
    """
    num_queries = similarity_matrix.shape[0]

    # Lấy index của top K giống nhất
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)

    # Đáp án đúng (vị trí thứ i khớp với thứ i)
    ground_truth = torch.arange(num_queries, device=similarity_matrix.device).view(-1, 1)

    # Đếm số câu đoán trúng
    correct = (top_k_indices == ground_truth).any(dim=1).sum().item()

    return (correct / num_queries) * 100


def print_recalls(similarity_matrix, name=""):
    """In kết quả Recall@1, @5, @10 cho cả T2I và I2T."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    print("  --- Text-to-Image (T2I) ---")
    r1_t2i = calculate_recall_at_k(similarity_matrix, k=1)
    r5_t2i = calculate_recall_at_k(similarity_matrix, k=5)
    r10_t2i = calculate_recall_at_k(similarity_matrix, k=10)
    print(f"    Recall@1  : {r1_t2i:.2f}%")
    print(f"    Recall@5  : {r5_t2i:.2f}%")
    print(f"    Recall@10 : {r10_t2i:.2f}%")

    print("  --- Image-to-Text (I2T) ---")
    r1_i2t = calculate_recall_at_k(similarity_matrix.T, k=1)
    r5_i2t = calculate_recall_at_k(similarity_matrix.T, k=5)
    r10_i2t = calculate_recall_at_k(similarity_matrix.T, k=10)
    print(f"    Recall@1  : {r1_i2t:.2f}%")
    print(f"    Recall@5  : {r5_i2t:.2f}%")
    print(f"    Recall@10 : {r10_i2t:.2f}%")
    print(f"{'='*50}")

    return {
        "name": name,
        "t2i": {"R@1": r1_t2i, "R@5": r5_t2i, "R@10": r10_t2i},
        "i2t": {"R@1": r1_i2t, "R@5": r5_i2t, "R@10": r10_i2t},
    }


def collate_fn_pil(batch):
    """Collate function trả về PIL Images (không transform) để model tự xử lý."""
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    return images, captions


def get_device():
    """Lấy device, kiểm tra CUDA có hoạt động không."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.zeros(1).to(device)
        except Exception:
            device = torch.device("cpu")
    return device


# Default paths
DEFAULT_CSV = "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/ML_test/test.csv"
DEFAULT_IMG_DIRS = [
    "Full_Data/AIO_conquer-20260619T153349Z-3-001/AIO_conquer/images",
    "Full_Data/AIO_conquer-20260619T153349Z-3-002/AIO_conquer/images",
    "Full_Data/AIO_conquer-20260619T153349Z-3-003/AIO_conquer/images",
]
