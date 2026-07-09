# Mini-CLIP

Mini-CLIP là một phiên bản nhẹ của [CLIP](https://openai.com/research/clip) (Contrastive Language-Image Pre-training) cho bài toán truy xuất ảnh–văn bản thời trang. Cấu hình mặc định dùng **ResNet-18** làm Image Encoder và **DistilBERT** làm Text Encoder; ngoài ra còn hỗ trợ cấu hình thay thế **ViT-B/16 + BERT-base** để so sánh kiến trúc.

Repo này gộp ba nhánh trước đây (baseline, benchmark 3 model, và so sánh ViT+BERT) thành một codebase duy nhất, đồng thời benchmark so sánh với các model lớn hơn (CLIP, OpenCLIP, SigLIP).

## Kiến trúc

| Component      | Mini-CLIP (baseline)    | Mini-CLIP (variant) | CLIP (OpenAI)    | OpenCLIP (LAION) | SigLIP (Google)          |
|----------------|-------------------------|---------------------|------------------|------------------|--------------------------|
| Image Encoder  | ResNet-18               | ViT-B/16 (frozen)   | ViT-B/32         | ViT-B/32         | ViT-B/16                 |
| Text Encoder   | DistilBERT (66M params) | BERT-base (frozen)  | Transformer      | Transformer      | Transformer              |
| Loss Function  | InfoNCE (Symmetric)     | InfoNCE (Symmetric) | InfoNCE          | InfoNCE          | Sigmoid Loss             |
| Embedding Dim  | 512                     | 512                 | 512              | 512              | 768                      |

Cấu hình kiến trúc được chọn qua factory `get_encoders()` trong `models.py`, đặt bằng biến `IMAGE_TYPE` (`resnet18`/`vit`) và `TEXT_TYPE` (`distilbert`/`bert`) ở đầu `train.py`. Checkpoint được đặt tên theo cấu hình, ví dụ `best_resnet18_distilbert.pth`.

## Cấu trúc thư mục

```
Mini-CLIP/
├── Data.py                        # MiniCLIPDataset (đọc CSV image/caption)
├── models.py                      # ImageEncoder/TextEncoder (+ ViT/BERT), MiniCLIP, contrastive_loss, get_encoders
├── train.py                       # Training + early stopping (chọn kiến trúc qua IMAGE_TYPE/TEXT_TYPE)
├── benchmark/                     # Benchmark scripts
│   ├── utils.py                   # Recall@K, shared utilities
│   ├── benchmark_mini_clip.py     # Benchmark Mini-CLIP
│   ├── benchmark_clip.py          # Benchmark OpenAI CLIP (ViT-B/32)
│   ├── benchmark_openclip.py      # Benchmark OpenCLIP (ViT-B/32, LAION-2B)
│   └── benchmark_siglip.py        # Benchmark SigLIP (siglip-base-patch16-224)
├── visualize/                     # Visualization scripts
│   ├── similarity_matrix.py       # Cosine similarity heatmap
│   └── before_after.py            # Before/After training comparison
├── scripts/
│   └── run_all_benchmarks.py      # Run all benchmarks + bảng so sánh
└── README.md
```

## Cài đặt

```bash
pip install torch torchvision transformers pandas pillow tqdm matplotlib seaborn
```

## Sử dụng

### Train model

```bash
python train.py            # mặc định ResNet-18 + DistilBERT
```

Để train cấu hình ViT-B/16 + BERT-base, đặt `IMAGE_TYPE = "vit"` và `TEXT_TYPE = "bert"` ở đầu `train.py` rồi chạy lại.

### Benchmark

Chạy benchmark cho từng model riêng lẻ:

```bash
python -m benchmark.benchmark_mini_clip
python -m benchmark.benchmark_clip
python -m benchmark.benchmark_openclip
python -m benchmark.benchmark_siglip
```

Hoặc chạy tất cả cùng lúc và xem bảng so sánh:

```bash
python scripts/run_all_benchmarks.py
```

### Visualization

```bash
python -m visualize.similarity_matrix
python -m visualize.before_after
```

## Metrics

Sử dụng **Recall@K** (K = 1, 5, 10) cho cả hai chiều:
- **Text-to-Image (T2I)**: Nhập câu text, tìm ảnh khớp nhất.
- **Image-to-Text (I2T)**: Nhập ảnh, tìm câu text khớp nhất.
