# Mini-CLIP

Mini-CLIP là một phiên bản nhẹ của [CLIP](https://openai.com/research/clip) (Contrastive Language-Image Pre-training), sử dụng **ResNet-18** làm Image Encoder và **DistilBERT** làm Text Encoder.

Project này được xây dựng để học cách triển khai contrastive learning cho bài toán Image-Text Retrieval, đồng thời benchmark so sánh với các model lớn hơn.

## Kiến trúc

| Component      | Mini-CLIP               | CLIP (OpenAI)    | SigLIP (Google)          |
|----------------|-------------------------|------------------|--------------------------|
| Image Encoder  | ResNet-18               | ViT-B/32         | ViT-B/16                 |
| Text Encoder   | DistilBERT (66M params) | Transformer      | Transformer              |
| Loss Function  | InfoNCE (Symmetric)     | InfoNCE          | Sigmoid Loss             |
| Embedding Dim  | 512                     | 512              | 768                      |

## Cấu trúc thư mục

```
Mini_CLIP/
├── data/                          # Dataset module
│   └── dataset.py                 # MiniCLIPDataset
├── models/                        # Model definitions
│   └── mini_clip.py               # ImageEncoder + TextEncoder + MiniCLIP
├── benchmark/                     # Benchmark scripts
│   ├── utils.py                   # Recall@K, shared utilities
│   ├── benchmark_mini_clip.py     # Benchmark Mini-CLIP
│   ├── benchmark_clip.py          # Benchmark OpenAI CLIP
│   ├── benchmark_openclip.py      # Benchmark OpenCLIP
│   └── benchmark_siglip.py        # Benchmark SigLIP
├── visualize/                     # Visualization scripts
│   ├── similarity_matrix.py       # Cosine similarity heatmap
│   └── before_after.py            # Before/After training comparison
├── scripts/                       # Utility scripts
│   ├── run_all_benchmarks.py      # Run all benchmarks at once
├── train.py                       # Training script
├── requirements.txt               # Dependencies
└── README.md
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### Train model

```bash
python train.py
```

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
