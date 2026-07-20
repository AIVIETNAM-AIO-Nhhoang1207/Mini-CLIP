<div align="center">

# Mini-CLIP

### Mô hình gọn nhẹ theo kiến trúc CLIP cho tác vụ truy vấn ảnh–văn bản trong thời trang

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-Educational-lightgrey)](#giấy-phép)

Mini-CLIP là một triển khai thu nhỏ lấy cảm hứng từ CLIP, được thiết kế cho việc học tập và thử nghiệm truy vấn đa thức ảnh–văn bản trên dữ liệu thời trang.

[Video Demo](https://www.youtube.com/watch?v=unUhRajTKJU) •
[Model Checkpoint](./best_mini_clip.pth) •
[Mã nguồn](https://github.com/AIVIETNAM-AIO-Nhhoang1207/Mini-CLIP)

</div>

---

## Tổng quan

Mini-CLIP học một không gian nhúng chung (shared embedding space) cho cả hình ảnh và văn bản.

Trong quá trình huấn luyện, các cặp ảnh–chú thích khớp nhau sẽ được kéo lại gần nhau, trong khi các cặp không khớp sẽ bị đẩy ra xa. Sau khi huấn luyện, mô hình có thể thực hiện hai tác vụ truy vấn:

* **Truy vấn Văn bản sang Ảnh (Text-to-Image Retrieval):** cho trước một chú thích, truy xuất các hình ảnh liên quan nhất.
* **Truy vấn Ảnh sang Văn bản (Image-to-Text Retrieval):** cho trước một hình ảnh, truy xuất các chú thích liên quan nhất.

Dự án tập trung vào việc triển khai các ý tưởng cốt lõi đằng sau CLIP bằng cách sử dụng các bộ mã hóa (encoders) tiền huấn luyện nhỏ gọn và dễ tiếp cận hơn.

---

## Demo

Video minh họa cho dự án có sẵn trên YouTube:

[![Mini-CLIP Demo](https://img.youtube.com/vi/unUhRajTKJU/maxresdefault.jpg)](https://www.youtube.com/watch?v=unUhRajTKJU)

---

## Kiến trúc mô hình

Mini-CLIP chứa hai bộ mã hóa độc lập:

| Thành phần         | Kiến trúc                  | Kích thước đầu ra |
| ------------------ | -------------------------- | ----------------: |
| Bộ mã hóa Ảnh      | ResNet-18 tiền huấn luyện  |               512 |
| Bộ mã hóa Văn bản  | DistilBERT tiền huấn luyện |               512 |
| Độ tương đồng      | Cosine similarity          |                 — |
| Mục tiêu huấn luyện| Symmetric contrastive loss |                 — |

### Bộ mã hóa Ảnh (Image Encoder)

Bộ mã hóa ảnh sử dụng backbone ResNet-18 tiền huấn luyện.

Lớp phân loại gốc bị loại bỏ, và biểu diễn hình ảnh thu được được chiếu vào một không gian nhúng 512 chiều.

```text
Ảnh đầu vào
    │
    ▼
ResNet-18 tiền huấn luyện
    │
    ▼
Đặc trưng hình ảnh
    │
    ▼
Lớp chiếu tuyến tính (Linear projection)
    │
    ▼
Vector nhúng ảnh 512 chiều
```

### Bộ mã hóa Văn bản (Text Encoder)

Bộ mã hóa văn bản sử dụng `distilbert-base-uncased`.

Biểu diễn của token đầu ra đầu tiên (CLS token) được đưa qua một lớp chiếu tuyến tính để thu được vector nhúng văn bản 512 chiều.

```text
Chú thích đầu vào
    │
    ▼
DistilBERT tokenizer
    │
    ▼
DistilBERT tiền huấn luyện
    │
    ▼
Biểu diễn văn bản
    │
    ▼
Lớp chiếu tuyến tính (Linear projection)
    │
    ▼
Vector nhúng văn bản 512 chiều
```

### Không gian nhúng chung (Shared Embedding Space)

Cả vector nhúng ảnh và văn bản đều được chuẩn hóa L2 (L2-normalized) trước khi tính toán độ tương đồng.

```text
Ảnh     ──► Bộ mã hóa Ảnh     ──► Nhúng Ảnh     ──┐
                                                 ├──► Ma trận tương đồng (Similarity Matrix)
Văn bản ──► Bộ mã hóa Văn bản ──► Nhúng Văn bản ──┘
```

---

## Hàm mất mát tương phản (Contrastive Learning Objective)

Đối với một batch chứa các cặp ảnh–chú thích tương ứng:

$$ (I_1,T_1), (I_2,T_2), \ldots, (I_N,T_N) $$

mô hình tính toán độ tương đồng giữa từng vector nhúng ảnh và từng vector nhúng văn bản.

Các phần tử trên đường chéo chính của ma trận tương đồng đại diện cho các cặp khớp nhau. Các phần tử còn lại đại diện cho các cặp không khớp.

Mini-CLIP áp dụng hàm mất mát Cross-Entropy theo cả hai chiều truy vấn:

$$ \mathcal{L} = rac{\mathcal{L}_{image
ightarrow text} + \mathcal{L}_{text
ightarrow image}}{2} $$

Một tham số nhiệt độ (temperature parameter) có thể học được được dùng để điều chỉnh quy mô của logits tương đồng.

---

## Cấu trúc thư mục

```text
Mini-CLIP/
├── Data.py                 # Dataset để tải hình ảnh và chú thích
├── models.py               # Các bộ mã hóa ResNet-18 và DistilBERT
├── train.py                # Huấn luyện, đánh giá val và dừng sớm (early stopping)
├── benchmark.py            # Đánh giá chỉ số Recall@K
├── visualize.py            # Trực quan hóa độ tương đồng
├── visualize_compare.py    # Trực quan hóa so sánh
├── best_mini_clip.pth      # Checkpoint mô hình đã huấn luyện
├── .gitignore
├── .gitattributes
└── README.md
```

---

## Định dạng tập dữ liệu

Dự án yêu cầu các file CSV chứa tối thiểu hai cột:

```csv
image_filename,caption
image_0001.jpg,a black short sleeve shirt
image_0002.jpg,a red floral summer dress
image_0003.jpg,white running shoes
```

Các cột bắt buộc:

| Cột              | Mô tả                                       |
| ---------------- | ------------------------------------------- |
| `image_filename` | Tên của file ảnh tương ứng                  |
| `caption`        | Mô tả bằng ngôn ngữ tự nhiên của hình ảnh  |

Bộ tải dữ liệu (Dataset loader) có thể tìm kiếm hình ảnh trên một hoặc nhiều thư mục chứa ảnh.

Cấu trúc thư mục dữ liệu gợi ý:

```text
Full_Data/
└── dataset/
    ├── images/
    │   ├── image_0001.jpg
    │   ├── image_0002.jpg
    │   └── ...
    └── ML_test/
        ├── train.csv
        ├── val.csv
        └── test.csv
```

> Các đường dẫn dữ liệu hiện đang được cấu hình trực tiếp bên trong `train.py`, `benchmark.py`, `visualize.py` và `visualize_compare.py`. Hãy cập nhật chúng cho phù hợp với vị trí dữ liệu trên máy bạn trước khi chạy các script.

---

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/AIVIETNAM-AIO-Nhhoang1207/Mini-CLIP.git
cd Mini-CLIP
```

### 2. Tạo môi trường ảo

Sử dụng `venv`:

```bash
python -m venv .venv
```

Kích hoạt môi trường trên Windows:

```bash
.venv\Scripts\activate
```

Kích hoạt môi trường trên Linux hoặc macOS:

```bash
source .venv/bin/activate
```

### 3. Cài đặt các thư viện phụ thuộc

```bash
pip install torch torchvision transformers pandas pillow tqdm matplotlib seaborn
```

Các thư viện chính:

* Python 3.9+
* PyTorch
* Torchvision
* Hugging Face Transformers
* Pandas
* Pillow
* tqdm
* Matplotlib
* Seaborn

---

## Cấu hình

Trước khi huấn luyện hoặc đánh giá, hãy cập nhật đường dẫn tập dữ liệu bên trong các script Python.

Ví dụ cấu hình:

```python
train_csv_path = "path/to/train.csv"
val_csv_path = "path/to/val.csv"
test_csv_path = "path/to/test.csv"

img_dirs = [
    "path/to/images"
]
```

Dự án tự động sử dụng CUDA khi có GPU tương thích:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Huấn luyện

Chạy lệnh:

```bash
python train.py
```

Cấu hình huấn luyện mặc định sử dụng:

| Tham số                        | Trị số  |
| ------------------------------ | ------: |
| Kích thước batch (Batch size)  |      64 |
| Số epoch tối đa                |     100 |
| Kích thước không gian nhúng    |     512 |
| Tốc độ học của backbone        |  `1e-5` |
| Tốc độ học của lớp chiếu       |  `1e-3` |
| Kiên nhẫn dừng sớm (Patience)  |       5 |
| Độ cải thiện val tối thiểu     |   0.005 |

Backbone hình ảnh và văn bản sử dụng tốc độ học nhỏ hơn so với các lớp chiếu và tham số logit scale có thể học.

Mô hình tốt nhất sẽ được lưu tại:

```text
best_mini_clip.pth
```

Quá trình huấn luyện sẽ dừng sớm khi validation loss không cải thiện đủ tốt trong 5 epoch liên tiếp.

---

## Đánh giá

Chạy lệnh:

```bash
python benchmark.py
```

Script đánh giá (benchmark) sẽ:

1. Tải checkpoint `best_mini_clip.pth`.
2. Mã hóa toàn bộ hình ảnh và chú thích trong tập test.
3. Chuẩn hóa các vector nhúng.
4. Tính toán ma trận tương đồng ảnh–văn bản.
5. Đánh giá hiệu năng truy vấn bằng chỉ số Recall@K.

---

## Chỉ số đánh giá

Dự án đánh giá khả năng truy vấn theo cả hai chiều.

### Truy vấn Văn bản sang Ảnh (Text-to-Image Retrieval)

Một câu chú thích được dùng làm câu truy vấn (query), và mô hình sẽ truy xuất các hình ảnh giống nhất.

* T2I Recall@1
* T2I Recall@5
* T2I Recall@10

### Truy vấn Ảnh sang Văn bản (Image-to-Text Retrieval)

Một hình ảnh được dùng làm câu truy vấn, và mô hình sẽ truy xuất các chú thích giống nhất.

* I2T Recall@1
* I2T Recall@5
* I2T Recall@10

Công thức tính Recall@K:

$$ Recall@K = rac{	ext{số lượng truy vấn có kết quả đúng nằm trong top }K}{	ext{tổng số lượng truy vấn}} 	imes 100\% $$

Chỉ số Recall@K càng cao thể hiện hiệu năng truy vấn càng tốt.

---

## Trực quan hóa

### Trực quan hóa độ tương đồng

Chạy lệnh:

```bash
python visualize.py
```

Script này trực quan hóa điểm số tương đồng giữa ảnh và văn bản, giúp kiểm tra xem các cặp ảnh–chú thích khớp nhau có nhận được điểm tương đồng cao hơn các cặp không khớp hay không.

### So sánh mô hình

Chạy lệnh:

```bash
python visualize_compare.py
```

Script này cung cấp thêm hình ảnh trực quan so sánh kết quả truy vấn hoặc độ tương đồng.

Trước khi chạy các script trực quan hóa, hãy đảm bảo rằng:

* Đường dẫn tập dữ liệu đã chính xác;
* File `best_mini_clip.pth` đã tồn tại;
* Tất cả các thư viện phụ thuộc đã được cài đặt.

---

## Sử dụng Checkpoint đã tiền huấn luyện

Repository đã đi kèm sẵn checkpoint đã huấn luyện:

```text
best_mini_clip.pth
```

Bạn có thể tải checkpoint này như sau:

```python
import torch

from models import ImageEncoder, TextEncoder
from train import MiniCLIP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_encoder = ImageEncoder(embed_dim=512)
text_encoder = TextEncoder(embed_dim=512)

model = MiniCLIP(
    image_encoder=image_encoder,
    text_encoder=text_encoder
).to(device)

state_dict = torch.load(
    "best_mini_clip.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()
```

Kiến trúc mô hình được dùng để tải checkpoint phải trùng khớp với kiến trúc đã sử dụng trong quá trình huấn luyện.

---

## Quy trình hoạt động của dự án

```text
Hình ảnh thời trang và chú thích
            │
            ▼
     Dataset loader
            │
            ▼
Bộ mã hóa ảnh và văn bản
            │
            ▼
 Vector nhúng chuẩn hóa
            │
            ▼
Ma trận tương đồng ảnh–văn bản
            │
            ▼
Symmetric contrastive loss
            │
            ▼
   Mini-CLIP đã huấn luyện
            │
            ▼
Truy vấn Văn bản-sang-Ảnh / Ảnh-sang-Văn bản
```

---

## Các tính năng hiện có

* Bộ mã hóa ảnh ResNet-18 tiền huấn luyện
* Bộ mã hóa văn bản DistilBERT tiền huấn luyện
* Không gian nhúng chung 512 chiều
* Tham số nhiệt độ tương đồng có thể học
* Hàm mất mát tương phản đối xứng ảnh–văn bản
* Dừng sớm dựa trên tập validation
* Tự động chọn thiết bị CPU/CUDA
* Đánh giá truy vấn Văn bản-sang-Ảnh
* Đánh giá truy vấn Ảnh-sang-Văn bản
* Các chỉ số Recall@1, Recall@5 và Recall@10
* Trực quan hóa độ tương đồng
* Đi kèm checkpoint đã huấn luyện sẵn

---

## Hạn chế

Triển khai hiện tại chủ yếu phục vụ cho mục đích học tập và thử nghiệm.

Các hạn chế hiện tại bao gồm:

* Đường dẫn dữ liệu bị hard-code bên trong các script;
* Chưa hỗ trợ cấu hình qua dòng lệnh (command-line);
* Mô hình bị chuyên biệt hóa theo tập dữ liệu huấn luyện;
* Quá trình truy vấn giả định các cặp ảnh–chú thích đã được dóng hàng trong tập đánh giá;
* Chưa đi kèm ứng dụng tìm kiếm tương tác độc lập;
* Repository chưa cung cấp file `requirements.txt` cố định phiên bản;
* Quá trình huấn luyện có thể tốn nhiều bộ nhớ GPU do cả hai bộ mã hóa đều được tinh chỉnh (fine-tune).

---

## Hướng phát triển tiếp theo

Các cải tiến trong tương lai có thể bao gồm:

* Chuyển cấu hình sang các tham số dòng lệnh hoặc file YAML;
* Bổ sung file `requirements.txt`;
* Thêm các script suy luận (inference) cho ảnh và chú thích tùy chỉnh;
* Tạo ứng dụng truy vấn demo bằng Streamlit hoặc Gradio;
* So sánh Mini-CLIP với các mô hình CLIP và SigLIP tiền huấn luyện gốc;
* Hỗ trợ các bộ mã hóa ảnh và văn bản thay thế khác;
* Ghi lại nhật ký thực nghiệm (logging) bằng TensorBoard hoặc Weights & Biases;
* Báo cáo kích thước mô hình, thời gian suy luận và dung lượng GPU tiêu thụ;
* Bổ sung chỉ số Mean Reciprocal Rank (MRR) và Median Rank;
* Hỗ trợ nhiều chú thích cho mỗi hình ảnh;
* Bổ sung unit test và tích hợp liên tục (CI).

---

## Mục đích giáo dục

Dự án này được phát triển như một bài triển khai học tập về học tương phản đa thức (multimodal contrastive learning).

Dự án minh họa:

* Cách kết hợp các mô hình thị giác và ngôn ngữ tiền huấn luyện;
* Cách ánh xạ hình ảnh và văn bản vào một không gian vector chung;
* Cách học tương phản căn chỉnh các mẫu đa thức tương ứng;
* Cách đánh giá các hệ thống truy vấn đa phương thức.

Dự án được lấy cảm hứng từ ý tưởng cốt lõi của CLIP nhưng không phải là triển khai chính thức từ OpenAI.

---

## Tài liệu tham khảo

* Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021.
* Tài liệu PyTorch.
* Tài liệu Torchvision ResNet.
* Tài liệu Hugging Face Transformers.
* DistilBERT: *DistilBERT, a distilled version of BERT*.

---

**Nguyễn Huy Hoàng**

* GitHub: [AIVIETNAM-AIO-Nhhoang1207](https://github.com/AIVIETNAM-AIO-Nhhoang1207)
* Demo dự án: [YouTube](https://www.youtube.com/watch?v=unUhRajTKJU)

---
Repository này được cung cấp cho mục đích học tập và nghiên cứu.
---

<div align="center">

Nếu dự án này hữu ích, hãy cân nhắc tặng repository một ngôi sao (star).

</div>
