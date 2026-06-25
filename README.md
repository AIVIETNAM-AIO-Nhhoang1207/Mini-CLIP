# 📊 Mini-CLIP Benchmark Report

Báo cáo này trình bày kết quả đánh giá (benchmark) so sánh trực tiếp hiệu năng giữa hai kiến trúc mô hình cho dự án **Mini-CLIP** trên tập test (18,089 queries).

Hai kiến trúc được đem ra so sánh bao gồm:
1. **ResNet18 + DistilBERT** (Bản gốc)
2. **ViT + BERT** (Bản mới)

---

## 📈 Kết quả đánh giá chi tiết

### 1. Text-to-Image (Tìm ảnh bằng văn bản)

| Chỉ số đánh giá | ResNet18 + DistilBERT | ViT + BERT | Chênh lệch (ViT so với ResNet) |
| :--- | :---: | :---: | :---: |
| **Recall@1**  | **21.68%** | 21.40% | 🔻 0.28% |
| **Recall@5**  | **51.00%** | 49.62% | 🔻 1.38% |
| **Recall@10** | **65.22%** | 63.80% | 🔻 1.42% |

### 2. Image-to-Text (Tìm văn bản bằng ảnh)

| Chỉ số đánh giá | ResNet18 + DistilBERT | ViT + BERT | Chênh lệch (ViT so với ResNet) |
| :--- | :---: | :---: | :---: |
| **Recall@1**  | **23.99%** | 23.10% | 🔻 0.89% |
| **Recall@5**  | **53.70%** | 51.74% | 🔻 1.96% |
| **Recall@10** | **66.93%** | 65.64% | 🔻 1.29% |

---

## ⏱️ Hiệu năng trích xuất đặc trưng (Feature Extraction)
- **ResNet18 + DistilBERT**: Trích xuất khá nhanh (khoảng ~5.05 batches/s), hoàn thành bài test trong khoảng gần 1 phút.
- **ViT + BERT**: Quá trình trích xuất chậm hơn rõ rệt (khoảng ~1.90 batches/s), mất tới khoảng 2 phút rưỡi để hoàn thành.

---

## 💡 Nhận xét & Kết luận

**Nhận xét tổng quan**
Dựa trên các chỉ số **Recall** ở cả hai chiều (Text-to-Image và Image-to-Text), phiên bản **ResNet18 + DistilBERT** gốc đang thể hiện **hiệu năng nhỉnh hơn một chút** (chênh khoảng 1 - 2%) so với kiến trúc **ViT + BERT** trên tập dataset hiện tại. 
Bên cạnh đó, tốc độ xử lý của ResNet cũng tỏ ra vượt trội và nhẹ nhàng hơn.

**Nguyên nhân có thể:**
1. **Dataset Size**: ViT (Vision Transformer) thường yêu cầu lượng dữ liệu khổng lồ (hàng triệu ảnh) để học được đặc trưng tốt so với mạng CNN. Với lượng dữ liệu nhỏ, ResNet với inductive bias tốt sẽ học nhanh và hội tụ tốt hơn.
2. **Hyperparameters**: Mô hình ViT+BERT hiện tại có thể cần được tinh chỉnh (finetune) thêm về learning rate, số epoch, hoặc data augmentation so với bộ thông số mặc định cũ.
3. **Kích thước mô hình**: BERT to hơn DistilBERT, ViT cũng nặng hơn ResNet18. Việc model lớn đi kèm dataset nhỏ rất dễ dẫn đến hiện tượng khó học hoặc overfitting (đặc biệt trong quá trình Contrastive Learning).

**Đề xuất bước tiếp theo**
Nếu bạn vẫn muốn sử dụng kiến trúc **ViT+BERT**, bạn có thể:
- Áp dụng Data Augmentation mạnh hơn cho ảnh đầu vào.
- Train thêm nhiều Epochs hơn với Learning Rate nhỏ hơn.
- Đóng băng (Freeze) một số layers đầu tiên của ViT/BERT và chỉ train các layer cuối để tránh quá tải đối với tập dữ liệu nhỏ.
