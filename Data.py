import os
import csv
from PIL import Image
from torch.utils.data import Dataset


class MiniCLIPDataset(Dataset):
    """
    Dataset cho Mini-CLIP.
    Đọc file CSV chứa cặp (image, caption), trả về (image, caption).
    Hỗ trợ tìm ảnh trong nhiều thư mục (img_dir có thể là list).
    """

    def __init__(self, csv_file, img_dir, transform=None):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.data = list(reader)
        self.img_dirs = img_dir if isinstance(img_dir, list) else [img_dir]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # Lấy tên file ảnh từ CSV
        if "image_filename" in row:
            img_name = str(row["image_filename"])
        elif "image_path" in row:
            img_name = str(row["image_path"]).split("/")[-1]
        else:
            raise KeyError("CSV must contain either 'image_filename' or 'image_path'")

        # Tìm ảnh trong các thư mục
        img_path = None
        for d in self.img_dirs:
            p = os.path.join(d, img_name)
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
            img_path = os.path.join(self.img_dirs[0], img_name)

        image = Image.open(img_path).convert("RGB")
        caption = str(row["caption"])

        if self.transform:
            image = self.transform(image)

        return image, caption
