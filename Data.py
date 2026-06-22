import torch
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset

class MiniCLIPDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dirs = img_dir if isinstance(img_dir, list) else [img_dir]
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx]['image_filename'])
        
        img_path = None
        for d in self.img_dirs:
            p = os.path.join(d, img_name)
            if os.path.exists(p):
                img_path = p
                break
                
        if img_path is None:
            img_path = os.path.join(self.img_dirs[0], img_name)
            
        image = Image.open(img_path).convert('RGB')
        
        caption = str(self.df.iloc[idx]['caption'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, caption
