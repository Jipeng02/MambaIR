import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class LabColorizationDataset(Dataset):
    def __init__(self, img_dir, img_size=256, max_samples=None):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('png','jpg','jpeg'))]
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]  # 只保留前 max_samples 张图片
        self.transform = T.Compose([
            T.Resize((img_size,img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)  # [3,H,W], [0,1]
        img_np = np.array(img.permute(1,2,0)) * 255
        img_lab = rgb_to_lab(img_np)
        L = img_lab[:,:,0:1] / 100.0
        ab = (img_lab[:,:,1:] + 128) / 255.0
        import torch
        L = torch.from_numpy(L).permute(2,0,1).float()
        ab = torch.from_numpy(ab).permute(2,0,1).float()
        return L, ab

def rgb_to_lab(img):
    import cv2
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)