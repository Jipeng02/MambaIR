import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ColorizationPatchDataset(Dataset):
    def __init__(self, img_dir, patch_size=64, transform=None, max_samples=None):
        self.img_paths = [os.path.join(img_dir, f)
                          for f in os.listdir(img_dir)
                          if f.endswith('.png') or f.endswith('.jpg')]
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]
        self.patch_size = patch_size
        self.transform = transform
        self.to_gray = T.Grayscale(num_output_channels=1)
        self.cache = []  # Cache all patches

        self._generate_patches()

    def _generate_patches(self):
        """ Pre-slice all images into 64x64 patches and store (img_idx, patch_coords) """
        for img_idx, img_path in enumerate(self.img_paths):
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            for top in range(0, h - self.patch_size + 1, self.patch_size):
                for left in range(0, w - self.patch_size + 1, self.patch_size):
                    self.cache.append((img_idx, left, top))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        img_idx, left, top = self.cache[idx]
        img = Image.open(self.img_paths[img_idx]).convert('RGB')

        # Crop 64x64 patch
        patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))

        # Apply transforms
        color = self.transform(patch) if self.transform else T.ToTensor()(patch)
        gray = self.to_gray(patch)
        gray = self.transform(gray) if self.transform else T.ToTensor()(gray)

        return gray, color
