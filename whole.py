from basicsr.archs.mambairv2_arch import MambaIRv2
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaIRv2(
    img_size=64,
    patch_size=1,
    in_chans=1,        # 会被强制为 1
    embed_dim=174,
    d_state=16,
    depths=(6, 6, 6, 6, 6, 6),
    num_heads= [6, 6, 6, 6, 6, 6],
    window_size=16,
    inner_rank=64,
    num_tokens=128,
    convffn_kernel_size=5,
    mlp_ratio=2.0,
    upsampler='',      # 会被强制为 ''
    upscale=1,         # 会被强制为 1
    resi_connection='1conv'

).to(device)

# Load the full checkpoint (could be .pth or .pt file)
checkpoint = torch.load('/content/mambairv2_ColorDN_15.pth', map_location='cpu')

state_dict = checkpoint.get('params', checkpoint)  # 'params' if it's a dict, else the plain dict

# Remove the incompatible keys
to_ignore = []
for k in state_dict.keys():
    # Ignore the first and last conv layers
    if k.startswith('conv_first') or k.startswith('conv_last'):
        to_ignore.append(k)
    # If using conv_after_body or upsampling, you may want to check those too

for k in to_ignore:
    print(f"Skip loading {k}")
    state_dict.pop(k)


# Now load the rest (strict=False allows missing/unmatched keys)
model.load_state_dict(state_dict, strict=False)
# kaiming/he initialization for the first conv layer
nn.init.kaiming_normal_(model.conv_first.weight, mode='fan_out', nonlinearity='relu')
if model.conv_first.bias is not None:
    nn.init.zeros_(model.conv_first.bias)
#zero initialization for the last conv layer
nn.init.zeros_(model.conv_last.weight)
if model.conv_last.bias is not None:
    nn.init.zeros_(model.conv_last.bias)
    
input_tensor = torch.randn(2, 1, 64, 64).to(device)  # 输入也放到 device
# output = model(input_tensor)
# print(output.shape)  # Should print torch.Size([2, 2, 64, 64])
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class LabColorizationDataset(Dataset):
    def __init__(self, img_dir, img_size=64, max_samples=None):
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

img_dir = '/content/drive/MyDrive/PatternAnalysis-2025/data/train2017'
dataset = LabColorizationDataset(img_dir, img_size=64,max_samples=2000)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for step, (L, ab) in enumerate(loader, start=1):
        L, ab = L.to(device), ab.to(device)
        optimizer.zero_grad()
        ab_pred = model(L)
        loss = criterion(ab_pred, ab)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * L.size(0)
        print(f"Epoch {epoch+1} Step {step}/{len(loader)} | loss={loss.item():.6f}")
    print(f"[Epoch {epoch+1}] mean_loss={epoch_loss/len(loader.dataset):.6f}")


torch.save(model.state_dict(), "color_model_lab_trained.pth")