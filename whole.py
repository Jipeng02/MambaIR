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
checkpoint = torch.load('/content/MambaIR/mambairv2_ColorDN_15.pth', map_location='cpu')

state_dict = checkpoint.get('params', checkpoint)  # 'params' if it's a dict, else the plain dict

# Remove the incompatible keys
to_ignore = []
for k in state_dict.keys():
    # Ignore the first and last conv layers
    if k.startswith('conv_first'):
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
    
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ColorizationDataset(Dataset):
    def __init__(self, img_dir, transform=None, max_samples=None):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]
        self.transform = transform
        self.to_gray = T.Grayscale(num_output_channels=1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        color = self.transform(img) if self.transform else T.ToTensor()(img)
        gray = self.to_gray(img)
        gray = self.transform(gray) if self.transform else T.ToTensor()(gray)
        return gray, color

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

img_dir = '/content/drive/MyDrive/PatternAnalysis-2025/data/train2017'
dataset = ColorizationDataset(img_dir, transform=T.Compose([T.Resize((64,64)), T.ToTensor()]))
loader = DataLoader(dataset, batch_size=4, shuffle=True)
import random
from torchvision.utils import save_image
import os

# Ensure model is in evaluation mode
model.eval()

# Randomly select an image from the dataset
idx = random.randint(0, len(dataset) - 1)
gray, color = dataset[idx]  # gray: [1, H, W], color: [3, H, W]

# Add batch dimension and move to device
gray_input = gray.unsqueeze(0).to(device)  # [1, 1, 64, 64]

# Predict with model (no gradient)
with torch.no_grad():
    pred = model(gray_input)  # Expected output: [1, 3, 64, 64]

# Clamp output to [0, 1] for valid image saving
pred = pred.clamp(0, 1).squeeze(0).cpu()   # [3, 64, 64]
gray = gray.cpu()
color = color.cpu()

# Save images to /content
os.makedirs('/content', exist_ok=True)

save_image(gray, '/content/input_grayscale.png')       # shape: [1, 64, 64]
save_image(pred, '/content/predicted_rgb.png')         # shape: [3, 64, 64]
save_image(color, '/content/ground_truth_rgb.png')     # shape: [3, 64, 64]

print("Images saved to /content:")
print("- input_grayscale.png")
print("- predicted_rgb.png")
print("- ground_truth_rgb.png")

# num_epochs = 2
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for step, (gray, color) in enumerate(loader):
#         gray, color = gray.to(device), color.to(device)
#         optimizer.zero_grad()
#         pred = model(gray)
#         loss = criterion(pred, color)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * gray.size(0)
#         print(f"Epoch {epoch+1} Step {step+1}, Loss: {loss.item():.6f}")  # 打印每一步的loss
#     print(f"Epoch {epoch+1}, Avg Loss: {running_loss/len(loader.dataset):.6f}")


# torch.save(model.state_dict(), "color_model_lab_trained.pth")
