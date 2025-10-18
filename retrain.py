from basicsr.archs.mambairv2_arch import MambaIRv2
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaIRv2(
    img_size=128,
    patch_size=1,
    in_chans=3,
    embed_dim=174,
    d_state=16,
    depths=(6, 6, 6, 6, 6, 6),
    num_heads= [6, 6, 6, 6, 6, 6],
    window_size=16,
    inner_rank=64,
    num_tokens=128,
    convffn_kernel_size=5,
    mlp_ratio=2.0,
    upsampler='',
    upscale=1,
    resi_connection='1conv'

).to(device)

# Load the full checkpoint (could be .pth or .pt file)
checkpoint = torch.load('./MambaIR/color_model_lab_trained_100epoch_val.pth', map_location='cpu')

state_dict = checkpoint.get('params', checkpoint)  # 'params' if it's a dict, else the plain dict

# Remove the incompatible keys
# to_ignore = []
# for k in state_dict.keys():
#     # Ignore the first and last conv layers
#     if k.startswith('conv_first'):
#         to_ignore.append(k)
#     # If using conv_after_body or upsampling, you may want to check those too

# for k in to_ignore:
#     print(f"Skip loading {k}")
#     state_dict.pop(k)


# Now load the rest (strict=False allows missing/unmatched keys)
model.load_state_dict(state_dict, strict=False)

# Freeze all parameters except the first convolution layer.
# 微调时仅调整输入映射，其余层保持预训练权重。
for name, param in model.named_parameters():
    param.requires_grad = name.startswith('conv_first')


import os
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
        gray_tensor = self.transform(gray) if self.transform else T.ToTensor()(gray)
        # 将单通道灰度复制到三个通道，以匹配预训练模型的输入规格
        gray_stacked = gray_tensor.repeat(3, 1, 1)
        return gray_stacked, color

from torch.utils.data import DataLoader

criterion = nn.L1Loss()
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

img_dir = './data/val2017'
dataset = ColorizationDataset(img_dir, transform=T.Compose([T.Resize((128,128)), T.ToTensor(),]))
loader = DataLoader(dataset, batch_size=4, shuffle=True)

num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for step, (gray, color) in enumerate(loader):
        gray, color = gray.to(device), color.to(device)
        optimizer.zero_grad()
        pred = model(gray)
        loss = criterion(pred, color)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * gray.size(0)
    
    avg_loss = running_loss / len(loader.dataset)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}")

    # 每20个epoch保存一次模型
    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f"./color_model_lab_epoch{epoch+1}.pth")

# 最后一次保存（防止刚好不是20的倍数时丢失最终模型）
torch.save(model.state_dict(), "./color_model_lab_trained_final.pth")