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

num_epochs = 2
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
# ===== Inference & Visualization =====
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

# 小工具：把 (L, ab) 还原成 RGB（numpy uint8, HxWx3）
def lab_to_rgb_np(L_t, ab_t):
    """
    L_t: torch.Tensor [1,H,W], 取值范围 [0,1]，需要先乘以 100
    ab_t: torch.Tensor [2,H,W], 取值范围 [0,1]，需要先*255-128
    """
    L = (L_t.squeeze(0).cpu().numpy() * 100.0).astype(np.float32)        # [H,W]
    ab = (ab_t.cpu().numpy() * 255.0 - 128.0).astype(np.float32)         # [2,H,W]
    a = ab[0, ...]
    b = ab[1, ...]
    lab = np.stack([L, a, b], axis=-1)                                   # [H,W,3] in LAB
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)                           # float32, 0-255 ish
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb

# 1) 载入训练好的权重（也可以省略这步，因为你上面刚训练完）
ckpt_path = "color_model_lab_trained.pth"
state = torch.load(ckpt_path, map_location="cpu")
_ = model.load_state_dict(state, strict=False)
model.eval()

# 2) 随机抽取 3 个样本索引
indices = random.sample(range(len(dataset)), 3)

# 3) 推理与可视化
fig = plt.figure(figsize=(9, 9))  # 3 行 × 3 列
plt.subplots_adjust(hspace=0.15, wspace=0.05)

for i, idx in enumerate(indices):
    # 从 dataset 取出单样本（L, ab），以及原图 RGB（便于对比）
    L, ab_gt = dataset[idx]                       # L:[1,64,64], ab:[2,64,64], 归一化过
    # 为了展示“原图 RGB”，从磁盘读原图再 resize 到 64x64
    img_path = dataset.img_paths[idx]
    orig_rgb = cv2.cvtColor(
        cv2.resize(cv2.imread(img_path), (64, 64)),
        cv2.COLOR_BGR2RGB
    )

    with torch.no_grad():
        L_in = L.unsqueeze(0).to(device)          # [1,1,64,64]
        ab_pred = model(L_in)                     # [1,2,64,64]，与训练时一致的归一化范围
        ab_pred = ab_pred.squeeze(0).clamp(0, 1)  # [2,64,64]，稳妥起见 clamp

    # 还原预测的 RGB
    pred_rgb = lab_to_rgb_np(L, ab_pred)

    # 灰度图只显示 L（0-1 之间，cmap='gray'）
    gray_img = (L.squeeze(0).cpu().numpy())      # [64,64], 0~1

    # 画三列：灰度 / 预测RGB / 原图RGB
    # 灰度
    ax = plt.subplot(3, 3, 3*i + 1)
    ax.imshow(gray_img, cmap='gray', vmin=0.0, vmax=1.0)
    ax.set_title(f"Sample {idx} - L")
    ax.axis('off')

    # 预测
    ax = plt.subplot(3, 3, 3*i + 2)
    ax.imshow(pred_rgb)
    ax.set_title("Predicted RGB")
    ax.axis('off')

    # 原图
    ax = plt.subplot(3, 3, 3*i + 3)
    ax.imshow(orig_rgb)
    ax.set_title("Original RGB")
    ax.axis('off')

plt.show()
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ==== 保存模型权重 ====
save_path = "/content/drive/MyDrive/color_model_lab_trained_final.pth"
torch.save(model.state_dict(), save_path)
print(f"✅ 模型权重已保存到: {save_path}")

# ==== 推理并保存 3 张图片 ====
os.makedirs("/content/drive/MyDrive", exist_ok=True)

model.eval()
indices = [0, 1, 2]  # 从训练集取前三张
for i, idx in enumerate(indices):
    L, ab_gt = dataset[idx]
    img_path = dataset.img_paths[idx]
    with torch.no_grad():
        L_in = L.unsqueeze(0).to(device)
        ab_pred = model(L_in).squeeze(0).clamp(0,1)

    # 转为RGB
    def lab_to_rgb_np(L_t, ab_t):
        import cv2
        L = (L_t.squeeze(0).cpu().numpy() * 100.0).astype(np.float32)
        ab = (ab_t.cpu().numpy() * 255.0 - 128.0).astype(np.float32)
        lab = np.stack([L, ab[0], ab[1]], axis=-1)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    pred_rgb = lab_to_rgb_np(L, ab_pred)
    gray_img = (L.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    orig_rgb = cv2.cvtColor(
        cv2.resize(cv2.imread(img_path), (64, 64)),
        cv2.COLOR_BGR2RGB
    )

    # 保存三张图片
    cv2.imwrite(f"/content/drive/MyDrive/infer_results/sample{i+1}_gray.png", gray_img)
    cv2.imwrite(f"/content/drive/MyDrive/infer_results/sample{i+1}_pred.png", cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"/content/drive/MyDrive/infer_results/sample{i+1}_orig.png", cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))
    print(f"✅ 已保存 sample{i+1} 三张图片到 /content/infer_results")

print("🎉 所有结果已保存完成！")
