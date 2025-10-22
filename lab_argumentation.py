# ==== Setup ====
from basicsr.archs.mambairv2_arch import MambaIRv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== Model (img_range=1.0 to match ToTensor [0,1]) ====
model = MambaIRv2(
    img_size=128, patch_size=1, in_chans=3,
    embed_dim=174, d_state=16,
    depths=(6,6,6,6,6,6),
    num_heads=[6,6,6,6,6,6],
    window_size=16, inner_rank=64, num_tokens=128,
    convffn_kernel_size=5, mlp_ratio=2.0,
    upsampler='', upscale=1,
    resi_connection='1conv',
    img_range=1.0
).to(device)

# ==== Load pretrained ====
ckpt_path = './new_color_model_last_no_lora_1.pth'  # 你指定的初始权重
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint.get('params', checkpoint)
model.load_state_dict(state_dict, strict=False)

# ==== Freeze backbone; only train conv_first / conv_after_body / conv_last ====
for p in model.parameters():
    p.requires_grad = False

train_modules = []
if hasattr(model, 'conv_first'):       
    train_modules.append(model.conv_first)
    print(f"✓ 找到 conv_first")
if hasattr(model, 'conv_after_body'):  
    train_modules.append(model.conv_after_body)
    print(f"✓ 找到 conv_after_body")
if hasattr(model, 'conv_last'):        
    train_modules.append(model.conv_last)
    print(f"✓ 找到 conv_last")

if not train_modules:
    print("⚠️ 警告：没有找到任何可训练模块，解冻所有参数！")
    for p in model.parameters():
        p.requires_grad = True
else:
    for m in train_modules:
        for p in m.parameters():
            p.requires_grad = True

# 验证是否有可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

if trainable_params == 0:
    raise RuntimeError("❌ 没有可训练参数！请检查模型结构。")

# （可选）若你想"洗掉"末端旧先验，可解开重置末端两层：
# with torch.no_grad():
#     if hasattr(model, 'conv_after_body'): model.conv_after_body.reset_parameters()
#     if hasattr(model, 'conv_last'):       model.conv_last.reset_parameters()

# ==== 数据增强：随机色彩扰动（教师扰动）====
class ColorJitter(object):
    """随机调整亮度、对比度、饱和度、色相"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image with random color jitter
        """
        import random
        from PIL import ImageEnhance
        
        # 随机亮度
        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
        
        # 随机对比度
        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
        
        # 随机饱和度
        if self.saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation_factor)
        
        # 随机色相（通过 HSV 转换）
        if self.hue > 0:
            import numpy as np
            from PIL import Image as PILImage
            
            img_array = np.array(img).astype(np.float32) / 255.0
            # RGB -> HSV
            import colorsys
            h, s, v = [], [], []
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    r, g, b = img_array[i, j]
                    h_val, s_val, v_val = colorsys.rgb_to_hsv(r, g, b)
                    # 随机调整色相
                    h_val = (h_val + random.uniform(-self.hue, self.hue)) % 1.0
                    h.append(h_val)
                    s.append(s_val)
                    v.append(v_val)
            
            # HSV -> RGB
            rgb_array = np.zeros_like(img_array)
            idx = 0
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    r, g, b = colorsys.hsv_to_rgb(h[idx], s[idx], v[idx])
                    rgb_array[i, j] = [r, g, b]
                    idx += 1
            
            img = PILImage.fromarray((rgb_array * 255).astype(np.uint8))
        
        return img

# ==== Dataset with Augmentation ====
class ColorizationDataset(Dataset):
    def __init__(self, img_dir, transform=None, max_samples=None, use_augmentation=True):
        self.img_paths = [os.path.join(img_dir, f)
                          for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]
        self.transform = transform
        self.to_gray = T.Grayscale(num_output_channels=1)
        
        # 教师扰动：随机色彩增强
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.color_jitter = ColorJitter(
                brightness=0.3,  # ±30% 亮度
                contrast=0.3,    # ±30% 对比度
                saturation=0.4,  # ±40% 饱和度
                hue=0.1          # ±10% 色相
            )
            print("✓ 启用数据增强：随机亮度/对比度/饱和度/色相")

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        
        # 教师扰动：对原始彩色图进行随机色彩增强
        if self.use_augmentation and torch.rand(1).item() > 0.3:  # 70% 概率应用增强
            img = self.color_jitter(img)
        
        color = self.transform(img) if self.transform else T.ToTensor()(img)
        gray  = self.to_gray(img)
        gray_tensor = self.transform(gray) if self.transform else T.ToTensor()(gray)
        gray_stacked = gray_tensor.repeat(3, 1, 1)  # 灰度三通道
        return gray_stacked, color

img_dir   = '../data'
transform = T.Compose([T.Resize((128,128)), T.ToTensor()])
# 启用数据增强（教师扰动）
dataset   = ColorizationDataset(img_dir, transform=transform, use_augmentation=True)
loader    = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

# ==== Optimizer (only the 3 layers; slightly higher LR) ====
def param_groups_for_decay(modules):
    decay, no_decay = [], []
    for m in modules:
        for n,p in m.named_parameters():
            if not p.requires_grad: continue
            if n.endswith('bias') or 'bn' in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
    return [{'params': decay, 'weight_decay': 1e-4, 'lr': 3e-4},
            {'params': no_decay, 'weight_decay': 0.0, 'lr': 3e-4}]

optimizer = torch.optim.AdamW(param_groups_for_decay(train_modules))

# ==== Lab utilities & ab-dominant loss ====
def _srgb_to_linear(x):
    """移除 @torch.no_grad() 以保持梯度，添加数值稳定性"""
    a = 0.055
    # 添加 epsilon 避免数值问题
    return torch.where(x <= 0.04045, x/12.92, torch.pow((x+a)/(1+a) + 1e-8, 2.4))

def _rgb_to_xyz(rgb):  # [B,3,H,W] -> [B,3,H,W]
    # 强制限制范围
    rgb = rgb.clamp(0, 1)
    x = _srgb_to_linear(rgb).permute(0,2,3,1)  # [B,H,W,3]
    M = rgb.new_tensor([[0.4124564,0.3575761,0.1804375],
                        [0.2126729,0.7151522,0.0721750],
                        [0.0193339,0.1191920,0.9503041]])
    xyz = torch.matmul(x, M.T).permute(0,3,1,2).contiguous()
    return xyz

def _f_lab(t):
    d = 6/29
    # 添加 epsilon 避免数值问题
    return torch.where(t > d**3, torch.pow(t + 1e-8, 1/3), t/(3*d**2) + 4/29)

def rgb_to_lab(rgb):
    xyz = _rgb_to_xyz(rgb)
    Xn,Yn,YnZ = 0.95047,1.0,1.08883
    x = xyz[:,0]/Xn; y = xyz[:,1]/Yn; z = xyz[:,2]/YnZ
    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return torch.stack([L,a,b], dim=1)

def colorization_loss(pred_rgb, gt_rgb, w_ab=1.0, w_L=0.15):
    pred_lab = rgb_to_lab(pred_rgb)
    gt_lab   = rgb_to_lab(gt_rgb)
    L1,a1,b1 = pred_lab[:,0:1], pred_lab[:,1:2], pred_lab[:,2:3]
    L2,a2,b2 = gt_lab[:,0:1],   gt_lab[:,1:2],   gt_lab[:,2:3]
    loss_ab = (a1-a2).abs().mean() + (b1-b2).abs().mean()
    loss_L  = (L1-L2).abs().mean()
    return w_ab*loss_ab + w_L*loss_L

# ==== Train (5 epochs) ====
num_epochs = 5
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    batch_count = 0

    for batch_idx, (gray, color) in enumerate(loader):
        gray  = gray.to(device, non_blocking=True)
        color = color.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 不使用 AMP，避免数值不稳定
        pred = model(gray)
        
        # 强制限制输出范围，避免 NaN
        pred = pred.clamp(0, 1)
        
        # 检查输入是否有 NaN
        if torch.isnan(pred).any() or torch.isnan(color).any():
            print(f"⚠️  Batch {batch_idx}: 检测到 NaN，跳过此批次")
            continue
        
        loss = colorization_loss(pred, color, w_ab=1.0, w_L=0.15)
        
        # 检查 loss 是否为 NaN
        if torch.isnan(loss):
            print(f"⚠️  Batch {batch_idx}: Loss is NaN, 跳过此批次")
            print(f"  pred range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
            print(f"  color range: [{color.min().item():.3f}, {color.max().item():.3f}]")
            continue

        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item() * gray.size(0)
        batch_count += 1
        
        # 每 50 个 batch 打印一次
        if (batch_idx + 1) % 50 == 0:
            avg_so_far = running_loss / (batch_count * gray.size(0))
            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(loader)}: Loss={avg_so_far:.6f}")

    if batch_count > 0:
        avg_loss = running_loss / (batch_count * loader.batch_size)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Lab-ab Loss: {avg_loss:.6f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs} - 所有批次都出现 NaN，训练失败！")
        break

# ==== Save last only ====
save_path = "./new_color_model_last_no_lora_1_argument_5.pth"
torch.save(model.state_dict(), save_path)
print(f"Saved final weights to: {save_path}")
