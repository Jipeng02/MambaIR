# ==== Setup ====
from basicsr.archs.mambairv2_arch import MambaIRv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import os
import lpips

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

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
ckpt_path = './full_finetuned_final.pth'
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint.get('params', checkpoint)
model.load_state_dict(state_dict, strict=False)
print(f"✓ 加载预训练权重: {ckpt_path}")

# ==== 解冻所有参数进行全量微调 ====
print("\n解冻所有参数进行全量微调...")
for p in model.parameters():
    p.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# ==== 数据增强：随机色彩扰动（教师扰动）====
class ColorJitter(object):
    """随机调整亮度、对比度、饱和度、色相"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, img):
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
        
        # 随机色相
        if self.hue > 0:
            import numpy as np
            from PIL import Image as PILImage
            import colorsys
            
            img_array = np.array(img).astype(np.float32) / 255.0
            h, s, v = [], [], []
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    r, g, b = img_array[i, j]
                    h_val, s_val, v_val = colorsys.rgb_to_hsv(r, g, b)
                    h_val = (h_val + random.uniform(-self.hue, self.hue)) % 1.0
                    h.append(h_val)
                    s.append(s_val)
                    v.append(v_val)
            
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
        
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.color_jitter = ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.4,
                hue=0.1
            )
            print("✓ 启用数据增强：随机亮度/对比度/饱和度/色相")

    def __len__(self): 
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        
        # 教师扰动
        if self.use_augmentation and torch.rand(1).item() > 0.3:
            img = self.color_jitter(img)
        
        color = self.transform(img) if self.transform else T.ToTensor()(img)
        gray = self.to_gray(img)
        gray_tensor = self.transform(gray) if self.transform else T.ToTensor()(gray)
        gray_stacked = gray_tensor.repeat(3, 1, 1)
        return gray_stacked, color

img_dir = '../data'
transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
dataset = ColorizationDataset(img_dir, transform=transform, use_augmentation=True)

# 内存优化：batch_size=1 + 梯度累积
batch_size = 1
accumulation_steps = 4  # 有效 batch_size = 1 * 4 = 4
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)

print(f"\n数据集大小: {len(dataset)} 张图片")
print(f"内存优化配置: batch_size={batch_size}, 梯度累积步数={accumulation_steps}")
print(f"有效 batch size = {batch_size * accumulation_steps}")

# ==== Optimizer (全量微调，使用较小学习率) ====
optimizer = torch.optim.AdamW([
    {'params': model.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
])

# 学习率调度器
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)

# ==== LPIPS 感知损失 ====
print("\n加载 LPIPS 感知损失模型...")
lpips_fn = lpips.LPIPS(net='vgg').to(device)
for p in lpips_fn.parameters():
    p.requires_grad = False
print("✓ LPIPS 模型加载完成")

# ==== Lab utilities & loss ====
def _srgb_to_linear(x):
    a = 0.055
    return torch.where(x <= 0.04045, x/12.92, torch.pow((x+a)/(1+a) + 1e-8, 2.4))

def _rgb_to_xyz(rgb):
    rgb = rgb.clamp(0, 1)
    x = _srgb_to_linear(rgb).permute(0,2,3,1)
    M = rgb.new_tensor([[0.4124564,0.3575761,0.1804375],
                        [0.2126729,0.7151522,0.0721750],
                        [0.0193339,0.1191920,0.9503041]])
    xyz = torch.matmul(x, M.T).permute(0,3,1,2).contiguous()
    return xyz

def _f_lab(t):
    d = 6/29
    return torch.where(t > d**3, torch.pow(t + 1e-8, 1/3), t/(3*d**2) + 4/29)

def rgb_to_lab(rgb):
    xyz = _rgb_to_xyz(rgb)
    Xn, Yn, YnZ = 0.95047, 1.0, 1.08883
    x = xyz[:,0]/Xn; y = xyz[:,1]/Yn; z = xyz[:,2]/YnZ
    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return torch.stack([L,a,b], dim=1)

def colorization_loss(pred_rgb, gt_rgb, w_ab=1.0, w_L=0.15):
    pred_lab = rgb_to_lab(pred_rgb)
    gt_lab = rgb_to_lab(gt_rgb)
    L1, a1, b1 = pred_lab[:,0:1], pred_lab[:,1:2], pred_lab[:,2:3]
    L2, a2, b2 = gt_lab[:,0:1], gt_lab[:,1:2], gt_lab[:,2:3]
    loss_ab = (a1-a2).abs().mean() + (b1-b2).abs().mean()
    loss_L = (L1-L2).abs().mean()
    return w_ab*loss_ab + w_L*loss_L

def combined_loss(pred_rgb, gt_rgb, lpips_fn, w_ab=1.0, w_L=0.15, w_lpips=0.5):
    """Lab L1 损失 + LPIPS 感知损失"""
    # Lab 损失
    loss_lab = colorization_loss(pred_rgb, gt_rgb, w_ab=w_ab, w_L=w_L)
    
    # LPIPS 感知损失
    pred_rgb_norm = pred_rgb * 2.0 - 1.0
    gt_rgb_norm = gt_rgb * 2.0 - 1.0
    loss_lpips = lpips_fn(pred_rgb_norm, gt_rgb_norm).mean()
    
    total_loss = loss_lab + w_lpips * loss_lpips
    
    return total_loss, loss_lab, loss_lpips

# ==== Train (5 epochs) ====
num_epochs = 10

print("\n" + "="*60)
print("开始全量微调训练（内存优化版本）")
print(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")
print(f"实际 Batch size: {loader.batch_size}, 梯度累积: {accumulation_steps}")
print(f"有效 Batch size: {loader.batch_size * accumulation_steps}")
print(f"总 epochs: {num_epochs}")
print("="*60 + "\n")

best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_loss_lab = 0.0
    running_loss_lpips = 0.0
    batch_count = 0
    
    # 梯度累积优化器重置
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (gray, color) in enumerate(loader):
        gray = gray.to(device, non_blocking=True)
        color = color.to(device, non_blocking=True)

        # 前向传播
        pred = model(gray)
        pred = pred.clamp(0, 1)
        
        if torch.isnan(pred).any() or torch.isnan(color).any():
            print(f"⚠️ Batch {batch_idx}: 检测到 NaN，跳过")
            continue
        
        # 组合损失：Lab + LPIPS
        loss, loss_lab, loss_lpips = combined_loss(
            pred, color, lpips_fn,
            w_ab=1.0, w_L=0.15, w_lpips=0.5
        )
        
        if torch.isnan(loss):
            print(f"⚠️ Batch {batch_idx}: Loss is NaN，跳过")
            continue

        # 梯度累积：除以累积步数以平均梯度
        loss = loss / accumulation_steps
        loss.backward()

        running_loss += loss.item() * accumulation_steps * gray.size(0)
        running_loss_lab += loss_lab.item() * gray.size(0)
        running_loss_lpips += loss_lpips.item() * gray.size(0)
        batch_count += 1
        
        # 每 accumulation_steps 步或最后一个batch时更新参数
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # 定期清理缓存
            if (batch_idx + 1) % (accumulation_steps * 10) == 0:
                torch.cuda.empty_cache()
        
        if (batch_idx + 1) % 50 == 0:
            avg_total = running_loss / (batch_count * gray.size(0))
            avg_lab = running_loss_lab / (batch_count * gray.size(0))
            avg_lpips = running_loss_lpips / (batch_count * gray.size(0))
            print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(loader)}: "
                  f"Total={avg_total:.6f} (Lab={avg_lab:.6f}, LPIPS={avg_lpips:.6f})")

    # Epoch 结束
    scheduler.step()
    torch.cuda.empty_cache()  # Epoch 结束后清理缓存
    
    if batch_count > 0:
        avg_loss = running_loss / (batch_count * batch_size)
        avg_loss_lab = running_loss_lab / (batch_count * batch_size)
        avg_loss_lpips = running_loss_lpips / (batch_count * batch_size)
        current_lr = optimizer.param_groups[0]['lr']
        
        print("\n" + "="*60)
        print(f"Epoch {epoch+1}/{num_epochs} 完成")
        print(f"  Total Loss: {avg_loss:.6f} (Lab: {avg_loss_lab:.6f}, LPIPS: {avg_loss_lpips:.6f})")
        print(f"  学习率: {current_lr:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "./best_full_finetuned_epoch_15.pth")
            print(f"  ✓ 保存最佳模型 (Loss: {avg_loss:.6f})")
        
        print("="*60 + "\n")
    else:
        print(f"Epoch {epoch+1}/{num_epochs} - 所有批次都出现 NaN！")
        break

# ==== Save final model ====
save_path = "./full_finetuned_final_epoch_15.pth"
torch.save(model.state_dict(), save_path)
print(f"\n✓ 训练完成！")
print(f"✓ 最终模型保存到: {save_path}")
print(f"✓ 最佳模型 (Loss={best_loss:.6f}) 保存到: ./best_full_finetuned_epoch_15.pth")
