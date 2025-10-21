# ==== Setup ====
from basicsr.archs.mambairv2_arch import MambaIRv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import os
from torchvision.utils import save_image
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==== 显存优化1: 清理缓存 ====
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# ==== Model ====
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
ckpt_path = './new_color_model_last_no_lora.pth'
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint.get('params', checkpoint)
model.load_state_dict(state_dict, strict=False)
print(f"✓ 加载预训练权重: {ckpt_path}")

# 释放 checkpoint 显存
del checkpoint, state_dict
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==== 关键改进1: 重置末端层 ====
print("\n重置末端层权重...")
with torch.no_grad():
    if hasattr(model, 'conv_after_body'):
        model.conv_after_body.reset_parameters()
        print("✓ 已重置 conv_after_body")
    if hasattr(model, 'conv_last'):
        model.conv_last.reset_parameters()
        print("✓ 已重置 conv_last")

# ==== 显存优化2: 只训练末端3层，不解冻更多层 ====
for p in model.parameters():
    p.requires_grad = False

train_modules = []
if hasattr(model, 'conv_first'):
    train_modules.append(model.conv_first)
    print("✓ 训练 conv_first")
if hasattr(model, 'conv_after_body'):
    train_modules.append(model.conv_after_body)
    print("✓ 训练 conv_after_body")
if hasattr(model, 'conv_last'):
    train_modules.append(model.conv_last)
    print("✓ 训练 conv_last")

if not train_modules:
    print("⚠️ 警告：没有找到任何可训练模块！")
    for p in model.parameters():
        p.requires_grad = True
else:
    for m in train_modules:
        for p in m.parameters():
            p.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

if trainable_params == 0:
    raise RuntimeError("❌ 没有可训练参数！")

# ==== Dataset ====
class ColorizationDataset(Dataset):
    def __init__(self, img_dir, transform=None, max_samples=None):
        self.img_paths = [os.path.join(img_dir, f)
                          for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]
        self.transform = transform
        self.to_gray = T.Grayscale(num_output_channels=1)

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        color = self.transform(img) if self.transform else T.ToTensor()(img)
        gray = self.to_gray(img)
        gray_tensor = self.transform(gray) if self.transform else T.ToTensor()(gray)
        gray_stacked = gray_tensor.repeat(3, 1, 1)
        return gray_stacked, color

img_dir = '../data'
transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
dataset = ColorizationDataset(img_dir, transform=transform)

# ==== 显存优化3: batch_size=1 + 梯度累积8次 ====
BATCH_SIZE = 1  # 最小 batch size，极度显存友好
ACCUMULATION_STEPS = 8  # 累积8次，等效 batch_size=8
# 显存优化4: 关闭 pin_memory
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                   num_workers=2, pin_memory=False)  

print(f"\n数据集大小: {len(dataset)} 张图片")
print(f"实际 batch size: {BATCH_SIZE}, 累积步数: {ACCUMULATION_STEPS}, 等效 batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")

# ==== 提高学习率 + 学习率调度器 ====
def param_groups_for_decay(modules, lr=2e-3):  # 提高到 2e-3 补偿慢收敛
    decay, no_decay = [], []
    for m in modules:
        for n, p in m.named_parameters():
            if not p.requires_grad: continue
            if n.endswith('bias') or 'bn' in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
    return [
        {'params': decay, 'weight_decay': 1e-4, 'lr': lr},
        {'params': no_decay, 'weight_decay': 0.0, 'lr': lr}
    ]

optimizer = torch.optim.AdamW(param_groups_for_decay(train_modules, lr=2e-3))

num_epochs = 30  # 增加到30轮补偿小batch size
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

# ==== Lab loss ====
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

# ==== 训练循环 ====
print("\n" + "="*60)
print("🚀 低显存优化训练")
print(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")
print(f"Batch size: {BATCH_SIZE} (累积 {ACCUMULATION_STEPS} 步)")
print(f"总 epochs: {num_epochs}")
print("="*60 + "\n")

best_loss = float('inf')
os.makedirs('./training_samples', exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    batch_count = 0
    
    optimizer.zero_grad()

    for batch_idx, (gray, color) in enumerate(loader):
        gray = gray.to(device, non_blocking=True)
        color = color.to(device, non_blocking=True)

        # 前向传播
        pred = model(gray).clamp(0, 1)
        
        # 检查 NaN
        if torch.isnan(pred).any() or torch.isnan(color).any():
            print(f"⚠️ Batch {batch_idx}: 检测到 NaN，跳过")
            continue
        
        loss = colorization_loss(pred, color, w_ab=1.0, w_L=0.15)
        
        if torch.isnan(loss):
            print(f"⚠️ Batch {batch_idx}: Loss is NaN，跳过")
            continue

        # ==== 梯度累积 ====
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        
        # 每累积 ACCUMULATION_STEPS 次后更新参数
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # ==== 显存优化5: 定期清理显存 ====
            if (batch_idx + 1) % (ACCUMULATION_STEPS * 10) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        running_loss += loss.item() * ACCUMULATION_STEPS * gray.size(0)
        batch_count += 1
        
        # 每 200 个 batch 打印（减少打印频率节省时间）
        if (batch_idx + 1) % 200 == 0:
            avg_so_far = running_loss / (batch_count * gray.size(0))
            # 显示显存使用情况
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(loader)}: "
                      f"Loss={avg_so_far:.6f}, 显存={mem_used:.2f}GB/{mem_reserved:.2f}GB")
            else:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(loader)}: Loss={avg_so_far:.6f}")

    # Epoch 结束
    if batch_count > 0:
        avg_loss = running_loss / (batch_count * BATCH_SIZE)
        current_lr = optimizer.param_groups[0]['lr']
        
        print("\n" + "="*60)
        print(f"Epoch {epoch+1}/{num_epochs} 完成")
        print(f"  平均 Loss: {avg_loss:.6f}")
        print(f"  当前学习率: {current_lr:.2e}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'state_dict': model.state_dict(), 
                       'epoch': epoch, 
                       'loss': avg_loss}, 
                      "./best_color_model_low_mem.pth")
            print(f"  ✓ 保存最佳模型 (Loss: {avg_loss:.6f})")
        
        # 每 10 个 epoch 保存样本图片（减少频率）
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for i in range(min(3, len(dataset))):
                    sample_gray, sample_color = dataset[i]
                    sample_pred = model(sample_gray.unsqueeze(0).to(device)).clamp(0, 1)
                    
                    comparison = torch.cat([
                        sample_gray[0:1].repeat(3, 1, 1),
                        sample_pred.squeeze(0).cpu(),
                        sample_color
                    ], dim=2)
                    
                    save_image(comparison, f'./training_samples/epoch_{epoch+1}_sample_{i}.png')
            print(f"  ✓ 保存样本图片")
            model.train()
        
        # 每 5 个 epoch 保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"./checkpoint_epoch_{epoch+1}.pth"
            torch.save({'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'epoch': epoch,
                       'loss': avg_loss}, 
                      checkpoint_path)
            print(f"  ✓ 保存检查点: {checkpoint_path}")
        
        print("="*60 + "\n")
        
        # 更新学习率
        scheduler.step()
        
        # ==== 显存优化6: Epoch 结束清理显存 ====
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"Epoch {epoch+1}/{num_epochs} - 所有批次都出现 NaN！")
        break

# ==== 保存最终模型 ====
save_path = "./final_color_model_low_mem.pth"
torch.save(model.state_dict(), save_path)
print(f"\n✓ 训练完成！最终模型保存到: {save_path}")
print(f"✓ 最佳模型 Loss: {best_loss:.6f}")