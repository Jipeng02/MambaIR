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
