from basicsr.archs.mambairv2_arch import MambaIRv2
import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据集类
class ColorizationDataset(Dataset):
    def __init__(self, img_dir, transform=None, max_samples=None):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                         if f.endswith('.png') or f.endswith('.jpg')]
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


def evaluate_model(model_path='./mamba.pth', 
                   data_dir='../data', 
                   max_samples=2000, 
                   batch_size=8,
                   img_size=128):
    """
    评估模型在数据集上的表现
    
    Args:
        model_path: 模型权重文件路径
        data_dir: 数据集目录
        max_samples: 最大评估样本数
        batch_size: 批次大小
        img_size: 图片尺寸
    """
    
    # 加载模型
    print("正在加载模型...")
    model = MambaIRv2(
        img_size=img_size,
        patch_size=1,
        in_chans=3,
        embed_dim=174,
        d_state=16,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=16,
        inner_rank=64,
        num_tokens=128,
        convffn_kernel_size=5,
        mlp_ratio=2.0,
        upsampler='',
        upscale=1,
        resi_connection='1conv'
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('params', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("模型加载完成！")
    
    # 准备数据集
    print(f"\n正在加载数据集: {data_dir}")
    dataset = ColorizationDataset(
        data_dir, 
        transform=T.Compose([T.Resize((img_size, img_size)), T.ToTensor()]),
        max_samples=max_samples
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"数据集大小: {len(dataset)} 张图片")
    
    # 统计变量
    total_samples = 0
    sum_pred_r, sum_pred_g, sum_pred_b = 0.0, 0.0, 0.0
    sum_gt_r, sum_gt_g, sum_gt_b = 0.0, 0.0, 0.0
    sum_delta_r, sum_delta_g, sum_delta_b = 0.0, 0.0, 0.0
    
    # 额外统计：标准差和像素级MSE
    sum_mse_r, sum_mse_g, sum_mse_b = 0.0, 0.0, 0.0
    
    print(f"\n开始评估...")
    print("="*60)
    
    # 遍历所有数据
    with torch.no_grad():
        for batch_idx, (gray, color) in enumerate(loader):
            # gray: [B, 3, H, W], color: [B, 3, H, W]
            gray_input = gray.to(device)
            color = color.to(device)
            
            # 模型预测
            pred = model(gray_input).clamp(0, 1)  # [B, 3, H, W]
            
            # 计算每张图片的通道均值
            pred_mean = pred.mean(dim=(2, 3))  # [B, 3]
            color_mean = color.mean(dim=(2, 3))  # [B, 3]
            delta = pred_mean - color_mean  # [B, 3]
            
            # 累加统计
            sum_pred_r += pred_mean[:, 0].sum().item()
            sum_pred_g += pred_mean[:, 1].sum().item()
            sum_pred_b += pred_mean[:, 2].sum().item()
            
            sum_gt_r += color_mean[:, 0].sum().item()
            sum_gt_g += color_mean[:, 1].sum().item()
            sum_gt_b += color_mean[:, 2].sum().item()
            
            sum_delta_r += delta[:, 0].sum().item()
            sum_delta_g += delta[:, 1].sum().item()
            sum_delta_b += delta[:, 2].sum().item()
            
            # 计算像素级MSE
            mse = ((pred - color) ** 2).mean(dim=(2, 3))  # [B, 3]
            sum_mse_r += mse[:, 0].sum().item()
            sum_mse_g += mse[:, 1].sum().item()
            sum_mse_b += mse[:, 2].sum().item()
            
            total_samples += gray.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"已处理 {total_samples}/{len(dataset)} 张图片...")
    
    # 计算平均值
    avg_pred_r = sum_pred_r / total_samples
    avg_pred_g = sum_pred_g / total_samples
    avg_pred_b = sum_pred_b / total_samples
    
    avg_gt_r = sum_gt_r / total_samples
    avg_gt_g = sum_gt_g / total_samples
    avg_gt_b = sum_gt_b / total_samples
    
    avg_delta_r = sum_delta_r / total_samples
    avg_delta_g = sum_delta_g / total_samples
    avg_delta_b = sum_delta_b / total_samples
    
    avg_mse_r = sum_mse_r / total_samples
    avg_mse_g = sum_mse_g / total_samples
    avg_mse_b = sum_mse_b / total_samples
    
    # 打印结果
    print("\n" + "="*60)
    print(f"评估完成！共 {total_samples} 张图片")
    print("="*60)
    print("\n【通道均值统计】")
    print(f"预测均值 (Pred): R={avg_pred_r:.4f}, G={avg_pred_g:.4f}, B={avg_pred_b:.4f}")
    print(f"真实均值 (GT):   R={avg_gt_r:.4f}, G={avg_gt_g:.4f}, B={avg_gt_b:.4f}")
    print(f"差值 (Δ):        R={avg_delta_r:.4f}, G={avg_delta_g:.4f}, B={avg_delta_b:.4f}")
    
    print("\n【像素级MSE】")
    print(f"MSE: R={avg_mse_r:.6f}, G={avg_mse_g:.6f}, B={avg_mse_b:.6f}")
    print(f"RMSE: R={np.sqrt(avg_mse_r):.6f}, G={np.sqrt(avg_mse_g):.6f}, B={np.sqrt(avg_mse_b):.6f}")
    
    print("\n【分析】")
    if abs(avg_delta_r) > abs(avg_delta_g) and abs(avg_delta_r) > abs(avg_delta_b):
        print("⚠️  红色通道偏差最大！")
    elif abs(avg_delta_g) > abs(avg_delta_r) and abs(avg_delta_g) > abs(avg_delta_b):
        print("⚠️  绿色通道偏差最大！")
    elif abs(avg_delta_b) > abs(avg_delta_r) and abs(avg_delta_b) > abs(avg_delta_g):
        print("⚠️  蓝色通道偏差最大！")
    else:
        print("✅ 各通道偏差相对均衡")
    
    print("="*60)
    
    # 返回统计结果
    return {
        'total_samples': total_samples,
        'pred_mean': (avg_pred_r, avg_pred_g, avg_pred_b),
        'gt_mean': (avg_gt_r, avg_gt_g, avg_gt_b),
        'delta': (avg_delta_r, avg_delta_g, avg_delta_b),
        'mse': (avg_mse_r, avg_mse_g, avg_mse_b),
        'rmse': (np.sqrt(avg_mse_r), np.sqrt(avg_mse_g), np.sqrt(avg_mse_b))
    }


if __name__ == '__main__':
    # 运行评估
    results = evaluate_model(
        model_path='./mamba.pth',
        data_dir='../data',
        max_samples=2000,
        batch_size=8,
        img_size=128
    )
    
    print("\n评估结束！")
