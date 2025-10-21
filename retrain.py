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
checkpoint = torch.load('./mamba.pth', map_location='cpu')

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
# import torch.nn as nn

# # 筛选出可用于 LoRA 的模块（一般是 Linear 或 Conv2d）
# target_module_types = (nn.Linear, nn.Conv2d)  # 可扩展支持更多模块

# # 存储目标模块名
# target_modules = []

# for name, module in model.named_modules():
#     if isinstance(module, target_module_types):
#         # 只保留最后一级模块名（如 q_proj，而不是 full.path.q_proj）
#         last_name = name.split('.')[-1]
#         if last_name not in target_modules:
#             target_modules.append(last_name)

# print("✅ 可用于 LoRA 的模块名如下:")
# print(target_modules)
from peft import get_peft_model, LoraConfig, TaskType

target_modules = [
    'conv_first', 'wqkv', 'proj', 'out_proj',
    'fc1', 'fc2', 'conv_after_body', 'conv_last'
]

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
    target_modules=target_modules
)

from peft.tuners.lora import LoraModel  # 👈 关键

model = LoraModel(model, lora_config,adapter_name="default")
# model.print_trainable_parameters()

# model.print_trainable_parameters()  # 可选：确认实际参与训练的参数




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

img_dir = '../data'
dataset = ColorizationDataset(img_dir, transform=T.Compose([T.Resize((128,128)), T.ToTensor(),]))
loader = DataLoader(dataset, batch_size=2, shuffle=True)

num_epochs = 5
scaler = torch.cuda.amp.GradScaler()  # ✅ 初始化放在外面，只需一次

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for step, (gray, color) in enumerate(loader):
        gray, color = gray.to(device), color.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # ✅ 使用 FP16 推理
            pred = model(gray)
            loss = criterion(pred, color)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * gray.size(0)  # ✅ 累计 loss（乘样本数）

    avg_loss = running_loss / len(loader.dataset)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}")


    # # 每1个epoch保存一次模型
    # if (epoch + 1) % 1 == 0:
    #     torch.save(model.state_dict(), f"./new_color_model_lab_epoch{epoch+1}.pth")

# 最后一次保存
torch.save(model.state_dict(), f"./new_color_model_lab_5.pth")