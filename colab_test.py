from basicsr.archs.mambairv2_arch import MambaIRv2
import torch
import os
from PIL import Image
import torchvision.transforms as T
import glob
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaIRv2(
    img_size=128,
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

# Load checkpoint
checkpoint = torch.load('/content/drive/MyDrive/full_finetuned_final_lab_2.pth', map_location='cpu')
state_dict = checkpoint.get('params', checkpoint)
model.load_state_dict(state_dict, strict=False)


img_dir = '/content/drive/MyDrive/1k'  # replace with your test image directory

# collect common image extensions (assume folder contains only grayscale images)
extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
image_paths = []
for ext in extensions:
    image_paths.extend(sorted(glob.glob(os.path.join(img_dir, ext))))

# keep original size conversion
transform = T.ToTensor()

output_dir = os.path.join(img_dir, 'predict_output')
os.makedirs(output_dir, exist_ok=True)

model.eval()
print(f"Found {len(image_paths)} images for inference in '{img_dir}'. Results will be saved to '{output_dir}'.")

# determine model window size if available (fallback to 16)
win = getattr(model, 'window_size', 16)

for img_path in image_paths:
    base = os.path.basename(img_path)
    # load as grayscale
    img = Image.open(img_path).convert('L')
    gray_tensor = transform(img)  # (1, H, W)
    # repeat channels if model expects 3 channels (keeps previous behaviour)
    gray_stacked = gray_tensor.repeat(3, 1, 1)

    # clip window_size to prevent misalignment 
    _, h, w = gray_stacked.shape
    h_ = h - h % win
    w_ = w - w % win
    gray_stacked = gray_stacked[:, :h_, :w_]

    gray_input = gray_stacked.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(gray_input)
    pred = pred.clamp(0, 1).squeeze(0).cpu()

    out_path = os.path.join(output_dir, base)
    save_image(pred, out_path)
    print(f"✓ {base} -> {os.path.relpath(out_path, img_dir)}")

print(f"✅ All inferences completed and saved to '{output_dir}'")
