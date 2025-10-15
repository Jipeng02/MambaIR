from basicsr.archs.mambairv2_arch import MambaIRv2
import torch
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

input_tensor = torch.randn(2, 1, 64, 64).to(device)  # 输入也放到 device
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([2, 2, 64, 64])