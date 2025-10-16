import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lab_dataset import LabColorizationDataset
from color_model import model  # <-- import your initialized model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Ensure model is on the right device

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

img_dir = '/path/to/your/training/images'
dataset = LabColorizationDataset(img_dir, img_size=64)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for L, ab in loader:
        L, ab = L.to(device), ab.to(device)
        optimizer.zero_grad()
        ab_pred = model(L)  # [B,2,H,W]
        loss = criterion(ab_pred, ab)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * L.size(0)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(loader.dataset):.6f}")

torch.save(model.state_dict(), "color_model_lab_trained.pth")