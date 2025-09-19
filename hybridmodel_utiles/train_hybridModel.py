from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import os

from HybridApp.hybrid_model import ImprovedDeepfakeDetector

# --- CONFIG ---
dataset_root = r"C:\Users\VISHWAJEET\Documents\dataset\real_and_fake_face"
batch_size = 4
epochs = 2
lr = 1e-4

# Save inside hybridModel folder (consistent with hybrid_model.py)
base_dir = os.path.dirname(os.path.dirname(__file__))  # go up to Major-Project
save_dir = os.path.join(base_dir, "hybridModel")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "deepfake_hybrid.pth")

# --- Prepare Detector & Transform ---
detector = ImprovedDeepfakeDetector()   # ✅ defined detector here
transform = detector.transform

# --- Dataset ---
dataset = ImageFolder(root=dataset_root, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = detector.model.to(device)

# Freeze backbones (optional, makes training faster)
for p in model.resnet.parameters():
    p.requires_grad = False
for p in model.vit.parameters():
    p.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = nn.CrossEntropyLoss()

# --- Training loop ---
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

print("✅ Training complete")

# --- Save trained model ---
torch.save(model.state_dict(), save_path)
print(f"✅ Model saved to {save_path}")
