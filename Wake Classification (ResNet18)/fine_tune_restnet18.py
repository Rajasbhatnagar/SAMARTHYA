import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import ResNet18_Weights
from torchvision.datasets.folder import make_dataset

# === PNG IMAGE LOADER ===
def png_loader(path):
    return Image.open(path).convert("RGB")

# === CUSTOM DATASET FOR PNG ===
class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        samples = make_dataset(root, class_to_idx, extensions=('png',))

        self.root = root
        self.loader = png_loader
        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

# === MAIN FUNCTION ===
def main():
    # === CONFIG ===
    data_dir = r"C:\Users\rajas\OneDrive\Desktop\drdo_enemy_vessel\OpenSARWake_1.0\train\cropped"
    batch_size = 16
    epochs = 20
    lr = 1e-4
    patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print("GPU model:", torch.cuda.get_device_name(0))

    # === TRANSFORMS ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # === LOAD DATA ===
    full_dataset = CustomImageFolder(data_dir, transform=transform)
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=[full_dataset.samples[i][1] for i in range(len(full_dataset))]
    )

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # === MODEL ===
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    train_acc_list, val_acc_list, loss_list = [], [], []
    best_val_acc, wait = 0, 0

    # === TRAINING LOOP ===
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            with autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)
        loss_list.append(running_loss)

        # === VALIDATION ===
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = out.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                total += y.size(0)
                correct += pred.eq(y).sum().item()
        val_acc = 100 * correct / total
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch+1}: Loss={running_loss:.3f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), "resnet18_cropped_png_classifier.pth")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # === CONFUSION MATRIX ===
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=full_dataset.classes)
    disp.plot(cmap="Blues")
    plt.title("Validation Set Confusion Matrix")
    plt.show()
    disp.figure_.savefig("confusion_matrix.png")

    # === PLOTS ===
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# === WINDOWS-SAFE ENTRY POINT ===
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
