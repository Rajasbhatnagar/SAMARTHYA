import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from sklearn.metrics import r2_score

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
IMG_DIR = "C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/train/cropped/positive"
GEOM_CSV = "C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/train_geometry_features.csv"

# Load and filter data
geom_df = pd.read_csv(GEOM_CSV).rename(columns={"image": "filename"})
geom_df['filename'] = geom_df['filename'].astype(str).apply(os.path.basename)
valid_filenames = set(os.listdir(IMG_DIR))
print("Total images in folder:", len(valid_filenames))
print("Sample image files:", list(valid_filenames)[:5])
print("geom_df['filename'] sample:", geom_df['filename'].head())
print("Matching files count:", geom_df['filename'].isin(valid_filenames).sum())
filtered = geom_df[geom_df["filename"].isin(valid_filenames)].reset_index(drop=True)

if filtered.empty:
    raise ValueError("Filtered dataset is empty. Check your filters and data files.")

# Dataset class
class WakeImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        heading_rad = np.radians(row['heading_deg'])
        target = torch.tensor([np.sin(heading_rad), np.cos(heading_rad)], dtype=torch.float32)
        geom = torch.tensor([row['length'], row['width'], row['aspect_ratio']], dtype=torch.float32)
        return image, geom, target

# Transform with augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Split dataset
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(filtered, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

if train_df.empty or val_df.empty or test_df.empty:
    raise ValueError("Train/val/test split contains zero samples. Adjust your filtering or split ratios.")

train_dataset = WakeImageDataset(train_df, IMG_DIR, transform)
val_dataset = WakeImageDataset(val_df, IMG_DIR, transform)
test_dataset = WakeImageDataset(test_df, IMG_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model
class CNNHybridRegressor(nn.Module):
    def __init__(self):
        super(CNNHybridRegressor, self).__init__()
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, image, geom):
        features = self.backbone(image)
        x = torch.cat((features, geom), dim=1)
        return self.fc(x)

# Training
model = CNNHybridRegressor().to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):  # extended training
    model.train()
    train_loss = 0
    for images, geoms, targets in train_loader:
        images, geoms, targets = images.to(device), geoms.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images, geoms)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, geoms, targets in val_loader:
            images, geoms, targets = images.to(device), geoms.to(device), targets.to(device)
            outputs = model(images, geoms)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_dataset):.4f}, Val Loss: {val_loss/len(val_dataset):.4f}")

# Evaluation
model.eval()
y_true_deg, y_pred_deg, angular_errors = [], [], []
with torch.no_grad():
    for images, geoms, targets in test_loader:
        images, geoms = images.to(device), geoms.to(device)
        preds = model(images, geoms).cpu().numpy()
        sin_pred, cos_pred = preds[:, 0], preds[:, 1]
        pred_angles = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360

        true_sin = targets[:, 0].numpy()
        true_cos = targets[:, 1].numpy()
        true_angles = np.degrees(np.arctan2(true_sin, true_cos)) % 360

        angular_diff = np.abs((pred_angles - true_angles + 180) % 360 - 180)

        y_true_deg.extend(true_angles)
        y_pred_deg.extend(pred_angles)
        angular_errors.extend(angular_diff)

mean_ang_error = np.mean(angular_errors)
rmse = np.sqrt(np.mean(np.square(angular_errors)))
r2 = r2_score(y_true_deg, y_pred_deg)

print(f"Angular MAE: {mean_ang_error:.2f}°, Angular RMSE: {rmse:.2f}°, R²: {r2:.2f}")

# Save model
torch.save(model.state_dict(), "cnn_hybrid_heading_regressor_resnet34.pth")