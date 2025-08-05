import os
import signal
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

# === CONFIG ===
model_path = Path("C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/1st_model_yolo/yolo11m.pt")
inference_source_root = Path("C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/cropped_yolo_filtered")
polygon_csv_output = Path("C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/1st_model_yolo/runs/detect/polygon_corners/detection_corners.csv")
heading_model_path = Path("C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/3rd_model_hybrid(restnet34MLP)/cnn_hybrid_heading_regressor_resnet34.pth")
heading_predictions_csv = Path("C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/heading_predictions.csv")
imgsz = 640
conf_thres = 0.15
allowed_classes = {0, 1}

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset for heading estimation
class HeadingDataset(Dataset):
    def __init__(self, df, image_root, transform=None):
        self.df = df
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        img_path = self.image_root / filename
        if not img_path.exists():
            return None
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        geom = torch.tensor([row['x2'] - row['x1'], row['y3'] - row['y2'], abs((row['x2'] - row['x1']) / (row['y3'] - row['y2'] + 1e-6))], dtype=torch.float32)
        return image, geom, filename

# Model definition
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

def handle_exit(sig, frame):
    print("\nProcess interrupted.")
    sys.exit(0)

def visualize_results(df):
    if df.empty:
        print("No detection results to visualize.")
        return

    plt.figure(figsize=(10, 6))
    df['class_id'].value_counts().sort_index().plot(kind='bar')
    plt.title("Detections per Class")
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.show()

def collate_skip_none(batch):
    return [item for item in batch if item is not None]

def main():
    print(f"Using device: {device}")

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Load detection corners CSV
    if not polygon_csv_output.exists():
        raise FileNotFoundError(f"Detection corners CSV not found: {polygon_csv_output}")
    df = pd.read_csv(polygon_csv_output)
    if df.empty:
        raise ValueError("Detection corners CSV is empty.")

    print(f"Loaded {len(df)} detections from CSV")
    visualize_results(df)

    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = HeadingDataset(df, inference_source_root, transform)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_skip_none)

    # Load hybrid model
    model = CNNHybridRegressor().to(device)
    model.load_state_dict(torch.load(heading_model_path, map_location=device))
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Estimating Headings"):
            if not batch:
                continue
            images, geoms, filenames = zip(*batch)
            images = torch.stack(images).to(device)
            geoms = torch.stack(geoms).to(device)
            preds = model(images, geoms).cpu().numpy()
            angles = np.degrees(np.arctan2(preds[:, 0], preds[:, 1])) % 360
            for fname, angle in zip(filenames, angles):
                predictions.append({"filename": fname, "predicted_heading_deg": angle})

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(heading_predictions_csv, index=False)
    print(f"Saved heading predictions to {heading_predictions_csv}")

if __name__ == '__main__':
    main()
