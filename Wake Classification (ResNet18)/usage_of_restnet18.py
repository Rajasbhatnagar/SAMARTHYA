import os
import shutil
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Config
WAKE_TYPE_0_DIR = r"C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/1st_model_yolo/runs/inference/opensar_infer/crops/wake_type_0"
WAKE_TYPE_2_DIR = r"C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/1st_model_yolo/runs/inference/opensar_infer/crops/wake_type_2"
MODEL_PATH = r"C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/2nd_model_finetuned_restnet18/resnet18_cropped_png_classifier.pth"
OUTPUT_CSV = "filtered_positive_wakes.csv"
SAVE_DIR = "filtered_images"
BATCH_SIZE = 64

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset
class CroppedWakeDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        exts = (".png", ".jpg", ".jpeg")
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(exts)]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, path

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, weights_only= True))
model = model.to(device)
model.eval()

# Classification function and copy valid images
def classify_and_filter(folder_path, label):
    dataset = CroppedWakeDataset(folder_path, transform=transform)
    print(f"Found {len(dataset)} image files in {folder_path}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    filtered = []
    dest_folder = os.path.join(SAVE_DIR, label)
    os.makedirs(dest_folder, exist_ok=True)

    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc=f"Processing {folder_path}"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            for path, pred in zip(paths, preds.cpu().numpy()):
                if pred == 1:
                    filtered.append(path)
                    shutil.copy(path, dest_folder)

    return filtered

def main():
    # Debug file counts
    exts = (".png", ".jpg", ".jpeg")
    num_0 = len([f for f in os.listdir(WAKE_TYPE_0_DIR) if f.lower().endswith(exts)])
    num_2 = len([f for f in os.listdir(WAKE_TYPE_2_DIR) if f.lower().endswith(exts)])
    print("Image files in wake_type_0:", num_0)
    print("Image files in wake_type_2:", num_2)

    # Run classification and copy filtered
    filtered_0 = classify_and_filter(WAKE_TYPE_0_DIR, "wake_type_0")
    filtered_2 = classify_and_filter(WAKE_TYPE_2_DIR, "wake_type_2")

    # Combine and save final CSV
    combined = filtered_0 + filtered_2
    print(f"Total valid wake images: {len(combined)}")

    if combined:
        df = pd.DataFrame({"filename": [os.path.basename(f) for f in combined]})
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved final filtered list to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
