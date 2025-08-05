import os
import signal
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# === CONFIG ===
model_path = Path("C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/1st_model_yolo/yolo11m.pt")
inference_source_root = Path("C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/cropped_yolo_filtered")
polygon_csv_output = Path("C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/1st_model_yolo/runs/detect/polygon_corners/detection_corners.csv")
imgsz = 640
conf_thres = 0.15
allowed_classes = {0, 1}  # Only retain class 0 and 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def main():
    print(f"Using device: {device}")

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Load existing detection corners CSV
    if not polygon_csv_output.exists():
        raise FileNotFoundError(f"Detection corners CSV not found: {polygon_csv_output}")

    df = pd.read_csv(polygon_csv_output)
    if df.empty:
        raise ValueError("Detection corners CSV is empty.")

    print(f"Loaded {len(df)} detections from CSV")
    visualize_results(df)

if __name__ == '__main__':
    main()
