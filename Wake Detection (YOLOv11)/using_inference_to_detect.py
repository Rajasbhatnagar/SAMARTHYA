import os
import signal
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import pandas as pd
from multiprocessing import freeze_support
from tqdm import tqdm

# === CONFIG ===
model_path = r"C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/1st_model_yolo/yolo11m.pt"
inference_source_root = Path(r"C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/1st_model_yolo/runs/inference/opensar_infer/crops")
polygon_csv_output = Path("runs/detect/polygon_corners/detection_corners.csv")
imgsz = 640
conf_thres = 0.15
allowed_classes = {0, 1}  # Only retain class 0 and 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    freeze_support()
    print(f"Using device: {device}")

    def handle_exit(sig, frame):
        print("\nProcess interrupted.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # === INFERENCE WITH PRE-TRAINED MODEL ===
    print("\nRunning YOLO inference to generate labels...")
    model = YOLO(model_path)

    image_paths = list(inference_source_root.rglob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found under {inference_source_root}")

    label_output_dir = Path("runs/detect/predict/labels")
    label_output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        result = model.predict(
            source=str(img_path),
            save=False,
            save_txt=True,
            save_conf=True,
            imgsz=imgsz,
            conf=conf_thres,
            project='runs/detect',
            name='predict',
            exist_ok=True,
            device=device
        )

        label_file = label_output_dir / (img_path.stem + ".txt")
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls, cx, cy, w, h = map(float, parts[:5])

                    if int(cls) not in allowed_classes:
                        continue

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = y1
                    x3 = x2
                    y3 = cy + h / 2
                    x4 = x1
                    y4 = y3

                    all_rows.append({
                        "filename": img_path.name,
                        "class_id": int(cls),
                        "x1": round(x1, 6), "y1": round(y1, 6),
                        "x2": round(x2, 6), "y2": round(y2, 6),
                        "x3": round(x3, 6), "y3": round(y3, 6),
                        "x4": round(x4, 6), "y4": round(y4, 6)
                    })

    df = pd.DataFrame(all_rows)
    polygon_csv_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(polygon_csv_output, index=False)
    print(f"Saved detection corners to: {polygon_csv_output}")
