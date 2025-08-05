import os
import shutil
import signal
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
from PIL import Image
from multiprocessing import freeze_support

# === CONFIG ===
# Paths
yaml_path = "opensar.yaml"
model_name = "yolo11m.pt"  # Updated to use YOLO11
save_dir = "runs/detect/opensar_train"
train_image_dir = r"C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/train/images"
val_image_dir = r"C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/val/images"
test_image_dir = r"C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/test/images"
output_crop_folder = r"C:/Users/rajas/OneDrive/Desktop/drdo_enemy_vessel/OpenSARWake_1.0/cropped_yolo"

# Detection params
imgsz = 640
conf_thres = 0.15

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    freeze_support()
    print(f"Using device: {device}")

    # === HANDLER FOR EARLY EXIT ===
    def handle_exit(sig, frame):
        print("\nEarly stop detected. Saving model if training...")
        try:
            model.trainer.save_model()
            print("Model saved successfully.")
        except Exception as e:
            print(f"Warning: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # === TRAIN YOLO MODEL ===
    print("\nTraining YOLO11 on OpenSARWake...")
    model = YOLO(model_name)
    model.train(data=yaml_path, epochs=50, imgsz=imgsz, device=device, project="runs/detect", name="opensar_train")

    # === EVALUATE MODEL ===
    print("\nEvaluating trained model...")
    best_model_path = os.path.join(save_dir, "weights", "best.pt")
    model = YOLO(best_model_path)
    metrics = model.val(data=yaml_path, split="test", device=device)
    print("\nEvaluation Metrics:", metrics)

    # === PREDICT ON TEST SET ===
    print("\nPredicting on test set...")
    output_dir = os.path.join(save_dir, "test_predictions")
    os.makedirs(output_dir, exist_ok=True)
    model.predict(source=test_image_dir, save=True, save_txt=True, save_conf=True,
                  project=save_dir, name="test_predictions", device=device)
    print(f"\nPredictions saved to: {output_dir}")

    # === INFERENCE + CROP SAVING ===
    print("\nRunning inference and saving cropped wakes...")
    results = model.predict(
        source=train_image_dir,
        save=True,
        save_crop=True,
        save_txt=False,
        imgsz=imgsz,
        conf=conf_thres,
        project='runs/inference',
        name='opensar_infer',
        exist_ok=True,
        device=device   
    )

    # === MOVE CROPS TO FINAL FOLDER ===
    default_crop_path = Path("runs/inference/opensar_infer/crops")
    output_crop_folder = Path(output_crop_folder)
    os.makedirs(output_crop_folder, exist_ok=True)

    for class_dir in default_crop_path.glob("*"):
        for file in class_dir.glob("*"):
            target_path = output_crop_folder / file.name
            shutil.copy(file, target_path)

    print(f"Cropped detections saved to: {output_crop_folder}")
