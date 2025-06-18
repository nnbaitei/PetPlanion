# train_yolov8.py
import os
from ultralytics import YOLO

# === Config ===
MODEL_NAME = "yolov8n.pt"                      # หรือ yolov8s.pt, yolov8m.pt, etc.
DATASET_YAML_PATH = r"data.yaml"            # path ไปยัง Roboflow dataset (YOLO format)
SAVE_NAME = "yolov8n_result"               # ชื่อ folder ผลลัพธ์ใน runs/
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100
DEVICE = 0  # 0 = GPU, 'cpu' = ใช้ CPU

# === Training ===
def main():
    print("🚀 Starting YOLOv8 training...")
    model = YOLO(MODEL_NAME)  # load pretrained model
    model.train(
        data=DATASET_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name=SAVE_NAME
    )
    print("✅ Training complete.")

if __name__ == "__main__":
    main()

