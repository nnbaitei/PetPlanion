import os
import subprocess
import sys

YOLO_DIR = "yolov5"
DATASET_YAML_PATH = r"../data.yaml"

TRAIN_PARAMS = {
    "img": 640,
    "batch": 16,
    "epochs": 100,
    "data": DATASET_YAML_PATH,
    "weights": "yolov5n.pt",
    "name": "result"
}

def run_command(command, description):
    subprocess.run(command, check=True)
    print(description)

def main():
    if not os.path.exists(YOLO_DIR):
        git_command = ["git", "clone", "https://github.com/ultralytics/yolov5.git"]
        run_command(git_command, "Downloading YOLOv5 Repository")

    os.chdir(YOLO_DIR)

    pip_command = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    run_command(pip_command, "Installing YOLOv5 Requirements")

    train_cmd_list = [sys.executable, "train.py"]
    for key, value in TRAIN_PARAMS.items():
        train_cmd_list.append(f"--{key}")
        train_cmd_list.append(str(value))

    run_command(train_cmd_list, "Model Training")

if __name__ == "__main__":
    main()