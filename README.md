# PetPlanion

## Dataset
Link : https://universe.roboflow.com/jccalugasupeduph/dog-and-cat-face-detection/dataset/1

Before training, set the paths for the train, test, and val datasets in the data.yaml file.

## Project Structure
```
├── video/ # Input videos for detection
│ └── *.mp4
├── csv_output/ # CSV files of detection results (optional)
│ └── *.csv
├── model/ # Trained YOLO model weights
│ └── best.pt # YOLOv5 model
│ └── yolov8_best.pt # YOLOv8 model
├── output/ # Output videos with bounding boxes
│ └── *.mp4
├── main.py # Main application entry (optional)
├── test_model.py # Inference script for YOLOv5
├── test_model_v8.py # Inference script for YOLOv8
├── train_yolov5.py # Training script for YOLOv5
├── train_yolov8.py # Training script for YOLOv8
```

## Train
You can train a YOLOv5 model using either of the two scripts provided:

#### 1. Using the custom script (train_yolov5.py)

Run the custom training script with:
```bash
python train_yolov5.py
```
#### 2. Using the standard YOLOv5 script (train.py)

First, clone the YOLOv5 repository:
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
```
Then, train the model from scratch:
```bash 
python train.py --data data.yaml --epochs 100 --weights '' --cfg yolov5n.yaml --batch-size 16
```
