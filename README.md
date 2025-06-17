# PetPlanion

## Dataset
Link : https://universe.roboflow.com/jccalugasupeduph/dog-and-cat-face-detection/dataset/1

Before training, set the paths for the train, test, and val datasets in the data.yaml file. A sample data.yaml.

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