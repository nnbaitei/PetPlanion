# PetPlanion

## Train
You can train a YOLOv5 model using either of the two scripts provided:

#### 1. Using the custom script (train_yolov5.py)

Run the custom training script with:
```bash
python train_yolov5.py
```
#### 2. Using the standard YOLOv5 script (train.py)

Train a YOLOv5 model from scratch with the standard script:
```bash
python train.py --data data.yaml --epochs 100 --weights '' --cfg yolov5n.yaml --batch-size 16
```