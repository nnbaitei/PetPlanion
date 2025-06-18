import cv2
from ultralytics import YOLO
import os 
import csv
import numpy as np

def fullBody(model, cap, csv_name):
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {width}x{height} @ {fps:.2f} FPS")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_names = model.names[class_id]
                if class_names in ['dog', 'cat']:
                    class_name = class_names
                    conf = box.conf[0]
                    if conf > 0.8: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = ((x2-x1) * (y2-y1))
                        h = y2 - y1
                        label = f'{class_names} {conf:.2f}'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
        if area > (width*height)*0.5 or h > (height*0.5):
            front_cam = 1
        else:
            front_cam = 0

        with open(csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            header = ['front_cam', 'class_name']
            data_row = [front_cam, class_name]
            csv_writer.writerow(header)
            csv_writer.writerow(data_row)

        cv2.imshow('YOLOv8 Inference', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def onlyFace(model, cap, csv_name):
    pass


def main():
    csv_name = '../csv_output/data.csv'

    video_path = r"../video/istockphoto-1978695737-640_adpp_is.mp4"
    cap = cv2.VideoCapture(video_path)

    model = '../model/best.pt'
    fullBody(model, cap, csv_name)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
