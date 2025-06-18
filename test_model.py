import cv2
import torch
from pathlib import Path
from datetime import datetime
import os
from ultralytics import YOLO

# โหลดโมเดล YOLO

model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt')

model.conf = 0.25  # confidence threshold

# ตั้งค่าชื่อไฟล์
file_name = "istockphoto-1978695737-640_adpp_is.mp4"
video_folder = "video"
for file_name in os.listdir(video_folder):
    if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        input_video = f'video/{file_name}'
        output_video = f"output/{file_name}.avi"
        # เปิดวิดีโอ
        cap = cv2.VideoCapture(input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # ตั้งค่าตัวเขียนวิดีโอ
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # ตรวจจับ
            results = model(frame)
            # วาดกล่อง
            annotated = results.render()[0]
            out.write(annotated)

    # แสดงแบบ real-time (optional)
#    cv2.imshow('YOLO Detection', annotated)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

        cap.release()
        out.release()
#cv2.destroyAllWindows()

