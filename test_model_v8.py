import cv2
import os
from ultralytics import YOLO

# โหลดโมเดล YOLOv8
model = YOLO("model/yolov8_best.pt")
model.conf = 0.25  # confidence threshold (YOLOv8 ใช้ผ่าน args ใน predict ได้)

# โฟลเดอร์วิดีโอ input/output
video_folder = "video"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# ลูปผ่านทุกวิดีโอในโฟลเดอร์
for file_name in os.listdir(video_folder):
    if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        input_video = os.path.join(video_folder, file_name)
        output_video = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_out.avi")

        cap = cv2.VideoCapture(input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ตรวจจับวัตถุ
            results = model.predict(frame, imgsz=640, conf=0.25)

            for r in results:
                annotated = r.plot()
                out.write(annotated)

                # แสดง real-time (optional)
                # cv2.imshow("YOLOv8 Detection", annotated)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break

        cap.release()
        out.release()

# cv2.destroyAllWindows()

