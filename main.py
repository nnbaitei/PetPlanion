import cv2
from ultralytics import YOLO
import os 
import csv
from datetime import datetime
import sys # เพิ่ม Library 'sys' เพื่อใช้ในการออกจากโปรแกรมเมื่อเกิดข้อผิดพลาดร้ายแรง

# --- ค่าคงที่และการตั้งค่า (Constants and Settings) ---
CONF_THRESHOLD = 0.8
FRONT_CAM_AREA_RATIO = 0.5
FRONT_CAM_HEIGHT_RATIO = 0.5

VIDEO_FILE = "large.mp4"
MODEL_BODY_FILE = "yolov8n.pt"
MODEL_FACE_FILE = "yolov8_best.pt"
OUTPUT_CSV_FILE = 'data.csv'


def processFrame(model_body, model_face, frame, width, height):
    """
    ประมวลผล frame เดียว โดยตรวจจับร่างกายสัตว์เลี้ยงก่อน
    หากพบและอยู่ในระยะใกล้ จึงจะทำการตรวจจับใบหน้าต่อ
    """
    # สร้าง Dictionary เพื่อเก็บผลลัพธ์การตรวจจับ เริ่มต้นด้วยค่า default
    body_detection = {
        'found': False, 
        'class_name': 'none', 
        'confidence': 0,
        'front_cam': 0, 
        'box': None
    }
    face_detection = {
        'found': False, 
        'class_name': 'none', 
        'confidence': 0,
        'box': None, 
        'choosing': 0
    }
    
    # ขั้นที่ 1: ตรวจจับร่างกายสัตว์เลี้ยง (สุนัข, แมว)
    # verbose=False เพื่อไม่ให้แสดง log ที่ไม่จำเป็นของ YOLO ใน console
    body_results = model_body.predict(frame, imgsz=640, verbose=False) 
    for result in body_results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model_body.names[class_id]
                if class_name in ['dog', 'cat']:
                    conf = float(box.conf[0])
                    # กรองเฉพาะ detection ที่ดีที่สุด (มีความมั่นใจสูงสุดและเกิน threshold) ในเฟรมนี้
                    if conf > CONF_THRESHOLD and conf > body_detection['confidence']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        box_height = y2 - y1
                        
                        # อัปเดตข้อมูลการตรวจจับร่างกาย
                        body_detection.update({
                            'found': True, 
                            'class_name': class_name, 
                            'confidence': conf,
                            'box': (x1, y1, x2, y2)
                        })
                        
                        # ตรวจสอบว่าสัตว์เลี้ยงเข้ามาใกล้กล้องหรือไม่
                        if area > ((width * height) * FRONT_CAM_AREA_RATIO) or box_height > (height * FRONT_CAM_HEIGHT_RATIO):
                            body_detection['front_cam'] = 1
                        else:
                            body_detection['front_cam'] = 0
    
    # ขั้นที่ 2: ถ้าเจอตัวและอยู่ใกล้กล้อง ให้ตรวจจับใบหน้า
    if body_detection['front_cam'] == 1:
        face_results = model_face.predict(frame, imgsz=640, verbose=False)
        for result in face_results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    # กรองเฉพาะ detection ใบหน้าที่ดีที่สุด
                    if conf > CONF_THRESHOLD and conf > face_detection['confidence']:
                        class_id = int(box.cls)
                        class_name = model_face.names[class_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # อัปเดตข้อมูลการตรวจจับใบหน้า
                        face_detection.update({
                            'found': True, 
                            'class_name': class_name, 
                            'confidence': conf,
                            'box': (x1, y1, x2, y2)
                        })

                        # Logic การตัดสินใจเลือกซ้าย/ขวา
                        base_line = x2 - y2
                        mid_line = y2 - y1
                        th = 0.5
                        if base_line < (y2 // 2) and mid_line < ((x2 // 2) - th):
                            face_detection['choosing'] = 1 # left
                        elif base_line < (y2 // 2) and mid_line > ((x2 // 2) + th):
                            face_detection['choosing'] = 2 # right
                        else:
                            face_detection['choosing'] = 0
    
    return body_detection, face_detection

def drawDetections(frame, body_detection, face_detection):
    """วาดกรอบและ label ต่างๆ ลงใน frame"""
    # วาดกรอบสำหรับ "ร่างกาย" ถ้าตรวจพบ
    if body_detection['found'] and body_detection['box']:
        x1, y1, x2, y2 = body_detection['box']
        label = f"Body: {body_detection['class_name']} {body_detection['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 32, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 32, 0), 2)

    # วาดกรอบสำหรับ "ใบหน้า" ถ้าเงื่อนไขครบ (อยู่ใกล้กล้อง และ ตรวจพบใบหน้า)
    if body_detection['front_cam'] == 1 and face_detection['found'] and face_detection['box']:
        x1, y1, x2, y2 = face_detection['box']
        label = f"Face: {face_detection['class_name']} {face_detection['confidence']:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.line(frame, (center_x, y1), (center_x, y2), (0, 128, 0), 2) # midline
        cv2.line(frame, (x1, center_y), (x2, center_y), (0, 255, 255), 2) # baseline
   
    return frame

def main():
    """
    ฟังก์ชันหลักสำหรับเปิดวิดีโอ, ประมวลผล, แสดงผล, และบันทึกข้อมูล พร้อมการจัดการ Error
    """
    video_path = os.path.join("video", VIDEO_FILE)
    model_body_path = MODEL_BODY_FILE
    model_face_path = os.path.join("model", MODEL_FACE_FILE)

    # --- 1. ตรวจสอบว่าไฟล์ที่จำเป็นทั้งหมดมีอยู่จริงหรือไม่ ---
    # การตรวจสอบนี้ช่วยป้องกันไม่ให้โปรแกรมเริ่มทำงานแล้วไปแครชกลางคัน
    required_files = [video_path, model_body_path, model_face_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"--- FATAL ERROR ---")
            print(f"No such file: '{file_path}'")
            print("Check the file path and try again.")
            sys.exit(1) # สั่งให้โปรแกรมปิดทันทีพร้อมแจ้งสถานะ Error

    # --- 2. โหลดโมเดล YOLO พร้อมดักจับข้อผิดพลาด (Error Handling) ---
    try:
        model_body = YOLO(model_body_path)
        model_face = YOLO(model_face_path)
    except Exception as e:
        print(f"--- FATAL ERROR ---")
        print(f"Can't load YOLO model: {e}")
        sys.exit(1)

    # --- 3. เปิดไฟล์วิดีโอ พร้อมดักจับข้อผิดพลาด ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"--- FATAL ERROR ---")
        print(f"Can't open video: '{video_path}'")
        print("Check the file path and try again.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Info: {width}x{height} @ {fps:.2f} FPS")
    print("Press 'q' to quit")

    # --- 4. เปิดไฟล์ CSV และเริ่มลูปประมวลผล ---
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Time", "Front_cam", "Choose"]) # เขียน Header
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                body_detection, face_detection = processFrame(model_body, model_face, frame, width, height)
                frame = drawDetections(frame, body_detection, face_detection)
                
                # เตรียมข้อความแสดงสถานะ
                status_text = "Body: Not Found"
                if body_detection['found']:
                    status_text = f"Body: {body_detection['class_name']} ({body_detection['confidence']:.2f})"
                    if face_detection['found']:
                        status_text = f"Face: {face_detection['class_name']} ({face_detection['confidence']:.2f})"

                choose_map = {0: "Not Choose", 1: "Choose Left", 2: "Choose Right"}
                choose_text = choose_map.get(face_detection['choosing'])
                
                # แสดงข้อความบนเฟรม
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, choose_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Pet Detection', frame)
                csv_writer.writerow([datetime.now(), body_detection['front_cam'], face_detection['choosing']])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except (IOError, PermissionError) as e:
        print(f"--- FATAL ERROR ---")
        print(f"Could not open CSV file: '{OUTPUT_CSV_FILE}'")
        print(f"Details: {e}")
        print("Check the file path and try again.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()