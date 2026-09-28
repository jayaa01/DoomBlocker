import cv2
import numpy as np
from ultralytics import YOLO

class FeatureExtractor:
    def __init__(self):
        print("Loading YOLOv8-Nano Model...")
        self.yolo = YOLO("yolov8n.pt")
        
        self.PERSON_CLASS_ID = 0  # Person / face location
        self.PHONE_CLASS_ID = 67  # Cell phone

        # State tracking for out-of-frame phone memory and motion velocity
        self.prev_hand_y = None
        self.phone_memory_counter = 0  # Frame buffer to remember phone presence

    def process_frame(self, frame):
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        phone_detected = False
        phone_conf = 0.0
        head_tilt = 0.0
        hand_velocity = 0.0

        # 1. YOLO Inference with Higher Confidence Threshold
        yolo_results = self.yolo(frame, verbose=False)[0]

        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Filter False Positives: Require >55% Confidence & Smartphone Aspect Ratio
            if cls_id == self.PHONE_CLASS_ID and conf > 0.55:
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                aspect_ratio = box_h / float(box_w)

                # Smartphone bounding boxes usually have height/width ratio between 1.2 and 3.0 (portrait/tilted)
                if 0.8 <= aspect_ratio <= 3.2:
                    phone_detected = True
                    phone_conf = conf
                    self.phone_memory_counter = 45  # Remember phone for ~45 frames (~1.5s) even if obscured
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Phone: {phone_conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif cls_id == self.PERSON_CLASS_ID and conf > 0.4:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                # Head position relative to overall frame
                face_center_y = y1 + ((y2 - y1) * 0.15)
                head_tilt = face_center_y / h

        # Handle Out-of-Frame Phone Memory Decay
        if not phone_detected and self.phone_memory_counter > 0:
            self.phone_memory_counter -= 1
            phone_detected = True
            phone_conf = 0.50  # Memory state confidence

        # 2. Track Vertical Optical Motion in Lower Half (Thumb/Hand Swiping)
        lower_region = gray[int(h * 0.45):, :]
        corners = cv2.goodFeaturesToTrack(lower_region, maxCorners=15, qualityLevel=0.2, minDistance=5)
        
        if corners is not None:
            avg_y = np.mean(corners[:, 0, 1])
            if self.prev_hand_y is not None:
                # Vertical motion displacement normalized
                hand_velocity = abs(avg_y - self.prev_hand_y) / (h * 0.55)
            self.prev_hand_y = avg_y

        hand_landmarks = [0.0] * 63

        return {
            "annotated_frame": frame,
            "hand_landmarks": hand_landmarks,
            "thumb_velocity": hand_velocity,
            "phone_detected": phone_detected,
            "phone_conf": phone_conf,
            "head_tilt": head_tilt
        }
