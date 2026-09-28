import os
import cv2
import time
import numpy as np
from src.extractors import FeatureExtractor

def main():
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(0)

    DATA_DIR = "data"
    CLASSES = ["studying", "scrolling", "idle"]
    SEQUENCE_LENGTH = 30

    for c in CLASSES:
        os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)

    print("\n" + "="*50)
    print(" DOOMBLOCKER AI - DATASET RECORDER")
    print("="*50)
    print("Controls:")
    print(" Press '1' -> Record STUDYING/TYPING data")
    print(" Press '2' -> Record SCROLLING/SWIPING data")
    print(" Press '3' -> Record IDLE/LOOKING AWAY data")
    print(" Press 'q' -> Quit Recorder\n")

    current_class = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        data = extractor.process_frame(frame)
        output_frame = data["annotated_frame"]

        cv2.putText(output_frame, "PRESS: [1] Studying  [2] Scrolling  [3] Idle  [Q] Quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if current_class is not None:
            cv2.putText(output_frame, f"RECORDING CLASS: {current_class.upper()}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("DoomBlocker AI - Dataset Recorder", output_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_class = "studying"
        elif key == ord('2'):
            current_class = "scrolling"
        elif key == ord('3'):
            current_class = "idle"

        if current_class is not None:
            print(f"\n[+] Preparing to record 3-second buffer for '{current_class}'... Get Ready!")
            time.sleep(1.0)

            class_dir = os.path.join(DATA_DIR, current_class)
            existing_files = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])

            for seq_idx in range(5):
                sequence_data = []

                for f_idx in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    metrics = extractor.process_frame(frame)
                    
                    frame_features = [
                        metrics["thumb_velocity"],
                        metrics["head_tilt"],
                        1.0 if metrics["phone_detected"] else 0.0,
                        metrics["phone_conf"]
                    ]
                    sequence_data.append(frame_features)

                    rec_frame = metrics["annotated_frame"]
                    cv2.putText(rec_frame, f"REC '{current_class}': Seq {seq_idx+1}/5 - Frame {f_idx+1}/30", 
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("DoomBlocker AI - Dataset Recorder", rec_frame)
                    cv2.waitKey(20)

                file_path = os.path.join(class_dir, f"seq_{existing_files + seq_idx}.npy")
                np.save(file_path, np.array(sequence_data))
                print(f" Saved: {file_path}")

            print(f"[✔] Batch completed for '{current_class}'. Press 1, 2, or 3 for next batch.")
            current_class = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
