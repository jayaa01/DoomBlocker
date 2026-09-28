import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.extractors import FeatureExtractor

def main():
    cap = cv2.VideoCapture(0)
    extractor = FeatureExtractor()

    print("Starting Video Stream. Press 'q' to exit...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        data = extractor.process_frame(frame)
        output_frame = data["annotated_frame"]

        cv2.putText(output_frame, f"Thumb Velocity: {data['thumb_velocity']:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(output_frame, f"Head Tilt Delta: {data['head_tilt']:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("DoomBlocker AI - Feature Extraction Test", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
