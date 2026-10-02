# DoomBlocker

DoomBlocker is a real-time computer vision utility that monitors study sessions via webcam, detecting phone usage and prolonged distraction using YOLOv8, optical flow motion analysis, and Random Forest classification.

## Key Features
- **Real-Time Detection**: Leverages YOLOv8-Nano for object detection and confidence scoring.
- **Low False Positives**: Combines optical flow motion tracking (in lower-center bounds) with temporal smoothing to distinguish writing/studying from phone scrolling.
- **Audio Alerts**: Automatically triggers asynchronous audio notifications when distraction exceeds configurable time thresholds.

## Tech Stack
- **Languages**: Python 3.10+
- **Computer Vision**: OpenCV, Ultralytics (YOLOv8)
- **Machine Learning**: Scikit-Learn, NumPy
- **Storage/Models**: Joblib
