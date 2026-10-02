# DoomBlocker 
> **Real-Time AI Focus Monitor & Distraction Detection System**

DoomBlocker is a hybrid computer vision desktop utility engineered to enforce study focus by detecting smartphone usage and scrolling behaviors from standard webcam feeds. By combining lightweight deep learning (**YOLOv8**), spatial optical flow tracking (**OpenCV**), and **Random Forest classification**, DoomBlocker distinguishes between normal study movements (e.g., taking notes, turning pages) and active smartphone distractions, triggering real-time alerts.

---

##  Project Overview

Maintaining deep focus while studying near a smartphone is a common challenge. Standard object detectors often trigger false positives or miss fine-grained hand interactions. DoomBlocker solves this using a **two-tier decision engine**:

1. **Object & Motion Tracking**: Real-time object recognition combined with targeted Lucas-Kanade optical flow within a bounded region-of-interest (ROI).
2. **Temporal & Heuristic Decision Rules**: Multi-frame rolling averages and threshold confidence scoring to eliminate momentary motion artifacts.

---

##  Key Features

* **High-Efficiency Object Detection**: Uses `YOLOv8-Nano` optimized for low-latency inference on standard CPU webcam streams.
* **Targeted Optical Flow Tracking**: Restricts feature tracking to the lower-center region of the frame to capture hand/thumb gestures while ignoring background motion and head posture shifts.
* **Smart Memory Decay**: Maintains phone tracking state across brief occlusions or partial frame exits.
* **False Positive Reduction**: Employs a 7-frame rolling temporal window to smooth out predictions and avoid false alarms triggered by rapid page turns or shifting positions.
* **Non-Blocking Asynchronous Alerts**: Triggers real-time audio cues and visual overlay alerts without freezing the video processing loop.

---

##  System Architecture

```text
[ Live Webcam Feed ]
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │                   Feature Extractor                    │
 ├──────────────────────────────┬─────────────────────────┤
 │ YOLOv8 Object Detection      │ Localized Optical Flow  │
 │ • Cell Phone Bounding Box    │ • Corner Feature Point  │
 │ • Confidence Metrics          │   Tracking (ROI)        │
 │ • Face/Head Position         │ • Thumb Velocity Vector │
 └──────────────┬───────────────┴────────────┬────────────┘
                │                            │
                └──────────────┬─────────────┘
                               │
                               ▼
 ┌────────────────────────────────────────────────────────┐
 │                Hybrid Decision Engine                  │
 ├────────────────────────────────────────────────────────┤
 │ • Random Forest Classifier Prediction                  │
 │ • Rule Override (Phone Confidence & Motion Thresholds) │
 │ • Temporal Smoothing (Rolling Window Deque)            │
 └─────────────────────────────┬──────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   [ State: STUDYING ]                  [ State: SCROLLING ]
   • Green Overlay                      • Red Overlay
   • Timer Reset                        • Sound Alarm Trigger
