S.I.G.N.S.I.G.H.T. — Seamless Interpretation, Gesture Navigation & Smart Interactive Gesture Handling Tool

SignSight is a real-time Sign Language Recognition system built to translate hand gestures into English alphabets, digits, and simple words. Powered by YOLO-based gesture detection and deployed through Streamlit, the system focuses on accessibility, accuracy, and an intuitive visual experience.

Its goal is to support inclusive communication by offering fast, lightweight gesture recognition adapted for academic, assistive, and experimental use-cases.

_____________________________________________________________________________________________________________________________________

Overview

SignSight analyzes live webcam input, detects the user’s hand sign, and instantly maps it to the predicted character or word. A smoothing system and prediction history maintain stability even in noisy environments.

The interface is optimized for readability and real-time feedback, offering fingerspelling mode, word mode, and customizable detection settings. With a streamlined design and efficient model loading, SignSight supports gesture recognition without GPU dependency.

_____________________________________________________________________________________________________________________________________

Key Features
Real-Time Detection

YOLO-powered model for alphabets (A–Z), digits (0–9), and basic words.

Fast inference using optimized frame processing.

History-based smoothing for stable predictions.

Fingerspelling Mode

Detects multiple sequential alphabets.

Builds words letter-by-letter for practice or learning use.

Word Recognition

Predicts complete gestures mapped to pre-trained sign classes.

Ideal for basic sign vocabulary demonstrations.

Modern Streamlit UI

Clean, responsive layout with webcam preview.

Dark/light theme adaptable styling.

Minimal distractions for focused gesture practice.

Lightweight Deployment

Runs on CPU.

No heavy libraries beyond OpenCV, Streamlit, and Ultralytics.

_____________________________________________________________________________________________________________________________________

Methodology

SignSight follows a structured pipeline based on computer vision, gesture detection, and dynamic result refinement.

Frame Capture
The webcam feed is processed using OpenCV with optimized frame resizing for better inference speed.

Gesture Detection
A YOLO model predicts bounding boxes and gesture classes in real time.

Confidence Filtering
Only predictions above a threshold are considered to ensure reliable output.

History Smoothing
A deque stores recent predictions, and the most frequent label is chosen to stabilize output.

Display Engine
Streamlit renders the real-time camera feed, overlays predictions, and maintains a clean, responsive interface.

This methodology ensures accurate gesture interpretation even under varying lighting, background noise, or hand movement.

_____________________________________________________________________________________________________________________________________

Installation
1. Clone the repository
git clone https://github.com/muskan-1301/Fake_signsight.git
cd Fake_signsight

2. Install dependencies
pip install -r requirements.txt

3. Run the application
streamlit run app.py


_____________________________________________________________________________________________________________________________________

Project Structure
/SignSight
│
├── app.py              # Main Streamlit application
├── model/              # YOLO model files
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation


_____________________________________________________________________________________________________________________________________

Future Improvements

Fingerspelling auto-word formation with spacing logic

ASL full-word dictionary expansion

Multi-hand detection and two-hand gestures

Noise-resistant tracking for low-light conditions

Exportable detection logs for research/education

Mobile app deployment (Flutter + TensorFlow Lite)

_____________________________________________________________________________________________________________________________________

License

This project is for educational purpose only,

