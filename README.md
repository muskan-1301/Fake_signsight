📘 SignSight – Real-Time Sign Language Recognition System

A Streamlit-based real-time ASL (American Sign Language) hand gesture recognition app

🌟 Overview

SignSight is a real-time Sign Language Detection system built using YOLOv8, OpenCV, and Streamlit.
It recognizes Alphabet letters (A–Z) and Digits (0–9) through live webcam feed and supports Fingerspelling, allowing users to build full words using hand gestures.

This project offers a clean UI, high-speed detection, stable predictions, and a smooth user experience.

✨ Key Features
🔤 1. Alphabet Recognition (A–Z)

Detects ASL alphabet signs in real time.

Stabilized predictions using majority voting.

Ideal for fingerspelling words.

🔢 2. Digit Recognition (0–9)

Detects ASL digits.

Optimized models for fast inference.

✋ 3. Fingerspelling System

Builds words letter-by-letter from hand gestures.

Includes manual controls:

🔄 Reset

⌫ Delete

␣ Space

🗑️ Clear

🎥 4. Smooth Webcam Integration

Low-latency camera feed.

Auto stabilization.

Optimized for Windows (DirectShow - CAP_DSHOW).

🎨 5. Professional UI (Streamlit)

Sidebar that includes model selection and controls.

Stats cards for:

Current Prediction

Fingerspelled Word

FPS (Frames per Second)

⚡ 6. YOLOv8 for Real-Time Detection

Fast and accurate detection.

Custom-trained models for high accuracy.

📂 Project Structure
SignSight/
├── app.py                 # Main Streamlit app
├── model/
│   ├── letters.pt         # YOLO model for alphabets
│   └── digits.pt          # YOLO model for digits
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation

🛠️ Tech Stack
Component	Technology
UI	Streamlit
Detection Model	YOLOv8 (Ultralytics)
Backend	Python
Image Processing	OpenCV
Stabilization	Majority Vote Buffer
Deployment	Local / GitHub
🚀 How to Run Locally
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the app
streamlit run app.py

3️⃣ Allow camera access when prompted.
📌 How It Works

User selects Alphabet Model or Digit Model.

The webcam stream starts.

YOLOv8 detects hand gestures in real-time.

For alphabets:

Predictions are stabilized using a sliding window.

The system adds letters into a word (fingerspelling).

UI displays:

Live camera feed

Current prediction

Fingerspelling word

FPS

📊 Fingerspelling Logic

To prevent jitter and wrong characters:

A prediction buffer stores last N predictions.

The system chooses the most frequent (stable) prediction.

Only adds a new character when:

It appears enough times

It is different from the last added one

This ensures:

No duplicate letters

Stable and accurate fingerspelling

Smooth typing-like experience

🧪 Training Details

Trained on ASL datasets for:

Alphabets A–Z

Digits 0–9

YOLOv8n model (optimized for speed)

Augmentations:

Rotation

Brightness

Flip

Hand position variation

📁 Requirements

Your requirements.txt should include:

ultralytics
opencv-python
numpy
streamlit


🎯 Future Improvements

Word-level recognition

Sentence-level translation

Audio output

Support for Indian Sign Language (ISL)

Better mobile-friendly UI

👩‍💻 Contributors

Muskan Dawar – Developer

Model Training Support

UI/UX Implementation

Streamlit Integration

🎉 Thank you for exploring SignSight!

If you like the project, ⭐ the repo on GitHub!