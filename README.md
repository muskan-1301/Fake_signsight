**S I G N S I G H T**


Seamless Interpretation & Gesture Navigation System

SignSight is a real-time Sign Language Recognition system designed to translate hand gestures into English alphabets, digits, and basic words. Built using YOLO, OpenCV, and Streamlit, the project aims to support inclusive communication with a fast, intuitive, and modern UI.

The system is optimized for students, developers, and assistive technology experiments, delivering smooth inference and stable gesture detection.

⸻ **Overview**

SignSight captures live webcam input, detects hand signs using a custom-trained YOLO model, and outputs predictions with smoothing for higher stability.
Features include fingerspelling mode, word detection, and a minimal, accessible interface.

With Streamlit’s responsive UI design and lightweight model loading, SignSight is ideal for demos, academic projects, and rapid experimentation.

⸻ **Key Features**

**🎯 Real-Time Gesture Detection**

● YOLO-based sign recognition

● Supports A–Z alphabets, 0–9 digits, and basic words

● High-confidence filtering for accuracy



**🔤 Fingerspelling Mode**

● Predicts letters sequentially

● Builds words character-by-character

● Ideal for ASL practice and learning



**🧠 Stable Prediction Engine**

● Uses a rolling history (deque)

● Outputs the most consistent result for stable detection



**🎨 Modern Streamlit UI**

● Clean, intuitive layout

●Live camera feed preview

● Responsive and minimal design



**⚡ Efficient & Lightweight**

● Runs smoothly on CPU

● Supports manual or automatic model loading




⸻ **Methodology**

1. Frame Capture

Input frames are processed via OpenCV with optimized resizing.

2. YOLO-Based Detection

Each frame is passed through the trained YOLO model for class predictions.

3. Confidence Thresholding

Low-confidence predictions are filtered out.

4. History-Based Smoothing

A rolling buffer stores past predictions to improve stability.

5. Streamlit Rendering

The UI displays the live feed and predictions in real time.

This workflow ensures balance between speed, accuracy, and stability.

⸻ **Model Downloads**

Your YOLO models are stored in Google Drive due to size limits.

📦 Letters Model (256 MB)

Direct Download:
👉 https://drive.google.com/uc?export=download&id=1IvBFgoHSmMqUC8qWTFfah7T0HMyvhQck

🔢 Digits Model (85 MB)

Direct Download:
👉 https://drive.google.com/uc?export=download&id=1XhB9wbBni09N90GHhAUFUBWJfFh2H2zO

⚠️ Place downloaded models in:
```
project/
 └── model/
      ├── letters.pt
      └── digits.pt
```

⸻ Installation
1.  Clone the Repository
```
git clone https://github.com/muskan-1301/Fake_signsight.git
cd Fake_signsight
```

2.  Install Dependencies
```
pip install -r requirements.txt
```

3.  Run the Application
```
streamlit run app.py
```

⸻ Project Structure
```
SignSight/
│
├──test.py                 # run locally
├── app.py                 # Main application file
├── model/                 # YOLO model files (download & place here)
│     ├── letters.pt
│     └── digits.pt
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

⸻ **Future Improvements**

● Two-hand gesture support

● Larger ASL word vocabulary

● Auto-spacing for fingerspelling

● Noise-resistant low-light tracking

● TensorFlow Lite mobile version

● Speech output integration




**NOTE:**

This project was created for academic/educational purposes only.
