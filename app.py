import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import time
import threading
import os

st.set_page_config(
    page_title="SignSight",
    page_icon="sign",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #F5F7FA; color: #2C3E50; }
    .block-container { padding-top: 3rem; padding-bottom: 1rem; max-width: 1600px; }
    header[data-testid="stHeader"] { display: none; }
    .main-header { text-align: center; padding: 2rem 0 1rem 0; margin-bottom: 1.5rem; }
    .main-title { font-size: 3.5rem; font-weight: 800; color: #2C3E50; margin-bottom: 0.5rem; }
    .main-subtitle { font-size: 1.1rem; color: #5D6D7E; font-weight: 500; }
    [data-testid="stSidebar"] { background-color: #F0F2F6; border-right: 1px solid #E0E4E8; padding-top: 2rem; }
    [data-testid="stSidebar"] .element-container { color: #2C3E50; }
    [data-testid="stSidebar"] * { color: #2C3E50 !important; }
    .sidebar-header { font-size: 1.1rem; font-weight: 700; color: #34495E !important; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
    .model-select-container { background-color: #FFFFFF; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; border: 1px solid #D5DBDB; }
    .model-label { font-size: 0.9rem; color: #34495E !important; margin-bottom: 0.5rem; font-weight: 700; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 3px 10px rgba(74, 144, 226, 0.25);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #357ABD 0%, #2A5F94 100%);
        box-shadow: 0 5px 15px rgba(74, 144, 226, 0.35);
        transform: translateY(-2px);
    }
    .video-frame { border-radius: 15px; overflow: hidden; border: 2px solid #D5DBDB; background: #FFFFFF; }
    .video-placeholder { background: #F8F9FA; border-radius: 15px; padding: 4rem 2rem; text-align: center; border: 2px dashed #B0BEC5; min-height: 500px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .stats-card { background: #FFFFFF; border-radius: 15px; padding: 1.5rem; border: 2px solid #E0E4E8; margin-bottom: 1rem; text-align: center; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); }
    .stat-label { font-size: 0.85rem; color: #34495E; font-weight: 700; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px; }
    .stat-value { font-size: 2.2rem; font-weight: 800; color: #1A2332; }
    .stat-value-word {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1A2332;
        min-height: 2.5rem;
        max-height: 4.8rem;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
        word-break: break-word;
        white-space: normal;
        align-items: center;
        justify-content: center;
    }
    .stRadio > label { color: #34495E !important; font-size: 0.95rem !important; font-weight: 700 !important; }
    .stRadio > div { background-color: transparent; padding: 0; }
    .stRadio label[data-baseweb="radio"] { color: #2C3E50 !important; }
    .stRadio label[data-baseweb="radio"] span { color: #2C3E50 !important; }
    .success-message { background-color: #27AE60; color: white; padding: 0.75rem; border-radius: 8px; text-align: center; font-weight: 600; margin-top: 1rem; }
    .fingerspell-controls { background: #FFFFFF; border-radius: 12px; padding: 1rem; margin-top: 1rem; border: 1px solid #D5DBDB; }
    .control-label { font-size: 0.9rem; color: #34495E !important; font-weight: 700; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .top-bar { position: fixed; top: 0; right: 0; padding: 1rem 2rem; z-index: 999; }
    .deploy-btn { background: #2C3E50; color: white; padding: 0.5rem 1.5rem; border-radius: 8px; font-weight: 600; font-size: 0.9rem; border: none; cursor: pointer; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    [data-testid="stImage"] { border-radius: 12px; overflow: hidden; }
    hr { border: none; border-top: 1px solid #D5DBDB; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'prediction_buffer' not in st.session_state:
    st.session_state.prediction_buffer = deque(maxlen=15)
if 'last_stable_prediction' not in st.session_state:
    st.session_state.last_stable_prediction = ""
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'show_reset_message' not in st.session_state:
    st.session_state.show_reset_message = False
if 'camera_index' not in st.session_state:
    st.session_state.camera_index = 0
if 'cam' not in st.session_state:
    st.session_state.cam = None

st.markdown("""
<div class="main-header">
    <div class="main-title">SignSight</div>
    <div class="main-subtitle">Real-Time Sign Language Recognition with Fingerspelling</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="model-select-container">', unsafe_allow_html=True)
    st.markdown('<div class="model-label">Select Model</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Model",
        ["Alphabet Model (A–Z)", "Digit Model (0–9)"],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="model-label" style="margin-bottom: 1rem;">Camera</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_clicked = st.button("▶️ Start", key="start_btn", width="stretch")
        if start_clicked:
            st.session_state.prediction_buffer = deque(maxlen=9)
            st.session_state.webcam_running = True
            st.session_state.last_stable_prediction = ""
            st.session_state.show_reset_message = False
            time.sleep(0.05)
    with col2:
        stop_clicked = st.button("⏹️ Stop", key="stop_btn", width="stretch")
        if stop_clicked:
            st.session_state.webcam_running = False
    st.markdown("---")

    if "Alphabet" in model_choice:
        st.markdown('<div class="fingerspell-controls">', unsafe_allow_html=True)
        st.markdown('<div class="control-label">Fingerspelling Controls</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", key="reset_btn", width="stretch"):
                st.session_state.current_word = ""
                st.session_state.prediction_buffer.clear()
                st.session_state.last_stable_prediction = ""
                st.session_state.show_reset_message = True

        with col2:
            if st.button("⌫ Delete", key="delete_btn", width="stretch"):
                if st.session_state.current_word:
                    st.session_state.current_word = st.session_state.current_word[:-1]
                    st.session_state.last_stable_prediction = ""
                    st.session_state.prediction_buffer.clear()

        col3, col4 = st.columns(2)
        with col3:
            if st.button("␣ Space", key="space_btn", width="stretch"):
                st.session_state.current_word += " "
                st.session_state.prediction_buffer.clear()
                st.session_state.last_stable_prediction = ""
        with col4:
            if st.button("🗑️ Clear", key="clear_btn", width="stretch"):
                st.session_state.current_word = ""
                st.session_state.show_reset_message = True
                st.session_state.last_stable_prediction = ""
                st.session_state.prediction_buffer.clear()

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.show_reset_message:
            st.markdown('<div class="success-message">Word reset!</div>', unsafe_allow_html=True)
            time.sleep(0.5)
            st.session_state.show_reset_message = False

class CameraStream:
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.frame = None
        self.stopped = True
        self.lock = threading.Lock()

    def start(self):
        if not self.stopped:
            return
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
        # release in stop()

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.cap is not None and self.cap.isOpened():
            try:
                self.cap.release()
            except:
                pass

def start_camera():
    if st.session_state.cam is None:
        st.session_state.cam = CameraStream(src=st.session_state.camera_index)
        st.session_state.cam.start()
    else:
        if st.session_state.cam.src != st.session_state.camera_index:
            try:
                st.session_state.cam.stop()
            finally:
                st.session_state.cam = CameraStream(src=st.session_state.camera_index)
                st.session_state.cam.start()

def stop_camera():
    if st.session_state.cam is not None:
        try:
            st.session_state.cam.stop()
        finally:
            st.session_state.cam = None

@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_path = "model/letters.pt" if "Alphabet" in model_choice else "model/digits.pt"
model_type = "alphabet" if "Alphabet" in model_choice else "digit"

col_left, col_right = st.columns([2.5, 1])
with col_left:
    frame_placeholder = st.empty()
with col_right:
    prediction_placeholder = st.empty()
    if model_type == "alphabet":
        word_placeholder = st.empty()
    fps_placeholder = st.empty()

if st.session_state.webcam_running:
    model = load_model(model_path)
    if model is None:
        st.error(f"Failed to load model from {model_path}. Please check the file path.")
        st.session_state.webcam_running = False
    else:
        start_camera()
        warmup_start = time.time()
        while time.time() - warmup_start < 0.3:
            _ = st.session_state.cam.read()
            time.sleep(0.02)

        prev_time = time.time()
        current_prediction = "None"
        imgsz = 320
        half = True

        try:
            while st.session_state.webcam_running:
                frame = st.session_state.cam.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                now = time.time()
                dt = now - prev_time if (now - prev_time) > 0 else 1e-6
                prev_time = now
                fps = 1.0 / dt

                try:
                    results = model(frame, conf=0.45, verbose=False, imgsz=imgsz, half=half)
                except TypeError:
                    results = model(frame, conf=0.45, verbose=False)
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.session_state.webcam_running = False
                    break

                if len(results) > 0 and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = model.names[cls] if cls in model.names else str(cls)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (226, 144, 74), 3)
                        label_text = f"{label} {conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (226, 144, 74), -1)
                        cv2.putText(frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        current_prediction = label

                        if model_type == "alphabet":
                            st.session_state.prediction_buffer.append(label)
                            if len(st.session_state.prediction_buffer) >= 7:
                                counter = Counter(st.session_state.prediction_buffer)
                                stable_prediction, count = counter.most_common(1)[0]
                                if count >= 7 and stable_prediction != st.session_state.last_stable_prediction:
                                    st.session_state.current_word += stable_prediction
                                    st.session_state.last_stable_prediction = stable_prediction
                else:
                    current_prediction = "None"

                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (74, 144, 226), 2)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", width="stretch")

                prediction_placeholder.markdown(f"""
                <div class="stats-card">
                    <div class="stat-label">Current Prediction</div>
                    <div class="stat-value">{current_prediction}</div>
                </div>
                """, unsafe_allow_html=True)

                if model_type == "alphabet":
                    word_display = st.session_state.current_word if st.session_state.current_word else "--"
                    word_placeholder.markdown(f"""
                    <div class="stats-card">
                        <div class="stat-label">Fingerspelled Word</div>
                        <div class="stat-value-word">{word_display}</div>
                    </div>
                    """, unsafe_allow_html=True)

                fps_placeholder.markdown(f"""
                <div class="stats-card">
                    <div class="stat-label">FPS</div>
                    <div class="stat-value">{fps:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

                time.sleep(0.01)

        finally:
            stop_camera()
else:
    frame_placeholder.markdown("""
    <div class="video-placeholder">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📷</div>
        <div style="font-size: 1.3rem; color: #2C3E50; font-weight: 600; margin-bottom: 0.5rem;">
            Click "Start" to begin webcam feed
        </div>
        <div style="font-size: 0.95rem; color: #5D6D7E;">
            Make sure your camera is connected and permissions are granted
        </div>
    </div>
    """, unsafe_allow_html=True)

    prediction_placeholder.markdown("""
    <div class="stats-card">
        <div class="stat-label">Current Prediction</div>
        <div class="stat-value">--</div>
    </div>
    """, unsafe_allow_html=True)

    if model_type == "alphabet":
        word_placeholder.markdown("""
        <div class="stats-card">
            <div class="stat-label">Fingerspelled Word</div>
            <div class="stat-value-word">--</div>
        </div>
        """, unsafe_allow_html=True)

    fps_placeholder.markdown("""
    <div class="stats-card">
        <div class="stat-label">FPS</div>
        <div class="stat-value">--</div>
    </div>
    """, unsafe_allow_html=True)
