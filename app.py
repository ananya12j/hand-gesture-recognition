import streamlit as st
import cv2
import numpy as np
import pyautogui
from tensorflow.keras.models import load_model
import time

# Load the trained model
model = load_model('gesturefinal.h5')

# List of gestures (ensure they match the order in your training data)
gestures = ['scroll_up', 'scroll_down', 'back', 'forward', 'screenshot', 'close_window', 'openapp', 'none']

# Define the confidence threshold and action cooldown settings
threshold = 0.7
action_cooldown = 1  # seconds between actions
last_action_time = 0

# Streamlit UI setup
st.set_page_config(page_title="Gesture Recognition Control", page_icon="🖐", layout="wide")
st.title("🤖 Real-time Gesture Recognition Control")

# Sidebar setup for customization options
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
action_cooldown = st.sidebar.slider("Action Cooldown (seconds)", 0.5, 5.0, 1.0, 0.5)

# Placeholder for displaying the video
st.markdown(
    """
    <style>
        .main { background-color: #f0f2f6; }
        h1 { color: #4A90E2; }
        .stButton>button { background-color: #4CAF50; color: white; font-size: 16px; }
    </style>
    """, unsafe_allow_html=True
)
st.write("### Gesture-based System Controls")

frame_placeholder = st.empty()
status_placeholder = st.sidebar.empty()

# Function to perform action based on recognized gesture
def perform_action(gesture):
    global last_action_time
    current_time = time.time()
    if current_time - last_action_time > action_cooldown:
        action_message = ""
        
        if gesture == 'scroll_up':
            pyautogui.scroll(500)
            action_message = "Scrolled up"
            
        elif gesture == 'scroll_down':
            pyautogui.scroll(-500)
            action_message = "Scrolled down"
            
        elif gesture == 'back':
            pyautogui.hotkey('alt', 'left')
            action_message = "Back action triggered"
            
        elif gesture == 'forward':
            pyautogui.hotkey('alt', 'right')
            action_message = "Forward action triggered"
            
        elif gesture == 'screenshot':
            time.sleep(2)
            screenshot = pyautogui.screenshot()
            screenshot.save("screenshot.png")
            action_message = "Screenshot taken"
            st.image(screenshot, caption="Screenshot")

        elif gesture == 'close_window':
            pyautogui.hotkey('winleft', 'down')
            action_message = "Window closed"
            
        elif gesture == 'openapp':
            pyautogui.hotkey('winleft', 'r')
            pyautogui.write('notepad')
            pyautogui.press('enter')
            action_message = "Opened Notepad"
        
        # Display action feedback
        status_placeholder.success(action_message)
        last_action_time = current_time

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    # Flip and crop the frame for gesture detection
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rect_x, rect_y, rect_w, rect_h = width // 4, height // 4, width // 2, height // 2
    cropped_frame = frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]

    # Preprocess the cropped frame for model prediction
    cropped_frame_resized = cv2.resize(cropped_frame, (64, 64))
    cropped_frame_normalized = cropped_frame_resized / 255.0
    cropped_frame_input = np.expand_dims(cropped_frame_normalized, axis=0)

    # Predict gesture
    prediction = model.predict(cropped_frame_input)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction, axis=1)

    # Determine the gesture and perform action if confidence is high enough
    if confidence > threshold:
        predicted_gesture = gestures[predicted_class[0]]
        perform_action(predicted_gesture)
    else:
        predicted_gesture = 'none'
    
    # Display prediction and confidence on the frame
    cv2.putText(frame, f"Predicted: {predicted_gesture} ({confidence*100:.2f}%)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw the rectangular box for the ROI
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)

    # Display the frame in Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

# Release the webcam
cap.release()