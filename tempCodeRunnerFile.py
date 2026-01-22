from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
from datetime import datetime
from collections import deque, Counter
import threading
import time

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load("model/gesture_mlp_model.pkl")

# Gesture label mapping
gesture_labels = {
    1:"Right_1", 2:"Right_2", 3:"Right_3", 4:"Right_4", 5:"Right_5",
    6:"Both_6", 7:"Both_7", 8:"Both_8", 9:"Both_9", 10:"Both_10",
    11:"All_the_best", 12:"Brilliant", 13:"Hungry", 14:"Victory", 15:"I",
    16:"Toilet", 17:"Rock", 18:"Tele_call", 19:"Protest", 20:"Call_me",
    21:"No", 22:"View", 23:"Hold_on", 24:"Quote", 25:"Great"
}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Global variables for analytics
gesture_history = deque(maxlen=100)
gesture_counter = Counter()
current_gesture = "No gesture detected"
confidence_score = 0.0
fps = 0
detection_active = True
lock = threading.Lock()

def normalize_hand(raw):
    if raw is None or len(raw) == 0:
        return [0] * 27
    
    if len(raw) < 48:
        raw = list(raw) + [0] * (48 - len(raw))
    
    face = raw[0]
    xs = raw[1:22]
    ys = raw[22:43]
    fingers = raw[43:48]
    
    if len(xs) < 21:
        xs = list(xs) + [0] * (21 - len(xs))
    if len(ys) < 21:
        ys = list(ys) + [0] * (21 - len(ys))
    if len(fingers) < 5:
        fingers = list(fingers) + [0] * (5 - len(fingers))
    
    if sum(xs) == 0 and sum(ys) == 0 and sum(fingers) == 0:
        return [0] * 27
    
    x0, y0 = xs[0], ys[0]
    xs = [x - x0 for x in xs]
    ys = [y - y0 for y in ys]
    
    rs = []
    for i in range(21):
        rx = xs[i]
        ry = ys[i]
        rs.append((rx * rx + ry * ry) ** 0.5)
    
    max_r = max(rs) if max(rs) != 0 else 1
    rs = [val / max_r for val in rs]
    
    return [face] + rs + list(fingers)

def extract_raw_features(result):
    left = None
    right = None
    
    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, info in enumerate(result.multi_handedness):
            label = info.classification[0].label
            hand = result.multi_hand_landmarks[idx]
            
            if label == "Left":
                left = hand
            else:
                right = hand
    
    raw = []
    
    if left:
        raw.extend(extract_48(left))
    else:
        raw.extend([0]*48)
    
    if right:
        raw.extend(extract_48(right))
    else:
        raw.extend([0]*48)
    
    return np.array(raw)

def extract_48(hand):
    wrist = hand.landmark[0]
    index_mcp = hand.landmark[5]
    face = 1 if index_mcp.x < wrist.x else -1
    
    xs = [lm.x for lm in hand.landmark]
    ys = [lm.y for lm in hand.landmark]
    
    tips = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)
    for i in range(1, 5):
        fingers.append(1 if hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y else 0)
    
    return [face] + xs + ys + fingers

def generate_frames():
    global current_gesture, confidence_score, fps, detection_active
    
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
        while True:
            if not detection_active:
                time.sleep(0.1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            # Draw hand landmarks
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            
            # Extract and predict
            raw = extract_raw_features(result)
            left_raw = raw[:48]
            right_raw = raw[48:]
            
            left_norm = normalize_hand(left_raw)
            right_norm = normalize_hand(right_raw)
            
            final_vec = np.array(left_norm + right_norm).reshape(1, -1)
            
            # Predict
            pred = model.predict(final_vec)[0]
            
            # Get prediction probabilities if available
            try:
                proba = model.predict_proba(final_vec)[0]
                confidence_score = float(max(proba) * 100)
            except:
                confidence_score = 95.0
            
            gesture_name = gesture_labels[pred]
            
            with lock:
                current_gesture = gesture_name
                gesture_history.append({
                    'gesture': gesture_name,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': confidence_score
                })
                gesture_counter[gesture_name] += 1
            
            # Draw UI elements on frame
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence_score:.1f}%", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/current_gesture')
def get_current_gesture():
    with lock:
        return jsonify({
            'gesture': current_gesture,
            'confidence': confidence_score,
            'fps': fps,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/gesture_history')
def get_gesture_history():
    with lock:
        return jsonify(list(gesture_history))

@app.route('/api/gesture_stats')
def get_gesture_stats():
    with lock:
        return jsonify({
            'total_detections': sum(gesture_counter.values()),
            'unique_gestures': len(gesture_counter),
            'top_gestures': dict(gesture_counter.most_common(5))
        })

@app.route('/api/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    return jsonify({'active': detection_active})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    with lock:
        gesture_history.clear()
        gesture_counter.clear()
    return jsonify({'status': 'success'})

@app.route('/api/gesture_list')
def get_gesture_list():
    return jsonify(gesture_labels)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)