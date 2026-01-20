import cv2
import mediapipe as mp
import numpy as np
import joblib
import math

print("RUNNING REALTIME SCRIPT")

# ----------------------------
# LOAD TRAINED MODEL
# ----------------------------
model = joblib.load("gesture_mlp_model.pkl")

# Gesture label mapping
gesture_labels = {
    1:"Right_1", 2:"Right_2", 3:"Right_3", 4:"Right_4", 5:"Right_5",
    6:"Both_6", 7:"Both_7", 8:"Both_8", 9:"Both_9", 10:"Both_10",
    11:"All_the_best", 12:"Brilliant", 13:"Hungry", 14:"Victory", 15:"I",
    16:"Toilet", 17:"Rock", 18:"Tele_call", 19:"Protest", 20:"Call_me",
    21:"No", 22:"View", 23:"Hold_on", 24:"Quote", 25:"Great"
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ----------------------------
# Safe normalize function (fixed)
# ----------------------------
def normalize_hand(raw):
    # If raw vector missing or invalid -> return empty hand (27 zeros)
    if raw is None or len(raw) == 0:
        return [0] * 27

    # Handle raw vectors of different lengths
    if len(raw) < 48:
        # Pad with zeros if too short
        raw = list(raw) + [0] * (48 - len(raw))

    face = raw[0]
    xs = raw[1:22]       # 21 x-values
    ys = raw[22:43]      # 21 y-values
    fingers = raw[43:48] # 5 finger states

    # Ensure correct sizes
    if len(xs) < 21:
        xs = list(xs) + [0] * (21 - len(xs))
    if len(ys) < 21:
        ys = list(ys) + [0] * (21 - len(ys))
    if len(fingers) < 5:
        fingers = list(fingers) + [0] * (5 - len(fingers))

    # Check if hand is inactive (all zeros after face)
    if sum(xs) == 0 and sum(ys) == 0 and sum(fingers) == 0:
        return [0] * 27

    # Translate so landmark 0 becomes (0,0)
    x0, y0 = xs[0], ys[0]
    xs = [x - x0 for x in xs]
    ys = [y - y0 for y in ys]

    # Create distance list (exactly 21 values)
    rs = []
    for i in range(21):
        rx = xs[i]
        ry = ys[i]
        rs.append((rx * rx + ry * ry) ** 0.5)

    # Normalize distances (avoid division by zero)
    max_r = max(rs) if max(rs) != 0 else 1
    rs = [val / max_r for val in rs]

    # Return exactly 27 values: [face (1) + distances (21) + fingers (5)]
    return [face] + rs + list(fingers)

print("normalize_hand() SAFE VERSION LOADED")

# ----------------------------
# Extract raw 96 features
# ----------------------------
def extract_raw_features(result):
    left = None
    right = None

    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, info in enumerate(result.multi_handedness):
            label = info.classification[0].label  # "Left" or "Right"
            hand = result.multi_hand_landmarks[idx]
            
            if label == "Left":
                left = hand
            else:
                right = hand

    raw = []

    # Process Left Hand
    if left:
        raw.extend(extract_48(left))
    else:
        raw.extend([0]*48)

    # Process Right Hand
    if right:
        raw.extend(extract_48(right))
    else:
        raw.extend([0]*48)

    return np.array(raw)

# 48-feature extraction
def extract_48(hand):
    # hand face
    wrist = hand.landmark[0]
    index_mcp = hand.landmark[5]
    face = 1 if index_mcp.x < wrist.x else -1

    xs = [lm.x for lm in hand.landmark]
    ys = [lm.y for lm in hand.landmark]

    # finger states (simple logic)
    tips = [4, 8, 12, 16, 20]
    fingers = []
    # thumb
    fingers.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)
    # 4 fingers
    for i in range(1, 5):
        fingers.append(1 if hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y else 0)

    return [face] + xs + ys + fingers


# ----------------------------
# REAL-TIME LOOP
# ----------------------------
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Draw hand landmarks
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Extract raw features
        raw = extract_raw_features(result)

        # Normalize to 54 features
        left_raw = raw[:48]
        right_raw = raw[48:]

        left_norm = normalize_hand(left_raw)
        right_norm = normalize_hand(right_raw)

        final_vec = np.array(left_norm + right_norm).reshape(1, -1)

        # Predict
        pred = model.predict(final_vec)[0]
        gesture_name = gesture_labels[pred]

        # Display
        cv2.putText(frame, gesture_name, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Real-Time Gesture Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
