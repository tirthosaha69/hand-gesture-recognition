import cv2
import mediapipe as mp
import numpy as np
import os
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# Detect if a finger is open using landmark positions
# -------------------------------
def finger_states(hand_landmarks):
    # Tip indices for fingers
    tip_ids = [4, 8, 12, 16, 20]

    states = []

    # thumb
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        states.append(1)
    else:
        states.append(0)

    # other 4 fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            states.append(1)
        else:
            states.append(0)

    return states  # list of 5 values (0 or 1)

# -------------------------------
# Determine hand orientation (front/back)
# Paper uses N/F/B, here we map:
# N = 0 (not present)
# F = 1 (front)
# B = -1 (back)
# -------------------------------
def hand_face(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]

    if index_mcp.x < wrist.x:
        return 1   # front
    else:
        return -1  # back


# -------------------------------
# Extract 48 features per hand
# -------------------------------
def extract_hand_features(hand_landmarks):
    features = []

    # Hand face
    features.append(hand_face(hand_landmarks))

    # 21 landmarks (x, y)
    for lm in hand_landmarks.landmark:
        features.append(lm.x)
    for lm in hand_landmarks.landmark:
        features.append(lm.y)

    # Finger states
    features.extend(finger_states(hand_landmarks))

    return features   # 48 values


# -------------------------------
# MAIN DATA COLLECTION
# -------------------------------
participant_id = input("Enter participant number (1-6): ")
gesture_id = input("Enter gesture ID (1-25): ")

save_dir = f"dataset/Raw/participant_{participant_id}/gesture_{gesture_id}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

sample = 0
MAX_SAMPLES = 100  # paper uses 100 variants per gesture

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        left_hand = None
        right_hand = None

        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, classification in enumerate(result.multi_handedness):
                label = classification.classification[0].label  # Left or Right
                hand_landmarks = result.multi_hand_landmarks[idx]

                if label == "Left":
                    left_hand = hand_landmarks
                else:
                    right_hand = hand_landmarks

        # Build 96-feature vector
        raw_vector = []

        # Left hand (48)
        if left_hand:
            raw_vector.extend(extract_hand_features(left_hand))
        else:
            raw_vector.extend([0] * 48)  # N state

        # Right hand (48)
        if right_hand:
            raw_vector.extend(extract_hand_features(right_hand))
        else:
            raw_vector.extend([0] * 48)

        raw_vector = np.array(raw_vector)

        # Save sample
        np.savetxt(f"{save_dir}/{sample}.txt", raw_vector)

        sample += 1
        print(f"Collected sample {sample}/100")

        # Draw for preview
        if left_hand:
            mp_draw.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS)
        if right_hand:
            mp_draw.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or sample >= MAX_SAMPLES:
            break

cap.release()
cv2.destroyAllWindows()