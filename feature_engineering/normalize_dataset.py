import numpy as np
import os

# Convert (x, y) to distance r
def xy_to_r(x, y):
    return np.sqrt(x*x + y*y)

# Normalize a single hand (48 raw features → 27 normalized)
def normalize_hand(raw):
    # If the raw vector is not exactly 48 values → treat as no hand
    if len(raw) != 48 or sum(raw) == 0:
        return [0] * 27

    face = raw[0]
    xs = raw[1:22]
    ys = raw[22:43]
    fingers = raw[43:48]

    # Safety check: ensure xs=21, ys=21, fingers=5
    if len(xs) != 21 or len(ys) != 21 or len(fingers) != 5:
        return [0] * 27

    # Translate so landmark 0 becomes origin
    x0, y0 = xs[0], ys[0]
    xs = [x - x0 for x in xs]
    ys = [y - y0 for y in ys]

    # Convert to r only
    rs = [xy_to_r(xs[i], ys[i]) for i in range(21)]

    max_r = max(rs) if max(rs) != 0 else 1
    rs = [r / max_r for r in rs]

    return [face] + rs + fingers
  # 54 features

def normalize_sample(raw_vector):
    raw_vector = list(raw_vector)

    # Must be exactly 96
    if len(raw_vector) != 96:
        return np.array([0] * 54)

    left_raw = raw_vector[:48]
    right_raw = raw_vector[48:]

    left_norm = normalize_hand(left_raw)
    right_norm = normalize_hand(right_raw)

    return np.array(left_norm + right_norm)

# -----------------------------
# PROCESS ALL SAMPLES
# -----------------------------
dataset_dir = "dataset/participant_2"
output_dir = "normalized/participant_2"
os.makedirs(output_dir, exist_ok=True)

for gesture_dir in os.listdir(dataset_dir):
    gpath = os.path.join(dataset_dir, gesture_dir)
    out_gpath = os.path.join(output_dir, gesture_dir)
    os.makedirs(out_gpath, exist_ok=True)

    for file in os.listdir(gpath):
        raw = np.loadtxt(os.path.join(gpath, file))
        normalized = normalize_sample(raw)
        np.savetxt(os.path.join(out_gpath, file), normalized)
        print("Normalized:", gesture_dir, file)
