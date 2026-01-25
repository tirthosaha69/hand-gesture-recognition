import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# LOAD DATA FROM ALL PARTICIPANTS
# =========================================================
def load_all_participants_data(normalized_root):
    X, y = [], []

    for participant in sorted(os.listdir(normalized_root)):
        participant_path = os.path.join(normalized_root, participant)
        if not os.path.isdir(participant_path):
            continue

        print(f"[INFO] Loading data from {participant}")

        for g in range(1, 26):
            gesture_dir = os.path.join(participant_path, f"gesture_{g}")
            if not os.path.exists(gesture_dir):
                continue

            for file in os.listdir(gesture_dir):
                vec = np.loadtxt(os.path.join(gesture_dir, file))
                if vec.shape[0] == 54:
                    X.append(vec)
                    y.append(g)

    return np.array(X), np.array(y)


# =========================================================
# LOAD DATASET
# =========================================================
X, y = load_all_participants_data("dataset/Normalized")

# =========================================================
# TRAIN–TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# BUILD MODEL
# =========================================================
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================
print("[INFO] Training model...")
model.fit(X_train, y_train)

# =========================================================
# SAVE MODEL
# =========================================================
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/gesture_mlp_model.pkl")

# =========================================================
# PREDICTIONS
# =========================================================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# =========================================================
# METRICS
# =========================================================
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

precision_macro = precision_score(y_test, y_test_pred, average="macro")
recall_macro = recall_score(y_test, y_test_pred, average="macro")
f1_macro = f1_score(y_test, y_test_pred, average="macro")

precision_micro = precision_score(y_test, y_test_pred, average="micro")
recall_micro = recall_score(y_test, y_test_pred, average="micro")
f1_micro = f1_score(y_test, y_test_pred, average="micro")

precision_weighted = precision_score(y_test, y_test_pred, average="weighted")
recall_weighted = recall_score(y_test, y_test_pred, average="weighted")
f1_weighted = f1_score(y_test, y_test_pred, average="weighted")

# =========================================================
# SAVE EVALUATION RESULTS
# =========================================================
os.makedirs("evaluation_results", exist_ok=True)

# ---- Accuracy ----
with open("evaluation_results/accuracy.txt", "w") as f:
    f.write(f"Training Accuracy: {train_acc:.4f}\n")
    f.write(f"Testing Accuracy : {test_acc:.4f}\n")

# ---- Classification Report ----
with open("evaluation_results/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_test_pred))

# ---- Extra Metrics ----
with open("evaluation_results/metrics_summary.txt", "w") as f:
    f.write("=== Macro Average ===\n")
    f.write(f"Precision: {precision_macro:.4f}\n")
    f.write(f"Recall   : {recall_macro:.4f}\n")
    f.write(f"F1-score : {f1_macro:.4f}\n\n")

    f.write("=== Micro Average ===\n")
    f.write(f"Precision: {precision_micro:.4f}\n")
    f.write(f"Recall   : {recall_micro:.4f}\n")
    f.write(f"F1-score : {f1_micro:.4f}\n\n")

    f.write("=== Weighted Average ===\n")
    f.write(f"Precision: {precision_weighted:.4f}\n")
    f.write(f"Recall   : {recall_weighted:.4f}\n")
    f.write(f"F1-score : {f1_weighted:.4f}\n")

# =========================================================
# CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(y_test, y_test_pred)

np.savetxt("evaluation_results/confusion_matrix.txt", cm, fmt="%d")

plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap='YlOrRd', aspect='auto')
plt.title("Confusion Matrix (Test Data)", fontsize=16, fontweight='bold')
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.colorbar(label='Count')

# Add text annotations on each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        # Use white text for dark cells, black for light cells
        text_color = 'white' if value > cm.max() / 2 else 'black'
        plt.text(j, i, str(value), ha='center', va='center', 
                color=text_color, fontsize=8, fontweight='bold')

plt.xticks(range(25), range(1, 26), fontsize=9)
plt.yticks(range(25), range(1, 26), fontsize=9)
plt.tight_layout()
plt.savefig("evaluation_results/confusion_matrix.png", dpi=150)
plt.close()

print("\n[INFO] Evaluation complete.")
print("[INFO] All metrics saved in 'evaluation_results/' folder.")
