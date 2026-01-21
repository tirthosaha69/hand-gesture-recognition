import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# ------------------------------
# LOAD NORMALIZED DATASET
# ------------------------------
def load_participant_data(path):
    X = []
    y = []
    
    for g in range(1, 26):  # gesture IDs 1–25
        gesture_dir = os.path.join(path, f"gesture_{g}")
        for file in os.listdir(gesture_dir):
            vec = np.loadtxt(os.path.join(gesture_dir, file))
            X.append(vec)
            y.append(g)  # label is gesture number
            
    return np.array(X), np.array(y)


print("Loading Participant 1 training data...")
X_train, y_train = load_participant_data("normalized/participant_1")

print("Dataset Loaded:")
print("X shape =", X_train.shape)
print("y shape =", y_train.shape)

# ------------------------------
# BUILD MLP CLASSIFIER
# EXACT SETTINGS FROM PAPER
# ------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(54, 54),  # 2 hidden layers
    activation='relu',
    solver='sgd',
    alpha=1e-5,  # L2 regularization
    max_iter=500,
    learning_rate_init=0.001,
    shuffle=True
)

print("\nTraining MLP model...")
model.fit(X_train, y_train)
print("Training completed!")

# ------------------------------
# SAVE MODEL
# ------------------------------
joblib.dump(model, "model/gesture_mlp_model.pkl")
print("\nModel saved as gesture_mlp_model.pkl")

# ------------------------------
# OPTIONAL: SELF-ACCURACY CHECK
# ------------------------------
pred = model.predict(X_train)
acc = accuracy_score(y_train, pred)
print("\nTraining Accuracy:", acc)
