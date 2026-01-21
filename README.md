# ✋ Real-Time Hand Gesture Recognition using Machine Learning

A machine learning–based real-time hand gesture recognition system that detects and classifies hand gestures using webcam input. The project follows the **complete ML lifecycle**—from data collection to deployment—and supports **multi-user, participant-independent gesture recognition**.

---

## 📌 Project Overview

Human–Computer Interaction (HCI) aims to make interaction with machines more natural and intuitive. Traditional input devices such as keyboards and mice are limited in expressiveness.
This project proposes a **vision-based hand gesture recognition system** that enables contactless interaction using hand gestures captured through a standard webcam.

The system uses:

* **MediaPipe** for hand landmark detection
* **Feature engineering & normalization** for robustness
* **Multi-Layer Perceptron (MLP)** for gesture classification
* **Real-time inference** for deployment

---

## 🎯 Objectives

* Collect hand gesture data from multiple participants
* Extract meaningful hand landmark–based features
* Normalize features for scale and translation invariance
* Train a supervised ML model for gesture classification
* Evaluate the model using standard ML metrics
* Deploy the trained model for real-time gesture prediction

---

## 🧠 Machine Learning Lifecycle Followed

1. **Data Collection** – Webcam-based gesture capture
2. **Feature Engineering & Preprocessing** – Landmark normalization
3. **Dataset Preparation** – Multi-participant labeled dataset
4. **Model Selection** – Multi-Layer Perceptron (MLP)
5. **Model Training** – Supervised learning
6. **Model Evaluation** – Accuracy, Precision, Recall, F1-score, Confusion Matrix
7. **Deployment** – Real-time gesture recognition

---

## 🏗️ System Architecture (High-Level)

```
Webcam Input
     ↓
Hand Detection (MediaPipe)
     ↓
Feature Extraction (Landmarks)
     ↓
Feature Normalization
     ↓
MLP Classifier
     ↓
Gesture Prediction (Real-Time Output)
```

---

## 🛠️ Tools & Technologies

| Component               | Technology                   |
| ----------------------- | ---------------------------- |
| Programming Language    | Python                       |
| Computer Vision         | OpenCV                       |
| Hand Landmark Detection | MediaPipe                    |
| Machine Learning        | Scikit-learn                 |
| Model                   | Multi-Layer Perceptron (MLP) |
| Visualization           | Matplotlib                   |
| Deployment              | Real-time webcam inference   |

---

## 📂 Project Structure

```
PROJECT/
│
├── data_collection/        # Dataset collection scripts
├── dataset/                # Raw gesture data
├── feature_engineering/    # Normalization & preprocessing
├── normalized/             # Processed dataset
├── model_training/         # Training & evaluation scripts
├── model/                  # Saved trained model (.pkl)
├── inference/              # Real-time prediction script
├── evaluation_results/     # Accuracy, metrics, confusion matrix
└── README.md
```

---

## 📊 Dataset Details

* **Gestures:** 25 static hand gestures
* **Participants:** Multiple users
* **Hands Supported:** Single-hand & two-hand gestures
* **Raw Features:** 96 per sample (48 per hand)
* **Normalized Features:** 54 per sample

### Raw Feature Breakdown (per hand)

* Hand orientation (front/back)
* 21 landmark x-coordinates
* 21 landmark y-coordinates
* 5 finger state indicators (open/closed)

---

## ⚙️ Feature Engineering & Preprocessing

To improve generalization and robustness, the following preprocessing steps are applied:

1. Translate landmarks so the wrist becomes the origin
2. Convert (x, y) coordinates into radial distances
3. Normalize distances using the maximum radius
4. Preserve finger states and hand orientation

**Result:**

```
96 raw features → 54 normalized features
```

### Benefits

* Scale invariance
* Translation invariance
* Reduced noise
* Improved model generalization

---

## 🤖 Model Details

### Model Chosen

**Multi-Layer Perceptron (MLP)**

### Architecture

| Layer          | Description        |
| -------------- | ------------------ |
| Input Layer    | 54 neurons         |
| Hidden Layer 1 | 128 neurons (ReLU) |
| Hidden Layer 2 | 64 neurons (ReLU)  |
| Output Layer   | 25 gesture classes |
| Optimizer      | Adam               |
| Regularization | L2                 |
| Early Stopping | Enabled            |

---

## 🏋️ Model Training

* **Train–Test Split:** 80% training, 20% testing
* **Stratified sampling** to maintain class balance
* **Early stopping** to prevent overfitting

---

## 📈 Model Evaluation

### Accuracy

* **Training Accuracy:** **84.40%**
* **Testing Accuracy:** **83.40%**

The small gap between training and testing accuracy indicates **good generalization**.

---

### Precision, Recall & F1-Score

**Macro Average**

* Precision: **0.8591**
* Recall: **0.8340**
* F1-score: **0.8317**

**Micro Average**

* Precision: **0.8340**
* Recall: **0.8340**
* F1-score: **0.8340**

**Weighted Average**

* Precision: **0.8591**
* Recall: **0.8340**
* F1-score: **0.8317**

**Interpretation**

* Macro scores show balanced performance across all gesture classes
* Micro scores reflect overall classification performance
* Weighted scores confirm robustness despite class variations

---

### Confusion Matrix

* Strong diagonal pattern → correct classification for most gestures
* Minor confusion between visually similar gestures (expected in vision-based systems)

---

## 🎥 Real-Time Deployment

### Deployment Flow

* Webcam captures live frames
* MediaPipe extracts hand landmarks
* Features normalized in real time
* Trained MLP model predicts gesture instantly

### Output

* Gesture name displayed on screen
* Low-latency real-time performance

---

## ✅ Results Summary

* Successfully recognized **25 hand gestures**
* Real-time gesture prediction achieved
* Stable performance across multiple users
* Balanced trade-off between accuracy and generalization

---

## 🔮 Future Scope

* Increase number of participants and gesture classes
* Support **dynamic gestures** using LSTM / temporal models
* Improve robustness under varying lighting conditions
* Deploy on mobile or embedded platforms
* Integrate gesture-based control systems

---

## 🏁 Conclusion

This project demonstrates a complete **machine learning–based real-time hand gesture recognition system** following the full ML lifecycle. By combining landmark-based feature extraction with a neural network classifier, the system achieves reliable gesture recognition with good generalization across participants.

---

## 🙌 Acknowledgements

* MediaPipe for hand landmark detection
* Scikit-learn for ML model implementation
* OpenCV for real-time computer vision support

---

### ⭐ If you like this project, consider giving it a star on GitHub!

