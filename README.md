# 📘 PROJECT REPORT

## **Real-Time Hand Gesture Recognition Using Machine Learning**

![Image](https://www.researchgate.net/publication/4280134/figure/fig1/AS%3A642911150555138%401530293371551/Block-diagram-of-the-hands-gesture-recognition-system.png)

![Image](https://towardsdatascience.com/wp-content/uploads/2024/11/1_dlG-Cju5ke-DKp8DQ9hiA%402x.jpeg)

![Image](https://ai.google.dev/static/mediapipe/images/solutions/hand-landmarks.png)

![Image](https://www.researchgate.net/publication/370243116/figure/fig4/AS%3A11431281390923848%401745295635418/MediaPipe-hand-landmarks-with-their-indices-hand-base-point-is-denoted-with-0-36.tif)

---

## **CHAPTER 1: INTRODUCTION**

### 1.1 Background

Human–Computer Interaction (HCI) has evolved rapidly with the advancement of computer vision and machine learning technologies. Traditional interaction devices such as keyboards and mice limit natural communication between humans and machines. Hand gesture recognition offers a more intuitive, contactless, and natural way to interact with digital systems.

Recent progress in real-time vision frameworks and supervised machine learning algorithms has enabled accurate gesture detection using webcam input. By analyzing hand landmarks and motion patterns, machines can interpret human gestures effectively.

---

### 1.2 Problem Statement

Despite the availability of gesture recognition systems, many suffer from limitations such as sensitivity to scale, orientation, lighting conditions, and real-time performance constraints.

**Problem Statement:**

> To design and implement a machine learning–based real-time hand gesture recognition system that accurately detects and classifies hand gestures using live webcam input.

---

### 1.3 Objectives

The primary objectives of this project are:

* To collect hand gesture data using live webcam input
* To extract meaningful hand landmark features
* To normalize and preprocess gesture data
* To train a supervised machine learning model for gesture classification
* To deploy the trained model for real-time gesture prediction

---

### 1.4 Scope of the Project

This project focuses on:

* Static hand gesture recognition
* Real-time classification using webcam input
* Multi-class gesture recognition (25 gestures)
* Machine learning–based classification approach

---

## **CHAPTER 2: LITERATURE REVIEW**

Hand gesture recognition systems typically follow either vision-based or sensor-based approaches. Vision-based methods are preferred due to their non-intrusive nature.

Previous research highlights:

* The effectiveness of hand landmark-based feature extraction
* The importance of normalization for scale and translation invariance
* The suitability of neural networks for non-linear gesture classification

This project adopts a landmark-based approach combined with a Multi-Layer Perceptron (MLP) classifier to achieve reliable performance.

---

## **CHAPTER 3: SYSTEM ARCHITECTURE**

### 3.1 Overall Workflow

![Image](https://www.researchgate.net/publication/281643059/figure/fig1/AS%3A391423158439936%401470333961038/Flowchart-of-hand-gesture-recognition.png)

![Image](https://ai.google.dev/static/mediapipe/images/solutions/examples/hand_gesture.png)

![Image](https://www.researchgate.net/publication/284626785/figure/fig4/AS%3A393491688509453%401470827137439/Architecture-of-gesture-recognition-system-5.png)

**System Flow:**

1. Webcam input capture
2. Hand landmark detection
3. Feature extraction
4. Feature normalization
5. Model training / prediction
6. Output gesture label

---

### 3.2 Tools & Technologies Used

| Component            | Technology                 |
| -------------------- | -------------------------- |
| Programming Language | Python                     |
| Computer Vision      | OpenCV                     |
| Hand Detection       | MediaPipe                  |
| Machine Learning     | Scikit-learn               |
| Model                | MLP Classifier             |
| Data Storage         | Text files                 |
| Deployment           | Real-time webcam inference |

---

## **CHAPTER 4: DATA COLLECTION (ML PHASE 1)**

### 4.1 Data Acquisition

Gesture data is collected using a standard webcam. The system captures video frames in real time and detects hand landmarks using MediaPipe.

Each gesture is recorded under controlled conditions to ensure variation in hand orientation and position.

---

### 4.2 Feature Extraction

For each detected hand, **48 features** are extracted:

* Hand orientation (front/back)
* 21 landmark x-coordinates
* 21 landmark y-coordinates
* 5 finger state values (open/closed)

For two hands:

```
48 (Left Hand) + 48 (Right Hand) = 96 raw features
```

---

### 4.3 Dataset Structure

```
dataset/
└── participant_1/
    ├── gesture_1/
    ├── gesture_2/
    └── gesture_25/
```

Each file represents one gesture sample.

---

## **CHAPTER 5: FEATURE ENGINEERING (ML PHASE 2)**

### 5.1 Need for Normalization

Raw hand landmark coordinates are sensitive to:

* Hand size
* Camera distance
* Position on screen

To address this, normalization is applied.

---

### 5.2 Normalization Technique

Steps performed:

1. Translate landmarks so wrist becomes origin
2. Convert (x, y) → radial distance (r)
3. Normalize distances by maximum value
4. Preserve finger state and orientation

This reduces:

```
96 raw features → 54 normalized features
```

---

### 5.3 Advantages

* Scale invariance
* Translation invariance
* Reduced noise
* Better model generalization

---

## **CHAPTER 6: DATA PREPARATION (ML PHASE 3)**

* Each gesture folder represents a class label
* Gesture IDs (1–25) are used as labels
* Data is stored in numerical vector format

This structured format enables supervised learning.

---

## **CHAPTER 7: MODEL SELECTION (ML PHASE 4)**

### 7.1 Chosen Model: Multi-Layer Perceptron (MLP)

MLP was selected due to:

* Ability to model non-linear relationships
* Suitability for fixed-length numerical input
* Efficient training for moderate datasets

---

### 7.2 Model Architecture

| Layer          | Details    |
| -------------- | ---------- |
| Input          | 54 neurons |
| Hidden Layer 1 | 54 neurons |
| Hidden Layer 2 | 54 neurons |
| Output         | 25 classes |
| Activation     | ReLU       |
| Optimizer      | SGD        |

---

## **CHAPTER 8: MODEL TRAINING (ML PHASE 5)**

### 8.1 Training Process

* Normalized data is loaded
* Model trained using labeled gesture samples
* Loss minimized using backpropagation

### 8.2 Model Saving

The trained model is stored as:

```
gesture_mlp_model.pkl
```

This allows reuse without retraining.

---

## **CHAPTER 9: MODEL EVALUATION (ML PHASE 6)**

### 9.1 Evaluation Method

* Training accuracy is calculated
* Prediction results compared with true labels

### 9.2 Observations

* High training accuracy achieved
* Model shows stable performance for real-time input

---

## **CHAPTER 10: REAL-TIME DEPLOYMENT (ML PHASE 7)**

![Image](https://techvidvan.com/tutorials/wp-content/uploads/sites/2/2021/07/landmark-output.gif)

![Image](https://1.bp.blogspot.com/-8SxmsK5VoJ0/XVrTpMrJDFI/AAAAAAAAEiM/nAa3vuj8a2sjgEPAeMKXD4m09yKUgjVIQCLcBGAs/s1600/Screenshot%2B2019-08-19%2Bat%2B9.51.25%2BAM.png)

![Image](https://pyimagesearch.com/wp-content/uploads/2023/05/hand-gesture-data-visualization-1.png)

### 10.1 Real-Time Inference

* Webcam captures live video
* Hand landmarks extracted per frame
* Features normalized in real time
* Model predicts gesture instantly

### 10.2 Output

* Gesture name displayed on screen
* System operates with minimal latency

---

## **CHAPTER 11: RESULTS**

* Successfully recognized 25 gestures
* Real-time performance achieved
* Accurate prediction under controlled lighting

---

## **CHAPTER 12: CONCLUSION**

This project successfully demonstrates a machine learning–based real-time hand gesture recognition system. By following the complete ML lifecycle—from data collection to deployment—the system achieves reliable gesture classification using webcam input.

The use of landmark-based features and MLP classification ensures both accuracy and efficiency.

---

## **CHAPTER 13: FUTURE SCOPE**

* Increase dataset size
* Add dynamic gesture recognition
* Use deep learning models (CNN/LSTM)
* Deploy on mobile or embedded systems
* Improve robustness under varied lighting


#   h a n d - g e s t u r e - r e c o g n i t i o n  
 