# Traffic Sign Recognition Using Deep Learning and CNN

## 1. Introduction
A deep-learning system that classifies traffic sign images into the 43 classes of the GTSRB dataset using a Convolutional Neural Network, with a Streamlit web interface.

## 2. Problem Statement
Manually recognising road signs is error-prone for drivers and is a core requirement for driver-assistance and autonomous vehicles. The goal is to automatically classify a traffic sign image.

## 3. Objectives
- Preprocess traffic sign images (resize 32×32, normalise).
- Train a CNN to classify 43 sign classes.
- Evaluate with accuracy, precision, recall, F1-score, confusion matrix.
- Provide a Streamlit app showing predicted class, confidence and top-3 predictions.

## 4. Technologies Used
Python, TensorFlow/Keras, OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn, Streamlit.

## 5. Dataset
GTSRB (German Traffic Sign Recognition Benchmark): ~39,000 training images, 43 classes.

### Dataset setup
**Option 1 – Kaggle CLI (automatic):**
```powershell
pip install kaggle
# put kaggle.json in C:\Users\<you>\.kaggle\  (Kaggle > Settings > Create New Token)
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p dataset --unzip
```
**Option 2 – Manual:** open https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign , click Download, extract the zip into `dataset/`.

Required structure (only `Train` is needed):
```
dataset/
└── Train/
    ├── 0/   (images of class 0)
    ├── 1/
    └── ... 42/
```
If extraction creates a nested folder (e.g. `dataset/archive/Train`), move `Train` directly under `dataset/`.

## 6. System Architecture
Input Image → Preprocessing → Resize 32×32 → Normalisation → CNN → Feature Extraction → Classification → Softmax → Prediction

## 7. Methodology
Load images with OpenCV → convert BGR→RGB → resize 32×32 → divide by 255 → 80/20 stratified train/validation split → train CNN → evaluate → save model → serve through Streamlit.

## 8. CNN Architecture
| Layer | Output |
|---|---|
| Conv2D(32, 3×3, ReLU) + MaxPool | 16×16×32 |
| Conv2D(64, 3×3, ReLU) + MaxPool | 8×8×64 |
| Conv2D(128, 3×3, ReLU) + MaxPool | 4×4×128 |
| Flatten | 2048 |
| Dense(256, ReLU) | 256 |
| Dropout(0.5) | 256 |
| Dense(43, Softmax) | 43 |

## 9. Training Process
Adam optimizer, sparse categorical crossentropy, accuracy metric, batch size 64, up to 15 epochs, EarlyStopping (patience 4) and ModelCheckpoint (best val accuracy → `models/traffic_sign_cnn.h5`).

## 10. Evaluation Metrics
Accuracy, precision, recall, F1-score (classification report) and confusion matrix. Graphs are saved as `accuracy_graph.png`, `loss_graph.png`, `confusion_matrix.png`.

## 11. How to Install (Windows PowerShell)
Use Python 3.10 or 3.11 (TensorFlow 2.15 does not support 3.12+).
```powershell
cd traffic-sign-recognition
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## 12. How to Train
```powershell
python train.py                                  # full dataset
python train.py --max-per-class 300 --epochs 8   # fast demo mode (~minutes)
```

## 13. How to Predict from the command line
```powershell
python predict.py dataset\Train\14\00014_00000_00000.png
```

## 14. How to Run Streamlit
```powershell
streamlit run app.py
```

## 15. Expected Output
Training prints accuracy/loss per epoch, then a classification report and confusion matrix. The web app shows the uploaded image, predicted sign, confidence % and top-3 predictions. (Insert your own measured accuracy here.)

## 16. Future Enhancements
Data augmentation, transfer learning (MobileNet), real-time webcam detection, sign detection (YOLO) before classification, mobile deployment.

## 17. Conclusion
A simple CNN can classify GTSRB traffic signs effectively, and wrapping it in Streamlit makes it easy to demonstrate.
