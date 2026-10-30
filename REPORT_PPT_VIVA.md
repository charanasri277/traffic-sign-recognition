# Project Report Content, PPT and Viva

## Abstract
Traffic sign recognition is a key component of driver-assistance systems and autonomous vehicles. This project presents a deep-learning system that classifies traffic sign images into 43 categories using a Convolutional Neural Network (CNN) trained on the German Traffic Sign Recognition Benchmark (GTSRB). Images are read with OpenCV, resized to 32×32 pixels and normalised to the 0–1 range. The CNN has three convolution–ReLU–max-pooling blocks followed by a fully connected layer, a dropout layer and a softmax output layer. The model is trained with the Adam optimizer and sparse categorical crossentropy loss, using early stopping and model checkpointing to limit overfitting. Performance is evaluated using accuracy, precision, recall, F1-score and a confusion matrix. A Streamlit web application lets a user upload an image and view the predicted sign, the confidence percentage and the top three predictions. The system is simple, lightweight and trains in a short time on a normal computer, making it a practical demonstration of deep learning for image classification. [Insert your final accuracy here.]

## Introduction
Road traffic signs communicate rules, warnings and directions to drivers. Missing or misreading a sign can cause accidents, so automatic recognition of signs is an important research area in intelligent transportation. Modern vehicles with advanced driver-assistance systems (ADAS) already read speed-limit and stop signs, and fully autonomous vehicles must recognise every sign reliably under varying conditions.

Traditional computer-vision approaches relied on hand-crafted features such as colour thresholds, shape detection and HOG descriptors, combined with classifiers like SVM. These methods are sensitive to lighting, blur, partial occlusion and viewpoint changes, and they need careful manual tuning. Deep learning, and especially Convolutional Neural Networks, learns features directly from raw pixels and has become the standard approach for image classification.

A CNN uses convolution layers to detect local patterns such as edges and corners, pooling layers to reduce spatial size and add tolerance to small shifts, and dense layers to combine the learned features into a decision. Because the same filters are applied across the whole image, CNNs need far fewer parameters than fully connected networks and generalise better on images.

This project uses the GTSRB dataset, a widely used benchmark containing more than 39,000 training images across 43 classes recorded under real driving conditions. A compact CNN is built with TensorFlow/Keras, trained on 32×32 colour images, and evaluated using standard metrics. The trained model is integrated into a Streamlit application so that any user can upload a sign image and receive the predicted class with a confidence score. The project focuses on classification of cropped sign images; detection of signs inside full road scenes is left as future work.

## Problem Statement
To design and implement a deep-learning model that automatically and accurately classifies a given traffic sign image into one of the 43 GTSRB classes, and to provide a simple web interface that displays the predicted sign and its confidence.

## Objectives
1. Study CNN-based image classification.
2. Preprocess GTSRB images (resize to 32×32, normalise).
3. Build and train a CNN classifier for 43 classes.
4. Evaluate the model using accuracy, precision, recall, F1-score and confusion matrix.
5. Develop a Streamlit application for real-time prediction.

## Literature / Related Work
- Early systems used colour segmentation, shape matching and HOG features with SVM or random forest classifiers; accuracy dropped under poor lighting and occlusion.
- Stallkamp et al. introduced the GTSRB benchmark (IJCNN 2011) and reported human performance of about 98.8%.
- Ciresan et al. used multi-column deep neural networks on GTSRB and reached accuracy above human level.
- Sermanet and LeCun applied multi-scale convolutional networks to traffic sign recognition.
- Later work uses transfer learning (VGG, ResNet, MobileNet) and detectors such as YOLO for detection plus classification.
*(Verify these references and add exact citations before submission.)*

## Proposed Methodology
1. Collect GTSRB training images.
2. Read images with OpenCV and convert BGR to RGB.
3. Resize to 32×32 and normalise by dividing by 255.
4. Split 80% training / 20% validation (stratified).
5. Build the CNN and compile with Adam and sparse categorical crossentropy.
6. Train with EarlyStopping and ModelCheckpoint.
7. Evaluate and save the model as `traffic_sign_cnn.h5`.
8. Deploy through Streamlit for user predictions.

## System Architecture
Input Image → Image Preprocessing → Resize to 32×32 → Normalization → CNN → Feature Extraction → Classification → Softmax → Traffic Sign Prediction

- **Input image:** a user-uploaded sign image.
- **Preprocessing/resize/normalisation:** makes all inputs the same size and scale (0–1).
- **CNN – feature extraction:** convolution and pooling layers learn edges, shapes and patterns.
- **Classification:** flatten and dense layers combine features.
- **Softmax:** converts outputs into 43 probabilities summing to 1; the highest is the prediction and its value is the confidence.

## CNN Architecture
| Layer | Details | Purpose |
|---|---|---|
| Input | 32×32×3 | RGB image |
| Conv2D 1 | 32 filters, 3×3, ReLU, same padding | Learns low-level features (edges) |
| MaxPooling 1 | 2×2 → 16×16×32 | Reduces size, adds shift tolerance |
| Conv2D 2 | 64 filters, 3×3, ReLU | Learns shapes and corners |
| MaxPooling 2 | 2×2 → 8×8×64 | Downsamples |
| Conv2D 3 | 128 filters, 3×3, ReLU | Learns complex sign patterns |
| MaxPooling 3 | 2×2 → 4×4×128 | Downsamples |
| Flatten | 2048 values | Converts feature maps to a vector |
| Dense | 256 units, ReLU | Combines features |
| Dropout | 0.5 | Reduces overfitting |
| Dense (output) | 43 units, Softmax | Class probabilities |

## Dataset Description
GTSRB contains over 50,000 images of German road signs in 43 classes (about 39,209 in the Kaggle training set). Images vary in size (about 15×15 to 250×250), lighting, blur and viewpoint. Classes are imbalanced (some speed-limit and priority classes have many more images than others). Each class is stored in a numbered folder (0–42).

## Training and Testing
The training folder is split 80/20 into training and validation sets with stratification. The model is trained for up to 15 epochs, batch size 64. EarlyStopping (patience 4) stops training when validation loss stops improving, and ModelCheckpoint saves the best model. [Optionally also test on the official `Test.csv` images.]

## Evaluation Metrics
- **Accuracy** = correct predictions / total predictions.
- **Precision** = TP / (TP + FP): of the signs predicted as a class, how many were right.
- **Recall** = TP / (TP + FN): of the real signs of a class, how many were found.
- **F1-score** = 2·(Precision·Recall)/(Precision+Recall): balance of precision and recall.
- **Confusion matrix**: table of true vs predicted classes; the diagonal shows correct predictions and off-diagonal cells show which classes are confused.

## Results
*(Replace the placeholders with your own output from `python train.py`.)*
- Training accuracy: **[INSERT]**
- Validation accuracy: **[INSERT]**
- Training loss: **[INSERT]**
- Validation loss: **[INSERT]**
- Macro / weighted precision, recall, F1: **[INSERT from classification report]**
- Figures: insert `accuracy_graph.png`, `loss_graph.png`, `confusion_matrix.png`.
- Streamlit screenshots: **[INSERT]**

Suggested discussion: compare training vs validation curves to comment on overfitting; name the classes with lowest F1 in your report and note they are usually visually similar signs (e.g. different speed limits) or classes with few images.

## Advantages
Simple and fast to train; learns features automatically; good accuracy on clean images; easy web demo; low memory footprint.

## Limitations
Classifies only cropped sign images (no detection in full scenes); 32×32 input loses detail; no data augmentation; limited to the 43 German sign classes; performance may drop on blur, night, glare or occlusion; no handling of unknown/non-sign images.

## Future Scope
Data augmentation, transfer learning, real-time webcam/video input, sign detection with YOLO/SSD, support for Indian traffic signs, deployment on mobile/embedded devices.

## Conclusion
The project shows that a simple CNN can classify GTSRB traffic signs and that the model can be delivered as an easy-to-use web app. The system forms a base for more advanced driver-assistance applications. [Add your final accuracy.]

---

# 10 PPT Slides

**1. Title** – Traffic Sign Recognition Using Deep Learning and CNN | Your name, roll no., guide, college, date

**2. Introduction** – Traffic signs guide safe driving • ADAS/autonomous cars need automatic recognition • CNNs learn features from images

**3. Problem Statement** – Manual sign reading is error-prone • Classical methods fail under lighting/occlusion • Need automatic, accurate 43-class classifier

**4. Objectives** – Preprocess images • Build CNN • Evaluate with accuracy/precision/recall/F1 • Streamlit web app

**5. Dataset** – GTSRB • ~39k training images • 43 classes • varying light, blur, size • resized to 32×32

**6. Proposed Methodology** – Image → Preprocess → Resize 32×32 → Normalize → CNN → Softmax → Prediction • 80/20 split • Adam + early stopping

**7. CNN Architecture** – 3 × (Conv + ReLU + MaxPool) with 32/64/128 filters → Flatten → Dense 256 → Dropout 0.5 → Dense 43 Softmax

**8. Results** – Accuracy [INSERT] • Loss [INSERT] • graphs • confusion matrix • app screenshot

**9. Applications and Future Scope** – Applications: ADAS, autonomous cars, road-sign inventory • Future: augmentation, transfer learning, real-time detection, mobile app

**10. Conclusion** – CNN classifies traffic signs effectively • Streamlit gives easy demo • Base for real-time driver-assistance systems • Thank you / Questions

---

# 20 Viva Questions and Answers

1. **What is CNN?** A neural network designed for images that uses convolution filters to learn spatial features automatically.
2. **Why CNN instead of traditional ML?** Traditional ML needs hand-crafted features; a CNN learns them from raw pixels, handles spatial structure, and generally gives higher accuracy on images.
3. **What is convolution?** Sliding a small filter (e.g. 3×3) over the image and computing dot products to produce a feature map that detects patterns like edges.
4. **What is pooling?** Downsampling of feature maps; max pooling keeps the maximum in each window (2×2), reducing size and computation and adding shift tolerance.
5. **Why use ReLU?** ReLU = max(0, x). It adds non-linearity, is cheap to compute and reduces the vanishing-gradient problem.
6. **Why use Softmax?** It converts the 43 output scores into probabilities that sum to 1, so we get a class and a confidence.
7. **What is overfitting?** When the model memorises training data and performs poorly on unseen data (high training accuracy, low validation accuracy).
8. **What is dropout?** Randomly switching off a fraction of neurons (50% here) during training so the network does not rely on specific neurons, reducing overfitting.
9. **Why normalize images?** Scaling pixels from 0–255 to 0–1 makes training faster and more stable.
10. **What is GTSRB?** German Traffic Sign Recognition Benchmark, a standard dataset of real-world German traffic sign images.
11. **How many classes are there?** 43.
12. **What is image classification?** Assigning a single label to an entire image.
13. **What is accuracy?** Correct predictions divided by total predictions.
14. **What is a confusion matrix?** A table of actual vs predicted classes showing correct predictions on the diagonal and mistakes elsewhere.
15. **Why use Adam optimizer?** It combines momentum and adaptive learning rates, converges quickly and needs little tuning.
16. **What is an epoch?** One complete pass through the whole training dataset.
17. **What is batch size?** Number of samples processed before the weights are updated (64 here).
18. **What happens during training?** Forward pass produces predictions, the loss compares them with true labels, backpropagation computes gradients, and Adam updates the weights; repeated over epochs.
19. **How does the uploaded image get classified?** It is converted to RGB, resized to 32×32, normalised, passed through the CNN, and softmax gives 43 probabilities; the highest is shown with its confidence.
20. **What are the limitations?** Only cropped sign classification, no detection in full scenes, low resolution, no augmentation, limited to the 43 German classes, weaker on poor lighting or occlusion.
