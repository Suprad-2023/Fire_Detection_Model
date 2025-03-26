# Forest Fire Detection Using Deep Learning

### Author: Sharad Kumar Pani
### Course: Data Science Capstone Project

## Table of Contents  
1. [Introduction](#introduction)  
2. [Dataset Overview](#dataset-overview)  
3. [Methodology](#methodology)  
   - [Exploratory Data Analysis](#exploratory-data-analysis)  
   - [Data Preprocessing](#data-preprocessing)  
   - [Feature Engineering](#feature-engineering)  
   - [Model Selection](#model-selection)  
   - [Model Training](#model-training)  
4. [Future Work](#future-work)  
5. [Conclusion](#conclusion)  

---

## **Introduction**  
Forest fires pose a serious threat to ecosystems, human settlements, and wildlife. Traditional fire detection methods, such as smoke sensors and satellite imaging, suffer from delayed detection and high false alarms.

This project aims to develop a deep learning-based image classification model to accurately detect Fire, Smoke, and Non-Fire from surveillance images using Convolutional Neural Networks (CNNs).

Key Objectives:
✔ Develop an deep learning model for fire detection using images.
✔ Achieve high accuracy while minimizing false positives.
✔ Deploy the model as a real-time monitoring system. 

## **Dataset Overview**  
📂 Dataset Name: Forest Fire, Smoke, and Non-Fire Dataset
📊 Total Images: 31,500 (Balanced across three classes) for Train
📊 Total Images: 10,500 (Balanced across three classes) for Test
📌 Classes:
Train
🔥 Fire → 10,500 images
🚫 Non-Fire → 10,500 images
🌫 Smoke → 10,500 images
Test
🔥 Fire → 3,500 images
🚫 Non-Fire → 3,500 images
🌫 Smoke → 3,500 images
🔄 Train-Test Split: 70% Train, 30% Test
🖼 Image Size: 224x224 pixels

## **Methodology**  

### **Exploratory Data Analysis**  
EDA was conducted to understand image distribution and class separability.
✔ Visualized sample images from each category
✔ Checked for duplicate images
✔ Analyzed pixel intensity distribution

### **Data Preprocessing**  
To ensure the dataset was suitable for model training:
✔ Resized images to 224x224 pixels
✔ Normalized pixel values (0-1 scaling)
✔ Applied data augmentation (rotation, flipping, zooming, shearing, etc.)
✔ Split data into training (80%) and validation (20%) sets 

### **Feature Engineering**  
✔ Extracted important features using CNN layers
✔ Applied image transformations to improve model robustness
✔ Tuned contrast and brightness for better generalization

### **Model Selection**  
Various deep learning models were evaluated, including:
✔ Custom CNN Model (3 Conv2D layers, MaxPooling, Flatten, Dense layers)
✔ Pretrained Models (Transfer Learning):

MobileNetV2 🏆 (Selected due to its speed and accuracy)
ResNet50
VGG16
🏆 Best Model: MobileNetV2 (achieved highest accuracy & lowest false alarms).

### **Model Training**  
✔ Optimizer: Adam
✔ Loss Function: Categorical Cross-Entropy
✔ Batch Size: 32
✔ Epochs: 7 (Early stopping used)
✔ Regularization Techniques Used:

Dropout (0.5) to prevent overfitting
L2 Regularization to avoid excessive weight updates


## **Future Work**  
🔮 To further improve the project, the following enhancements are planned:
✔ Real-time Fire Detection – Integrate with CCTV/Drones for live monitoring.
✔ Deploy Model on Edge Devices – Optimize for Raspberry Pi, Jetson Nano.
✔ Integrate Early Warning System – Send alerts via SMS/Email upon detection.
✔ Use Object Detection Instead of Classification – Implement YOLOv8 to detect fire locations in images.
✔ Expand Dataset – Include different weather conditions (fog, rain).

## **Conclusion**  
✔ Successfully developed a deep-learning-based fire detection model using Streamlit.
✔ Achieved 97% accuracy using MobileNetV2 Transfer Learning.
✔ Applied data augmentation & regularization to improve model generalization.
✔ The model is suitable for real-time fire monitoring applications.

🚀 Next Steps:
✔ Deploy the model in real-world fire detection systems.
✔ Improve real-time detection speed using optimized neural networks.
