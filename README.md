# Tomato Leaf Disease Classification Using Artificial Neural Networks (ANN)

## Project Overview
This project upgrades the tomato leaf disease classification pipeline from traditional machine learning to Deep Learning using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras**. To maximize model performance, automated hyperparameter tuning was conducted using **Optuna**. An interactive **Streamlit** web application is deployed for real-time image inference and disease prediction.

## Objective
To classify tomato leaf images into 10 distinct categories (healthy and diseased) using a fully connected deep neural network architecture optimized for accuracy and generalization.

## Dataset
* **Dataset Source:** New Plant Diseases Dataset (Augmented)
* **Selected Classes:** 10 Tomato leaf categories (~18,000+ training images)
  - Bacterial Spot
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Septoria Leaf Spot
  - Spider Mites (Two-spotted Spider Mite)
  - Target Spot
  - Tomato Yellow Leaf Curl Virus
  - Tomato Mosaic Virus
  - Healthy

## Technologies Used
* Python
* TensorFlow / Keras
* Optuna (Hyperparameter Optimization)
* NumPy
* Pandas
* Matplotlib
* Pillow (PIL)
* Streamlit

## Deep Learning & Optimization Workflow
1. Dataset collection and extraction from Google Drive
2. Automated directory parsing for train and validation splits
3. Image resizing (64x64) and pixel normalization (`[0.0, 1.0]`)
4. TensorFlow `tf.data` pipeline optimization with prefetching
5. Optuna hyperparameter optimization study (tuning layers, units, dropout, learning rate, and optimizers)
6. Final model training using optimized hyperparameters
7. Model evaluation and training history visualization (Accuracy & Loss curves)
8. Model serialization (`.keras` format)
9. Web application development and deployment using Streamlit

## Model Details
* **Architecture:** Sequential Feed-Forward Artificial Neural Network (ANN)
* **Input Layer:** Flattened RGB image tensor (`64 x 64 x 3 = 12,284` inputs)
* **Hidden Layers:** Dynamically optimized (3 Dense layers with ReLU activation and Dropout regularization)
* **Output Layer:** Dense Softmax layer (10 classes)
* **Optimizer:** SGD (Optimized via Optuna with learning rate ~0.0099)
* **Loss Function:** Sparse Categorical Crossentropy

## Model Performance
* **Optuna Best Trial Validation Accuracy:** 68.16%
* **Final Optimized Model Validation Accuracy:** ~69.64% *(Significant performance improvement over the traditional SVM baseline of 56.84%)*

## Web Application Features
The Streamlit application provides an intuitive interface supporting:
- Tomato leaf image upload (`.jpg`, `.jpeg`, `.png`)
- Real-time uploaded image preview
- Automated image preprocessing and tensor resizing
- Disease class prediction with confidence percentage
- Clean, responsive UI for agricultural diagnostics

## Project Structure
Tomato_Leaf_Disease_Classification_ANN/
│
├── app.py                      # Streamlit web application interface
├── requirements.txt            # Project dependencies and libraries
├── README.md                   # Project documentation
└── tomato_ann_model.keras      # Trained and optimized ANN deep learning model
