# Pneumonia Detection Using Multimodal Deep Learning

This repository contains a Jupyter notebook for developing a multimodal deep learning model to detect pneumonia using chest X-ray images and patient symptoms. The project leverages a Kaggle dataset for images and synthetic symptom text data, demonstrating preprocessing, model training, and prediction capabilities.

### Dataset Source

The dataset for this project is the Chest X-ray Images (Pneumonia) from Kaggle, consisting of labeled images for normal and pneumonia cases. It is augmented with synthetic symptom text data (e.g., "Patient reports fever, chest pain, fatigue") to enable multimodal learning.

### Key Features

**Data Collection:**
Downloaded chest X-ray dataset from Kaggle using API.
Generated synthetic symptom texts corresponding to image labels for training.

**Data Preprocessing:**
Applied image transformations including resizing to 224x224, grayscale conversion, and normalization.
Tokenized symptom texts using DistilBERT tokenizer and created padded sequences.
Split data into training, validation, and testing sets.

**Model Development:**
-ResNet18 (pre-trained) for image feature extraction (512-dimensional embeddings).
-DistilBERT for text feature extraction (768-dimensional embeddings).
-Fusion layers concatenating features, followed by dense layers, ReLU activation, dropout, and sigmoid output for binary classification.

**Model Training:**
Trained with binary cross-entropy loss and Adam optimizer.
Used batch size of 32 and monitored performance over epochs.

**Pneumonia Prediction:**
Developed inference functionality to process an X-ray image and symptom list, outputting prediction (Pneumonia or Normal) with confidence score.
Handled user inputs for symptoms via numbered selection (e.g., 1 for fever, 3 for chest pain).

**Model Deployment:**
Integrated Streamlit for a potential web application to upload images, select symptoms, and view predictions.

### Technologies Used

Python: Used for data preprocessing, model training, and inference.
PyTorch & Torchvision: Employed for building and training the ResNet18 model and data loaders.
Transformers (Hugging Face): Used for DistilBERT text processing.
Pandas & NumPy: Applied for data handling and synthetic symptom generation.
Scikit-learn: Used for train-test splitting and evaluation metrics (accuracy, precision, recall, F1-score).
Matplotlib & Seaborn: Employed for visualizing training results and confusion matrices.
Streamlit: Used to create an interactive web interface for predictions.

### Project Workflow

**Data Collection:**
Mounted Google Drive in Colab and set up Kaggle API.
Downloaded and extracted the chest X-ray dataset.
Generated synthetic symptom texts and saved metadata in a CSV file.

**Data Preprocessing:**
Defined image transformations and loaded images into datasets.
Tokenized symptoms with DistilBERT and prepared multimodal dataset class.
Created data loaders for training, validation, and testing.

**Model Implementation:**
Loaded pre-trained ResNet18 and DistilBERT models.
Defined fusion model to combine image and text features.
Set up optimizer, loss function, and training loop.

**Model Training:**
Trained the model on GPU, tracking loss and accuracy.
Evaluated on test set with metrics like accuracy.

**Prediction and Evaluation:**
Created prediction function for new images and symptoms.
![Output](https://github.com/AashishSaini16/Pneumonia-Detection-Using-Multimodal-Deep-Learning/blob/main/output.PNG)
