# Brain Tumor Detection

This project focuses on the detection of brain tumors using image classification. Leveraging convolutional neural networks (CNNs), specifically a custom deep learning model, this project aims to classify brain MRI images into different tumor categories.

## Project Overview

The goal of this project is to build a model that can classify MRI images of the brain into one of four categories:
- Glioma Tumor
- Meningioma Tumor
- No Tumor
- Pituitary Tumor

### Key Components

- **Data Loading**: The dataset consists of MRI images of the brain, categorized into four classes.
- **Preprocessing**: Images are resized and normalized for training.
- **Model Architecture**: A CNN model with multiple convolutional and pooling layers is used for classification.
- **Training**: The model is trained and validated using the dataset, with performance metrics plotted.
- **Evaluation**: The model is evaluated on a test set, and results are visualized.
- **Prediction**: The model can predict the class of new MRI images.

## Dataset Description

The dataset used in this project contains MRI images of the brain, categorized into four classes:

- **Classes**:
  - Glioma Tumor
  - Meningioma Tumor
  - No Tumor
  - Pituitary Tumor

### Details

- **Source**: The dataset is sourced from Kaggle. You can download it from the following link: [Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- **Image Size**: Images are resized to 150x150 pixels.
- **Total Images**: The dataset consists of training and testing images categorized into the four classes.

## Requirements

To run this project, you will need the following Python libraries:

- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- scikit-learn
- Matplotlib
- IPython (for widgets)

You can install the necessary packages using pip:

```bash
pip install tensorflow keras numpy pandas opencv-python scikit-learn matplotlib ipywidgets
