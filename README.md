# Fox-Detector

**A Keras-Based Binary Classifier Utilizing a Fine-Tuned DenseNet121 Model to Differentiate Between Fox and Non-Fox Images**

---

## Table of Contents

- [Introduction](#introduction)
- [Overview of Files](#overview-of-files)
- [Detailed Workflow](#detailed-workflow)
  - [1. Data Collection and Organization](#1-data-collection-and-organization)
  - [2. Image Normalization](#2-image-normalization)
  - [3. Model Training](#3-model-training)
  - [4. Model Testing and Deployment](#4-model-testing-and-deployment)
- [Final Thoughts](#final-thoughts)
- [Getting Started](#getting-started)
- [License](#license)

---

## Introduction

The **Fox-Detector** project is a comprehensive machine learning pipeline designed to accurately distinguish images of foxes from those of other animals. Leveraging the power of Keras and the robust DenseNet121 architecture, this binary classifier exemplifies the effective application of deep learning techniques in image classification tasks. The project encompasses data preparation, model training, evaluation, and deployment, providing a holistic approach to building reliable machine learning models.

---

## Overview of Files

- **`normalizador.py`**: 
  - *Purpose*: Handles image normalization, resizing, and dataset organization.
  - *Functionality*: Processes raw images, ensuring consistency in size and format, and generates a structured CSV file for easy data handling during training.

- **`training.py`**: 
  - *Purpose*: Serves as the main script for training the binary classification model.
  - *Functionality*: Implements data augmentation, model architecture setup, training routines with callbacks for optimization, and saves the trained model for future use.

- **`prueba_modelo.py`**: 
  - *Purpose*: Tests the trained model on new images and organizes them based on predictions.
  - *Functionality*: Loads the trained model, processes incoming images, classifies them, and sorts them into designated folders according to the predicted category.

- **`fox_detector_model.keras`**: 
  - *Description*: The trained Keras model file. Due to GitHub's file size limitations, the model is hosted on [Google Drive](https://drive.google.com/file/d/1IpDU7fCJNW0iV9jDn4H6YxVE9f7hwKdY/view?usp=sharing) for easy access and download.

---

## Detailed Workflow

### 1. Data Collection and Organization

**Rationale:**
A well-structured dataset is foundational to training an effective machine learning model. The initial step involves sourcing images of foxes and non-fox animals from platforms like Kaggle and the internet. 

**Process:**
- **Source Acquisition**: Gather a diverse set of images to ensure the model learns to recognize foxes across various environments, poses, and lighting conditions.
- **Categorization**: Systematically sort the images into two primary categories:
  - `si_zorros` (images containing foxes)
  - `no_zorros` (images without foxes)

**Importance:**
Organizing data into clear categories ensures that the model receives accurate labels during training, which is critical for supervised learning tasks. This categorization also facilitates balanced training, reducing potential biases.

### 2. Image Normalization

**Rationale:**
Consistency in image size and format is crucial for model performance. Normalization ensures that all input data adheres to the expected specifications of the neural network, enhancing training efficiency and accuracy.

**Process:**
Handled by `normalizador.py`, this step involves:
- **Resizing**: All images are resized to **256x256 pixels** to maintain uniformity, which simplifies the computational complexity and ensures compatibility with the DenseNet121 architecture.
- **Color Conversion**: Images are converted to **RGB** format, as pre-trained models like DenseNet121 are trained on three-channel color images.
- **Quality Enhancement**: Images are saved with high quality (95%) to preserve important features that aid in accurate classification.
- **Dataset Compilation**: A `dataset.csv` file is generated, cataloging each image's filename and corresponding label (1 for foxes, 0 for non-foxes). This CSV serves as the blueprint for data loading during training.

**Importance:**
Standardizing image dimensions and formats eliminates discrepancies that could hinder the learning process. Additionally, maintaining image quality ensures that the model has access to all relevant features necessary for accurate predictions.

### 3. Model Training

**Rationale:**
Training a robust model requires leveraging advanced architectures and implementing strategies to enhance generalization while preventing overfitting.

**Process:**
Executed via `training.py`, this step encompasses:
- **Architecture Selection**: Utilizes the **DenseNet121** model, renowned for its efficient parameter usage and strong performance in image classification tasks. DenseNet121's dense connectivity facilitates feature reuse and gradient flow, enhancing learning efficiency.
  
- **Data Augmentation**: Implements various augmentation techniques using `ImageDataGenerator` to artificially expand the dataset, improving the model's ability to generalize to unseen data. Techniques include:
  - **Rotation**, **Width/Height Shifts**, **Shear**, **Horizontal Flip**, and **Brightness Adjustments**.
  
- **Dataset Balancing**: Addresses class imbalance through **oversampling** of the minority class (fox images), ensuring that the model does not become biased towards the majority class.
  
- **Model Customization**:
  - **Global Average Pooling**: Reduces the spatial dimensions of the feature maps, lowering the number of parameters and mitigating overfitting.
  - **Dense Layers**: Adds fully connected layers with **ReLU** activation to learn complex patterns.
  - **Dropout**: Incorporates a **Dropout** layer with a rate of 0.5 to prevent overfitting by randomly disabling neurons during training.
  - **Output Layer**: Utilizes a **sigmoid** activation function for binary classification.
  
- **Compilation**:
  - **Optimizer**: Uses the **Adam** optimizer with a learning rate of 0.0001 for efficient gradient-based optimization.
  - **Loss Function**: Employs **binary cross-entropy** suitable for binary classification tasks.
  - **Metrics**: Tracks **accuracy** to monitor the model's performance.
  
- **Training Enhancements**:
  - **Early Stopping**: Implements `EarlyStopping` to halt training if the validation loss does not improve for five consecutive epochs, preventing overfitting and reducing unnecessary computations.
  - **Learning Rate Reduction**: Uses `ReduceLROnPlateau` to decrease the learning rate by a factor of 0.2 if the validation loss plateaus for three epochs, facilitating finer convergence.

- **Model Saving**: After training, the model is saved as `fox_detector_model.keras` for future inference and deployment.

**Importance:**
Selecting a powerful architecture like DenseNet121, combined with strategic data augmentation and regularization techniques, enhances the model's ability to learn meaningful patterns while maintaining generalizability. Implementing callbacks ensures efficient training and optimal performance.

### 4. Model Testing and Deployment

**Rationale:**
Testing the model on new, unseen data is essential to evaluate its real-world performance and reliability. Automating the classification and organization of images streamlines the deployment process.

**Process:**
Managed by `prueba_modelo.py`, this step involves:
- **Model Loading**: Loads the pre-trained `fox_detector_model.keras` for inference.
  
- **Image Processing**:
  - **Loading and Resizing**: Each image in the `todas_imagenes` directory is loaded and resized to **256x256 pixels**, ensuring consistency with the training data.
  - **Normalization**: Pixel values are scaled by dividing by 255.0, matching the preprocessing steps applied during training.
  
- **Prediction**:
  - **Inference**: The model predicts the probability of each image containing a fox.
  - **Classification**: Based on a threshold of 0.5, images are classified as either fox (`si_es_zorro`) or non-fox (`no_es_zorro`).
  
- **Organization**:
  - **File Sorting**: Images are automatically moved to their respective folders (`si_es_zorro` or `no_es_zorro`) based on the prediction outcome.
  
- **Error Handling**: Incorporates checks to ensure only valid image files are processed, enhancing the robustness of the script.

**Importance:**
Automating the classification and organization of images facilitates efficient deployment of the model in practical applications. It allows for seamless integration into workflows where large volumes of images need to be categorized swiftly and accurately.

---

## Final Thoughts

The **Fox-Detector** project embodies a full-fledged machine learning pipeline, meticulously crafted from data preparation to model deployment. Each component of the project emphasizes best practices in machine learning, such as data normalization, augmentation, model selection, regularization, and efficient deployment strategies. This endeavor not only delivers a functional classifier but also serves as an educational blueprint for building similar image classification systems. The iterative process of refining each step underscores the significance of attention to detail, adaptability, and continuous learning in the realm of machine learning and artificial intelligence.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/warc0s/Fox-Detector
   cd Fox-Detector
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install pandas scikit-learn tensorflow Pillow
   ```

### Usage

1. **Data Normalization**

   Process and organize your images:

   ```bash
   python normalizador.py
   ```

2. **Model Training**

   Train the binary classifier:

   ```bash
   python training.py
   ```

3. **Model Testing and Deployment**

   Classify new images and organize them accordingly:

   ```bash
   python prueba_modelo.py
   ```

### Accessing the Trained Model

Due to GitHub's file size limitations, the trained model (`fox_detector_model.keras`) is hosted on [Google Drive](https://drive.google.com/file/d/1IpDU7fCJNW0iV9jDn4H6YxVE9f7hwKdY/view?usp=sharing). Download it and place it in the root directory of the project to use with `prueba_modelo.py`.

---

## License

This project is licensed under the [MIT License](LICENSE).
