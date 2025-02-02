# Fox-Detector 🦊

**A Keras-Based Binary Classifier That Leverages a Fine-Tuned DenseNet121 Model to Distinguish Fox Images from Non-Fox Images**

---

## Introduction

Fox-Detector is my very first foray into the world of deep learning and computer vision. This project demonstrates a complete machine learning pipeline—from data preparation to model deployment—using Keras and the powerful DenseNet121 architecture. While I've worked hard to implement best practices and robust techniques, please note that some parts of this application might not be fully optimized.

---

## Workflow Overview

### 1. Data Collection & Organization 📸

- **Diverse Sourcing:** I collected a variety of images featuring both foxes and other subjects from public datasets and online sources.
- **Clear Categorization:** The images are divided into two classes:
  - **Foxes** (labeled as `1`)
  - **Non-Foxes** (labeled as `0`)

This organization ensures that the model trains on correctly labeled data, which is crucial for building a reliable classifier.

---

### 2. Image Normalization & Preparation 🖼️

- **Consistent Sizing:** Each image is resized to **256x256 pixels** to match the model’s input requirements.
- **Color Standardization:** All images are converted to **RGB** format, preserving essential visual details.
- **Quality Preservation:** Images are saved with 95% quality to maintain important features.
- **CSV Dataset Generation:** A `dataset.csv` file is created to map each image to its label, streamlining the data loading process during training.

This step ensures that the data fed into the network is uniform and optimized for learning.

---

### 3. Model Training & Optimization 🚀

- **Architecture:** The project utilizes **DenseNet121** with pre-trained ImageNet weights (excluding the top layers) to leverage robust feature extraction.
- **Data Augmentation:** Techniques like rotation, shifting, shearing, flipping, and brightness adjustments are applied to enrich the training dataset and improve generalization.
- **Balancing the Dataset:** Oversampling is used to balance the classes (foxes vs. non-foxes) and reduce bias.
- **Customized Layers:**
  - **Global Average Pooling:** Reduces feature map dimensions.
  - **Dense Layers & Dropout:** Adds fully connected layers with ReLU activation and a 0.5 dropout rate to help prevent overfitting.
  - **Sigmoid Output:** Produces a probability for binary classification.
- **Training Enhancements:**
  - **Early Stopping:** Monitors validation loss to halt training if improvements stop.
  - **Learning Rate Reduction:** Adjusts the learning rate dynamically to fine-tune convergence.

After training, the model is saved as `fox_detector_model.keras`, ready for inference and deployment.

---

### 4. Model Testing & Deployment 🏁

- **Automated Inference:** A dedicated script loads the trained model to process new images.
- **Preprocessing Consistency:** Incoming images are resized and normalized similarly to the training data.
- **Prediction & Sorting:** Based on a threshold of 0.5, images are classified as either fox or non-fox and then automatically moved into:
  - `si_es_zorro` (fox images)
  - `no_es_zorro` (non-fox images)
- **Seamless Integration:** This automated sorting makes it easier to integrate the classifier into practical workflows.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- **Python 3.7+**
- **pip** (Python package installer)

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/warc0s/Fox-Detector.git
   cd Fox-Detector
   ```

2. **Create & Activate a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages**

   ```bash
   pip install pandas scikit-learn tensorflow Pillow
   ```

---

## Usage Instructions

### 1. Normalize & Prepare Your Dataset

Run the script to resize images, convert them to RGB, and generate a `dataset.csv`:

```bash
python normalizador.py
```

### 2. Train the Model

Execute the training script to begin model training with data augmentation, early stopping, and learning rate adjustments:

```bash
python training.py
```

### 3. Test & Deploy the Model

After downloading the pre-trained model (if you haven't trained it locally), classify new images placed in the `todas_imagenes` folder:

```bash
python prueba_modelo.py
```

*Note: Due to GitHub's file size limitations, the trained model (`fox_detector_model.keras`) is hosted on [Google Drive](https://drive.google.com/file/d/1IpDU7fCJNW0iV9jDn4H6YxVE9f7hwKdY/view?usp=sharing). Download it and place it in the project's root directory.*

---

## Final Thoughts 💡

As my first deep learning and computer vision project, Fox-Detector represents both an achievement and a learning journey. While I’ve strived to implement sound methodologies and robust techniques, there is always room for improvement. I appreciate any constructive feedback and contributions to help enhance this project further.

Happy coding, and enjoy exploring the world of fox detection! 🦊✨

---

## License

This project is licensed under the [MIT License](LICENSE).
