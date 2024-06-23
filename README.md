# Fox-Detector
Keras-based binary classifier leveraging a fine-tuned DenseNet121 model to distinguish between fox and non-fox images.

This repository houses the Keras model and scripts utilized to train a binary classifier aimed at distinguishing images of foxes from those of other animals using the DenseNet121 architecture. Each step of the process has contributed to understanding the nuances of machine learning application, from data preparation to model evaluation.

## Overview of Files
- `normalizador.py`: Handles image normalization and dataset organization.
- `training.py`: Main script for model training.
- `prueba_modelo.py`: Tests the trained model and sorts images into corresponding folders.
- `fox_detector_model.keras`: Due to file size limitations on GitHub, the trained model file is hosted on Google Drive. You can access and download it [here](https://drive.google.com/file/d/1IpDU7fCJNW0iV9jDn4H6YxVE9f7hwKdY/view?usp=sharing).


## Detailed Steps

### Step 1: Data Collection and Organization
Images were sourced from Kaggle or the internet and systematically sorted into 'si_zorros' (fox images) and 'no_zorros' (non-fox images). This initial step of creating a well-organized dataset underscores the foundational role of structured data in training effective machine learning models.

### Step 2: Image Normalization
Handled by `normalizador.py`, this step involved resizing all images to 300x300 pixels and converting them to RGB. A `dataset.csv` was also generated to catalog the images and their respective labels. Standardizing the format and size of the images is crucial for consistent model input, facilitating more effective learning and prediction accuracy.

### Step 3: Model Training
`training.py` was used to train the model on the pre-processed images. Employing the DenseNet121 architecture, the script enhanced the dataset through various data augmentation techniques like rotation and brightness/contrast adjustments to ensure a balanced representation of classes. An early stopping mechanism was implemented during training, halting further epochs if no improvement in validation loss was observed after five iterations. This approach helped refine the training process, emphasizing the importance of model adaptability and the prevention of overfitting.

### Step 4: Model Testing and Usage
The `prueba_modelo.py` script applies the trained model to new images, classifying them into 'si_zorros' or 'no_zorros' based on their predicted categories. This phase highlighted the model’s practical application and provided insight into its performance and reliability in real-world scenarios.

## Conclusion
Through each phase of this project, from data preprocessing through to the final testing, practical insights were gained into fine-tuning convolutional neural networks and the critical steps necessary for developing a robust classifier.

Explore the provided scripts and the trained model to gain deeper insights into the specific techniques and strategies employed in this project.
