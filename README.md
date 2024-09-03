# 3D Segmentation Model for CT Abdomen Organs

## Overview

This project involves building a 3D segmentation model for segmenting abdominal organs from CT scans. The model is designed to accurately identify and segment the Liver, Right Kidney, Left Kidney, and Spleen. This project is particularly useful in the medical field for improving the diagnosis and treatment planning processes.

## Repository

- **GitHub Username:** [manas95826](https://github.com/manas95826)
- **Repository Name:** [5C-AI](https://github.com/manas95826/5C-AI)

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Validation and Inference](#validation-and-inference)
- [3D Visualization](#3d-visualization)
- [Results](#results)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/manas95826/5C-AI.git
cd 5C-AI
```

### 2. Install Dependencies

Ensure that you have Python 3.8+ installed. Install the required Python packages:

```bash
pip install torch torchvision nibabel numpy matplotlib scikit-image
```

### 3. Prepare the Dataset

Download the dataset from the following link and place it in the `data/` directory within the repository:

- [CT Abdomen Organ Segmentation Dataset](https://zenodo.org/records/7860267)

Organize the dataset into appropriate folders for training and validation.

### 4. Run the Model

You can train and evaluate the model using the following command:

```bash
python main.py
```

## Model Architecture

The model is based on the VNet architecture, which is well-suited for 3D medical image segmentation tasks. VNet uses convolutional layers and upsampling layers to perform pixel-wise segmentation in 3D.

### Key Components

- **Encoder:** Extracts features from the input CT scans through a series of convolutional and max-pooling layers.
- **Bottleneck:** Represents the core feature extraction stage with the highest level of abstraction.
- **Decoder:** Reconstructs the segmentation map using upsampling and convolutional layers, aligning it back to the original input size.
- **Final Layer:** A 3D convolutional layer that outputs the segmentation map with 4 channels, each corresponding to one of the organs (Liver, Right Kidney, Left Kidney, Spleen).

## Training Process

### 1. Data Preprocessing

- **Normalization:** CT scans are normalized to have a standard mean and variance.
- **Resampling:** The 3D volumes are resampled to a uniform voxel spacing to ensure consistent input dimensions.
- **Splitting:** The dataset is split into training and validation sets to ensure robust evaluation.

### 2. Training

The model is trained using the Cross-Entropy Loss function, with the Adam optimizer for updating the model weights. The training loop iterates over the dataset for a fixed number of epochs, and the loss is minimized through backpropagation.

### 3. Hyperparameters

- **Learning Rate:** 1e-4
- **Epochs:** 50
- **Batch Size:** Adjustable based on GPU memory

## Validation and Inference

### Dice Score

The Dice coefficient is used as the primary evaluation metric. It measures the overlap between the predicted segmentation and the ground truth, with values ranging from 0 (no overlap) to 1 (perfect overlap).

### Inference

After training, the model can predict organ segments on unseen CT scans. The predictions are then compared to the ground truth to calculate the Dice score for each organ.

## 3D Visualization

The segmented organs can be visualized in 3D to provide a clear understanding of the model's performance. The project includes a script to generate 3D visualizations of the predicted organ segments using `matplotlib` and `skimage`.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

def plot_3d(image, threshold=-300):
    verts, faces, _, _ = measure.marching_cubes(image, threshold)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    ax.add_collection3d(mesh)
    plt.show()

plot_3d(prediction[0].cpu().numpy())
```

## Contribution

Feel free to contribute to this project by opening issues or submitting pull requests. Let's improve the accuracy and efficiency of 3D medical image segmentation together!
