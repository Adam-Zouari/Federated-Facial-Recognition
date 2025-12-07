"""
Global configuration for the federated learning facial recognition project.
"""

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
IMG_SIZE = (128, 128)
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# Dataset paths (modify these according to your local setup)
DATA_PATHS = {
    'celeba': './data/CelebA',
    'vggface2': './data/VGGFace2'
}

# Train/Val/Test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Local training hyperparameters
LOCAL_EPOCHS = 20
LOCAL_BATCH_SIZE = 32
LOCAL_LEARNING_RATE = 0.0001  # Reduced from 0.001 for better convergence
EARLY_STOPPING_PATIENCE = 10  # Increased from 5 to 10

# Centralized training hyperparameters
CENTRAL_EPOCHS = 30
CENTRAL_BATCH_SIZE = 64
CENTRAL_LEARNING_RATE = 0.001

# Federated learning hyperparameters
FED_ROUNDS = 50
FED_EPOCHS_PER_ROUND = 5
FED_BATCH_SIZE = 32
FED_LEARNING_RATE = 0.001
FED_CLIENT_FRACTION = 1.0  # Fraction of clients to use per round

# FedProx hyperparameters
FEDPROX_MU = 0.01  # Proximal term coefficient

# Data augmentation
AUGMENTATION_CONFIG = {
    'horizontal_flip': 0.5,
    'rotation': 15,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
}

# Model architectures to train
MODEL_ARCHITECTURES = ['custom_cnn', 'resnet18', 'mobilenetv2']

# Output directories
OUTPUT_DIR = './outputs'
CHECKPOINT_DIR = './checkpoints'
LOGS_DIR = './logs'
PLOTS_DIR = './plots'

# MLflow configuration
MLFLOW_TRACKING_URI = './mlruns'
MLFLOW_EXPERIMENT_PREFIX = 'facial_recognition'
MLFLOW_ENABLE = True  # Set to False to disable MLflow tracking
