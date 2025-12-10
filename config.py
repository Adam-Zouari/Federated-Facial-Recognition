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

# Federated early stopping
FED_EARLY_STOPPING_PATIENCE = 15  # Number of rounds without improvement (None=disabled)
FED_EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement to reset patience

# VGGFace2 Federated Learning Configuration
VGGFACE2_FED_NUM_CLIENTS = 10  # Recommended: 5, 10, or 20
VGGFACE2_FED_PARTITION_STRATEGY = 'iid'  # 'iid' or 'non-iid'
VGGFACE2_FED_ALPHA = 0.5  # Dirichlet concentration for non-iid (0.1=very skewed, 10=balanced)

# FedProx hyperparameters
FEDPROX_MU = 0.01  # Proximal term coefficient

# Data augmentation
# Data augmentation configurations
AUGMENTATION_NONE = {}

AUGMENTATION_WEAK = {
    'horizontal_flip': 0.5,
    'rotation': 10,
    'color_jitter': {
        'brightness': 0.1,
        'contrast': 0.1,
        'saturation': 0.1,
        'hue': 0.05
    }
}

AUGMENTATION_STRONG = {
    'horizontal_flip': 0.5,
    'rotation': 30,  # Increased from 15 for more variation
    'color_jitter': {
        'brightness': 0.4,  # Increased from 0.2
        'contrast': 0.4,    # Increased from 0.2
        'saturation': 0.3,  # Increased from 0.2
        'hue': 0.15         # Increased from 0.1
    },
    'random_affine': {
        'degrees': 0,
        'translate': (0.1, 0.1),  # Up to 10% translation
        'scale': (0.9, 1.1),      # 90-110% scaling
        'shear': 10               # Up to 10 degrees shear
    },
    'random_perspective': 0.2,  # Distortion scale
    'random_erasing': {
        'p': 0.3,                   # 30% probability
        'scale': (0.02, 0.15),      # Erase 2-15% of image
        'ratio': (0.3, 3.3)         # Aspect ratio range
    },
    'gaussian_blur': {
        'kernel_size': 5,
        'sigma': (0.1, 2.0)
    }
}

# Default augmentation config (for backward compatibility)
AUGMENTATION_CONFIG = AUGMENTATION_STRONG

# Model architectures to train
MODEL_ARCHITECTURES = ['custom_cnn', 'resnet18', 'mobilenetv2']

# Output directories
OUTPUT_DIR = './outputs'
CHECKPOINT_DIR = './checkpoints'
LOGS_DIR = './logs'
PLOTS_DIR = './plots'

# MLflow configuration
# Using SQLite backend to avoid filesystem deprecation (Feb 2026)
MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'
MLFLOW_EXPERIMENT_PREFIX = 'facial_recognition'
MLFLOW_ENABLE = True  # Set to False to disable MLflow tracking
