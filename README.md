# Federated Learning for Facial Recognition

This project implements a complete federated learning pipeline for facial recognition using two datasets: CelebA and VGGFace2. Features include stratified data sampling, class-weighted loss, checkpoint resumption, and MLflow tracking with SQLite backend.

## Project Structure

```
project/
├── data/                    # Dataset storage
│   ├── CelebA/             # CelebA dataset with identity labels
│   └── VGGFace2/           # VGGFace2 dataset (train/val splits)
├── clients/                 # Client-specific dataset handlers
│   ├── celebA_client.py
│   └── vggface_client.py
├── models/                  # Model architectures (train from scratch)
│   ├── cnn_custom.py
│   ├── resnet.py
│   └── mobilenet.py
├── federated/              # Federated learning implementation
│   ├── fedavg.py
│   ├── fedprox.py
│   ├── server.py
│   └── client.py
├── centralized/            # Centralized baseline
│   └── train_global.py
├── utils/                  # Utility functions
│   ├── preprocessing.py
│   ├── metrics.py
│   └── plotting.py
├── outputs/                # Generated outputs
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
├── plots/                  # Visualization plots
├── config.py              # Global configuration
├── main.py                # Main entry point
├── train_local.py         # Local training script
├── run_federated.py       # Federated learning script
└── evaluate.py            # Evaluation script
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare datasets:
   - **CelebA**: Place in `data/CelebA/` with structure:
     ```
     CelebA/
     ├── img_align_celeba/
     ├── Anno/
     │   └── identity_CelebA.txt (required for identity labels)
     └── list_eval_partition.txt
     ```
   - **VGGFace2**: Place in `data/VGGFace2/` with structure:
     ```
     VGGFace2/
     ├── train/
     │   ├── n000001/
     │   ├── n000002/
     │   └── ...
     └── val/
         ├── n000001/
         └── ...
     ```

3. Update paths in `config.py` if needed.

## Usage

### 1. Explore Datasets
```bash
python main.py --mode explore
```

### 2. Train Local Models

**Basic Training:**
```bash
# Train on CelebA with ResNet-18
python train_local.py --client celeba --model resnet18

# Train on VGGFace2 with MobileNetV2
python train_local.py --client vggface2 --model mobilenetv2

# Custom hyperparameters
python train_local.py --client celeba --model resnet18 --epochs 30 --lr 0.0001 --batch-size 64
```

**Resume Training:**
```bash
# Resume from latest checkpoint (continues from last saved epoch)
python train_local.py --client celeba --model resnet18 --resume latest

# Resume from best checkpoint (continues from best validation accuracy)
python train_local.py --client celeba --model resnet18 --resume best
```

**Disable MLflow:**
```bash
python train_local.py --client celeba --model resnet18 --no-mlflow
```

### 3. Train Centralized Model (Optional)
```bash
python centralized/train_global.py --model resnet18
```

### 4. Run Federated Learning
```bash
# FedAvg
python run_federated.py --method fedavg --rounds 50

# FedProx
python run_federated.py --method fedprox --rounds 50
```

### 5. Evaluate Models
```bash
# Compare all models on a specific client
python evaluate.py --mode single --client celeba --model resnet18

# Generate comprehensive report
python evaluate.py --mode all
```

## Features

### Data Processing
- **Stratified Sampling**: Balanced class distribution across train/val/test splits
- **Class-Weighted Loss**: Automatic compensation for any remaining class imbalance
- **Image Preprocessing**: Resize to 128×128, normalization, augmentation (flip, rotation, color jitter)

### Training & Optimization
- **GPU Support**: Automatic CUDA detection and utilization
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Early Stopping**: Patience-based stopping to prevent overfitting
- **Complete Checkpointing**: Saves model, optimizer, scheduler, and epoch state
- **Training Resumption**: Resume from latest or best checkpoint with `--resume` flag

### Experiment Tracking
- **MLflow Integration**: SQLite backend for efficient experiment tracking
- **Comprehensive Logging**: Train/val loss and accuracy logged every epoch
- **Visualization**: Training curves, confusion matrices, ROC curves, t-SNE/PCA embeddings
- **Metrics**: Accuracy (6 decimal precision), Precision, Recall, F1-Score, ROC-AUC

### Models & Datasets
- **Dataset Support**: CelebA, VGGFace2
- **Models**: Custom CNN, ResNet-18, MobileNetV2 (all train from scratch)
- **Training Modes**: Local, Centralized, Federated (FedAvg, FedProx)

## Configuration

Edit `config.py` to customize:

### Image Processing
- `IMG_SIZE`: Image dimensions (default: 128×128)
- `NORMALIZE_MEAN/STD`: Normalization parameters
- `AUGMENTATION_CONFIG`: Data augmentation settings

### Training Hyperparameters
- `LOCAL_EPOCHS`: Number of epochs (default: 20)
- `LOCAL_BATCH_SIZE`: Batch size (default: 32)
- `LOCAL_LEARNING_RATE`: Learning rate (default: 0.0001)
- `EARLY_STOPPING_PATIENCE`: Patience epochs (default: 10)

### Federated Learning
- `FED_ROUNDS`: Number of federated rounds
- `FED_EPOCHS_PER_ROUND`: Local epochs per round
- `FED_CLIENT_FRACTION`: Fraction of clients per round
- `FEDPROX_MU`: Proximal term coefficient

### MLflow Tracking
- `MLFLOW_TRACKING_URI`: Database URI (default: `sqlite:///mlflow.db`)
- `MLFLOW_EXPERIMENT_PREFIX`: Experiment naming prefix
- `MLFLOW_ENABLE`: Enable/disable tracking

### Paths
- `DATA_PATHS`: Dataset locations
- `CHECKPOINT_DIR`: Checkpoint storage
- `PLOTS_DIR`: Visualization output

## Checkpoints

The system automatically saves two types of checkpoints:

1. **`best_model.pth`**: Saved when validation accuracy improves
   - Contains: model weights, best metrics, training history
   - Use for final evaluation

2. **`latest_checkpoint.pth`**: Saved after every epoch
   - Contains: model, optimizer, scheduler, epoch number
   - Use for resuming interrupted training

### Checkpoint Contents
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Current epoch number
- Best validation accuracy and loss
- Complete metrics history

## MLflow Tracking

View experiments in MLflow UI:
```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Logged Information
- **Parameters**: Model architecture, hyperparameters, dataset info
- **Metrics**: Train/val loss and accuracy per epoch, test metrics
- **Artifacts**: Training curves, confusion matrices, ROC curves, embeddings, metrics JSON
- **Models**: Registered models with version tracking
