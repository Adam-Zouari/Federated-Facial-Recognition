# Federated Learning for Facial Recognition

This project implements a complete federated learning pipeline for facial recognition using two datasets: CelebA and VGGFace2. Features include stratified data sampling, class-weighted loss, checkpoint resumption, and MLflow tracking with SQLite backend.

---

## 📱 Android Face ID App

An Android app is included for on-device face enrollment and verification using your trained models.

- **Features:**
  - Face enrollment (5 poses)
  - Real-time face verification
  - Local face embedding database
  - Google ML Kit for face detection
  - PyTorch Mobile for model inference

**Setup:**
1. Convert your PyTorch model to TorchScript:
   ```bash
   python convert_model_to_mobile.py --model checkpoints/local/vggface2_weak/best_model.pth --output Android_App/app/src/main/assets/model.pt
   ```
2. Open `Android_App` in Android Studio and build/run on a device or emulator.
3. See [Android_App/README.md](Android_App/README.md) for full instructions and troubleshooting.

---

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

**Data Augmentation Control:**
```bash
# No augmentation (only resize + normalize)
python train_local.py --client vggface2 --model mobilenetv2 --augmentation none

# Weak augmentation (flip + small rotation + mild color jitter)
python train_local.py --client vggface2 --model mobilenetv2 --augmentation weak

# Strong augmentation (all transforms: flip, rotation, affine, perspective, blur, erasing)
python train_local.py --client vggface2 --model mobilenetv2 --augmentation strong  # default
```

**Resume Training:**
```bash
# Resume from latest checkpoint (continues from last saved epoch)
python train_local.py --client celeba --model resnet18 --resume latest

# Resume from best checkpoint (continues from best validation AUC)
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

**Multi-Dataset Federated Learning (CelebA + VGGFace2):**
```bash
# FedAvg
python run_federated.py --method fedavg --rounds 50

# FedProx
python run_federated.py --method fedprox --rounds 50
```

**VGGFace2-Only Federated Learning (Multiple Clients):**

This is a dedicated federated learning setup for VGGFace2 with multiple clients and flexible data partitioning.

**Quick Start (Interactive):**
```bash
python run_vggface2_examples.py
```

**Quick Test (5 clients, ~1-2 hours):**
```bash
python run_vggface2_federated.py --num-clients 5 --partition iid --model mobilenetv2 --rounds 30
```

**Recommended Configuration (10 clients, ~2-3 hours):**
```bash
python run_vggface2_federated.py --num-clients 10 --partition iid --model resnet18 --rounds 50
```

**Realistic Non-IID Scenario (10 clients, ~3-4 hours):**
```bash
python run_vggface2_federated.py --num-clients 10 --partition non-iid --alpha 0.5 --model resnet18 --rounds 50
```

**Advanced Research Setup (20 clients, ~6-8 hours):**
```bash
python run_vggface2_federated.py \
    --num-clients 20 \
    --partition non-iid \
    --alpha 0.3 \
    --method fedprox \
    --model resnet18 \
    --rounds 100 \
    --client-fraction 0.5 \
    --augmentation strong
```

**VGGFace2 Federated Parameters:**
- `--num-clients`: Number of federated clients (5, 10, or 20)
- `--partition`: Data partition strategy (iid or non-iid)
- `--alpha`: Dirichlet concentration for non-iid (0.1=highly skewed, 0.5=moderate, 10=balanced)
- `--method`: Federated algorithm (fedavg or fedprox)
- `--client-fraction`: Fraction of clients to sample per round (0.0-1.0)

**📚 For detailed VGGFace2 federated learning documentation, see:**
- `VGGFACE2_FEDERATED_GUIDE.md` - Comprehensive usage guide
- `VGGFACE2_FEDERATED_SUMMARY.md` - Implementation summary and quick reference

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
- **Class-Weighted Loss**: Automatic compensation for class imbalance with caching
- **Configurable Augmentation**: Three levels (none/weak/strong) for controlled experiments
  - **None**: Only resize + normalize (baseline)
  - **Weak**: Horizontal flip + rotation (±10°) + mild ColorJitter
  - **Strong**: All transforms including affine, perspective, GaussianBlur, RandomErasing
- **Image Preprocessing**: Resize to 128×128, normalization

### Training & Optimization
- **GPU Support**: Automatic CUDA detection and utilization
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Early Stopping**: Patience-based stopping using AUC/EER metrics
- **Complete Checkpointing**: Saves model, optimizer, scheduler, and epoch state
- **Training Resumption**: Resume from latest or best checkpoint with `--resume` flag
- **Verification Metrics**: AUC and EER (Equal Error Rate) for face verification evaluation

### Experiment Tracking
- **MLflow Integration**: SQLite backend for efficient experiment tracking
- **Comprehensive Logging**: Logs train loss/accuracy, validation AUC/EER, learning rate per epoch
- **Visualization**: Training curves, ROC curves for verification
- **Verification Metrics**: AUC (Area Under ROC), EER (Equal Error Rate), positive/negative pair counts
- **Organized Outputs**: Separate directories per client and augmentation level
  - Checkpoints: `checkpoints/local/{client}_{augmentation}/`
  - Plots: `plots/local/{client}_{augmentation}/`

### Models & Datasets
- **Dataset Support**: CelebA, VGGFace2
- **Models**: Custom CNN, ResNet-18, MobileNetV2 (all train from scratch)
- **Training Modes**: Local, Centralized, Federated (FedAvg, FedProx)

## Configuration

Edit `config.py` to customize:

### Image Processing
- `IMG_SIZE`: Image dimensions (default: 128×128)
- `NORMALIZE_MEAN/STD`: Normalization parameters
- `AUGMENTATION_NONE/WEAK/STRONG`: Three augmentation levels for controlled experiments

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

### VGGFace2 Federated Learning
- `VGGFACE2_FED_NUM_CLIENTS`: Number of clients (default: 10)
- `VGGFACE2_FED_PARTITION_STRATEGY`: Partition strategy (iid or non-iid)
- `VGGFACE2_FED_ALPHA`: Dirichlet concentration for non-iid (default: 0.5)

### MLflow Tracking
- `MLFLOW_TRACKING_URI`: Database URI (default: `sqlite:///mlflow.db`)
- `MLFLOW_EXPERIMENT_PREFIX`: Experiment naming prefix
- `MLFLOW_ENABLE`: Enable/disable tracking

### Paths
- `DATA_PATHS`: Dataset locations
- `CHECKPOINT_DIR`: Checkpoint storage
- `PLOTS_DIR`: Visualization output

## Checkpoints

The system automatically saves checkpoints organized by client and augmentation level:

**Directory Structure:**
- `checkpoints/local/{client}_{augmentation}/best_model.pth`
- `checkpoints/local/{client}_{augmentation}/latest_checkpoint.pth`

Examples:
- `checkpoints/local/vggface2_none/best_model.pth`
- `checkpoints/local/vggface2_weak/best_model.pth`
- `checkpoints/local/vggface2_strong/best_model.pth`

**Checkpoint Types:**

1. **`best_model.pth`**: Saved when validation AUC improves
   - Contains: model weights, best AUC/EER, training history
   - Use for final evaluation

2. **`latest_checkpoint.pth`**: Saved after every epoch
   - Contains: model, optimizer, scheduler, epoch number
   - Use for resuming interrupted training

### Checkpoint Contents
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Current epoch number
- Best validation AUC and EER
- Complete metrics history

### Class Weights Cache
Class weights are cached separately and shared across augmentation levels:
- `checkpoints/class_weights/{client}_class_weights.pth`
- Computed once per dataset, reused for all augmentation experiments

## MLflow Tracking

View experiments in MLflow UI:
```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Logged Information
- **Parameters**: Model architecture, hyperparameters, augmentation level, dataset info
- **Metrics**: Train loss/accuracy, validation AUC/EER, learning rate per epoch, test verification metrics
- **Artifacts**: Training curves, ROC curves, verification metrics JSON
- **Models**: Registered models with version tracking

## Outputs

All training outputs are organized by client and augmentation level:

### Plots Directory
```
plots/local/{client}_{augmentation}/
├── {model}_training_curves.png      # Loss and AUC/EER over epochs
├── {model}_verification_roc_curve.png  # ROC curve for face verification
└── {model}_verification_metrics.json   # AUC, EER, thresholds, pair counts
```

Examples:
- `plots/local/vggface2_none/mobilenetv2_training_curves.png`
- `plots/local/vggface2_weak/mobilenetv2_verification_roc_curve.png`
- `plots/local/vggface2_strong/mobilenetv2_verification_metrics.json`
