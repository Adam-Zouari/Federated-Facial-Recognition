# Federated Learning for Facial Recognition

This project implements a complete federated learning pipeline for facial recognition using two datasets: CelebA and VGGFace2. All models train from scratch (no pretrained weights).

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
```bash
# Train on CelebA with ResNet-18
python train_local.py --client celeba --model resnet18

# Train on VGGFace2 with MobileNetV2
python train_local.py --client vggface2 --model mobilenetv2
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

- **Dataset Support**: CelebA, VGGFace2
- **Models**: Custom CNN, ResNet-18, MobileNetV2 (all train from scratch)
- **Local Training**: Independent training for each client
- **Centralized Baseline**: Combined dataset training
- **Federated Methods**: FedAvg, FedProx
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC curves
- **Visualizations**: t-SNE embeddings, confusion matrices, training curves

## Configuration

Edit `config.py` to customize:
- Image size and preprocessing
- Training hyperparameters
- Federated learning settings
- Data paths
