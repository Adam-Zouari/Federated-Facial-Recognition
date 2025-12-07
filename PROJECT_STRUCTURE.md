# Project Structure Documentation

## Overview

This document provides a comprehensive explanation of the project's folder structure and the role of each file in the federated learning facial recognition system.

---

## Root Directory

```
CV+FL/
├── config.py                 # Global configuration and hyperparameters
├── requirements.txt          # Python package dependencies
├── main.py                   # Main entry point (exploration and preprocessing)
├── train_local.py           # Local training script for individual clients
├── run_federated.py         # Federated learning orchestration script
├── evaluate.py              # Model evaluation and comparison script
├── README.md                # Project overview and quick start guide
├── CHANGES.md               # Change log documenting modifications
├── MLFLOW_GUIDE.md          # MLflow integration and usage guide
├── PROJECT_STRUCTURE.md     # This file - detailed structure documentation
├── data/                    # Dataset storage (not tracked in git)
├── models/                  # Model architecture definitions
├── clients/                 # Client-specific dataset handlers
├── federated/              # Federated learning implementation
├── centralized/            # Centralized training baseline
├── utils/                  # Utility functions and helpers
├── outputs/                # Training outputs (created at runtime)
├── checkpoints/            # Saved model checkpoints (created at runtime)
├── logs/                   # Training logs (created at runtime)
├── plots/                  # Generated visualizations (created at runtime)
└── mlruns/                 # MLflow tracking data (created at runtime)
```

---

## Core Configuration Files

### `config.py`
**Purpose**: Central configuration file for the entire project.

**Contents**:
- **Device Configuration**: CUDA/CPU detection
- **Image Preprocessing**: Image size (128x128), normalization parameters
- **Dataset Paths**: Paths to CelebA and VGGFace2 datasets
- **Training Hyperparameters**: 
  - Local training: epochs, batch size, learning rate
  - Centralized training: epochs, batch size, learning rate
  - Federated learning: rounds, epochs per round, client fraction
  - FedProx: proximal term coefficient (mu)
- **Data Augmentation**: Flip probability, rotation, color jitter settings
- **Output Directories**: Paths for outputs, checkpoints, logs, plots
- **MLflow Configuration**: Tracking URI, experiment prefix, enable/disable flag

**Key Variables**:
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (128, 128)
DATA_PATHS = {'celeba': './data/CelebA', 'vggface2': './data/VGGFace2'}
LOCAL_EPOCHS = 20
FED_ROUNDS = 50
MLFLOW_ENABLE = True
```

### `requirements.txt`
**Purpose**: Lists all Python package dependencies.

**Key Dependencies**:
- `torch>=2.0.0`, `torchvision>=0.15.0`: Deep learning framework
- `scikit-learn>=1.3.0`: Metrics computation
- `matplotlib>=3.7.0`, `seaborn>=0.12.0`: Visualization
- `Pillow>=10.0.0`: Image processing
- `facenet-pytorch>=2.5.3`: Face detection utilities
- `mlflow>=2.9.0`: Experiment tracking

---

## Main Entry Scripts

### `main.py`
**Purpose**: Main entry point for dataset exploration and preprocessing.

**Functionality**:
- Dataset loading and exploration
- Data preprocessing pipeline demonstration
- Visualization of sample images
- Dataset statistics reporting

**Usage**:
```bash
python main.py
```

### `train_local.py`
**Purpose**: Train models locally on individual client datasets.

**Functionality**:
- **LocalTrainer Class**: Handles single-client training
  - `train_epoch()`: One epoch of training
  - `validate()`: Validation loop
  - `train()`: Full training with early stopping
  - `save_checkpoint()` / `load_checkpoint()`: Model persistence
- **MLflow Integration**: Logs parameters, metrics, models, and plots
- **Evaluation**: Full test set evaluation with metrics and visualizations
- **Support for**: CelebA, VGGFace2 clients

**Usage**:
```bash
python train_local.py --client celeba --model resnet18 --epochs 20 --lr 0.001
```

**Arguments**:
- `--client`: Dataset to use (celeba, vggface2)
- `--model`: Architecture (custom_cnn, resnet18, mobilenetv2)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--batch-size`: Batch size
- `--no-mlflow`: Disable MLflow tracking

### `run_federated.py`
**Purpose**: Orchestrate federated learning across multiple clients.

**Functionality**:
- **Client Creation**: Initializes FederatedClient or FedProxClient for each dataset
- **Server Initialization**: Creates FederatedServer with chosen method
- **Federated Training**: Coordinates multiple rounds of:
  1. Client selection
  2. Local training
  3. Parameter aggregation
  4. Global model update
- **Evaluation**: Tests global model on each client's test set
- **Visualization**: Convergence plots, confusion matrices, client comparisons
- **MLflow Integration**: Tracks federated rounds and aggregated metrics

**Usage**:
```bash
python run_federated.py --method fedavg --model resnet18 --rounds 50
```

**Arguments**:
- `--method`: Federated algorithm (fedavg, fedprox)
- `--model`: Architecture
- `--rounds`: Number of communication rounds
- `--epochs-per-round`: Local epochs per round
- `--lr`: Learning rate
- `--client-fraction`: Fraction of clients per round
- `--no-mlflow`: Disable MLflow tracking

### `evaluate.py`
**Purpose**: Comprehensive model evaluation and comparison.

**Functionality**:
- **Model Loading**: Load trained models from checkpoints
  - `load_local_model()`: Client-specific local models
  - `load_centralized_model()`: Centralized baseline
  - `load_federated_model()`: FedAvg/FedProx global models
- **Evaluation**: Test all model types on client test sets
- **Comparison**: Side-by-side metrics comparison
- **Visualization**: ROC curves, performance charts
- **Report Generation**: JSON files with detailed results

**Usage**:
```bash
# Single client evaluation
python evaluate.py --mode single --client celeba --model resnet18

# Full report for all clients and models
python evaluate.py --mode all
```

---

## Model Architectures (`models/`)

### `models/__init__.py`
**Purpose**: Model factory and imports.

**Exports**:
- `create_model(model_name, num_classes)`: Factory function to create models
- All model classes: CustomCNN, ResNet18Classifier, MobileNetV2Classifier

### `models/cnn_custom.py`
**Purpose**: Custom CNN architecture designed for facial recognition.

**Architecture**:
```
Input (3, 128, 128)
    ↓
Conv Block 1: Conv(64) → BatchNorm → ReLU → MaxPool
Conv Block 2: Conv(128) → BatchNorm → ReLU → MaxPool
Conv Block 3: Conv(256) → BatchNorm → ReLU → MaxPool
Conv Block 4: Conv(512) → BatchNorm → ReLU → MaxPool
    ↓
Flatten
    ↓
FC Layer: 512 → 128 (embedding)
Dropout(0.5)
    ↓
FC Layer: 128 → num_classes (logits)
```

**Features**:
- 4 convolutional blocks with increasing filters
- Batch normalization for training stability
- Dropout for regularization
- 128-dimensional embedding space
- Returns embeddings for visualization

### `models/resnet.py`
**Purpose**: ResNet-18 adapted for facial recognition.

**Architecture**:
- **Base**: Pretrained ResNet-18 backbone (now set to `pretrained=False`)
- **Modification**: Replaces final FC layer with custom head
- **Embedding**: 512 → 128 dimensional embedding
- **Output**: num_classes logits

**Features**:
- Skip connections (residual blocks)
- Deep architecture (18 layers)
- Effective feature extraction
- Transfer learning capability (when pretrained=True)

**Code Structure**:
```python
self.resnet = models.resnet18(pretrained=False)
self.resnet.fc = nn.Linear(512, 128)  # Embedding layer
self.classifier = nn.Linear(128, num_classes)  # Classification layer
```

### `models/mobilenet.py`
**Purpose**: Lightweight MobileNetV2 for efficient inference.

**Architecture**:
- **Base**: MobileNetV2 with depthwise separable convolutions
- **Embedding**: 1280 → 128 dimensions
- **Output**: num_classes logits

**Features**:
- Depthwise separable convolutions (efficient)
- Inverted residual blocks
- Smaller model size (~3.5M parameters)
- Faster inference than ResNet
- Good for mobile/edge deployment

---

## Client Dataset Handlers (`clients/`)

### `clients/__init__.py`
**Purpose**: Client factory and dataset exports.

**Functions**:
- `get_client_data(client_name, batch_size, max_identities)`: Creates data loaders for a client

**Exports**: CelebADataset, VGGFace2Dataset

### `clients/celebA_client.py`
**Purpose**: CelebA dataset handler with official partitions.

**CelebADataset Class**:
- **Initialization**: Loads dataset from specified root directory
- **Partition Loading**: Uses `list_eval_partition.txt` for train/val/test splits
  - 0 = train, 1 = validation, 2 = test
- **Identity Mapping**: Uses `Anno/identity_CelebA.txt` for face identities
- **Label Remapping**: Converts identities to contiguous class indices
- **Identity Limiting**: Supports `max_identities` parameter for subset training

**File Structure Expected**:
```
CelebA/
├── Img/
│   └── img_align_celeba/
│       ├── 000001.jpg
│       ├── 000002.jpg
│       └── ...
├── Eval/
│   └── list_eval_partition.txt
└── Anno/
    └── identity_CelebA.txt
```

**Key Methods**:
- `__getitem__()`: Returns (image, label) tuple
- `__len__()`: Dataset size
- `get_num_classes()`: Number of unique identities

### `clients/vggface_client.py`
**Purpose**: VGGFace2 dataset handler.

**VGGFace2Dataset Class**:
- **Structure**: Expects train/ and val/ folders with person subdirectories
- **Automatic Splitting**: Creates test split from validation set
- **Identity Discovery**: Scans directories for person folders
- **Sorting**: Orders by number of images per identity
- **Limiting**: Supports `max_identities` for subset creation

**File Structure Expected**:
```
VGGFace2/
├── train/
│   ├── n000001/
│   │   ├── 0001_01.jpg
│   │   └── ...
│   ├── n000002/
│   └── ...
└── val/
    ├── n000003/
    └── ...
```

**Key Features**:
- Flexible train/val/test splitting
- Label remapping to contiguous indices
- Efficient directory scanning

---

## Federated Learning Implementation (`federated/`)

### `federated/__init__.py`
**Purpose**: Exports federated learning components.

**Exports**: FederatedClient, FedProxClient, FederatedServer, fedavg_aggregate

### `federated/client.py`
**Purpose**: Federated client implementation.

**FederatedClient Class**:
- **Attributes**:
  - `client_id`: Unique identifier
  - `model`: Local model
  - `train_loader`, `val_loader`: Data loaders
  - `optimizer`, `criterion`: Training components
- **Methods**:
  - `train(epochs)`: Local training for specified epochs
  - `get_model_params()`: Extract model parameters
  - `set_model_params()`: Load parameters from server
  - `evaluate()`: Validation evaluation

**FedProxClient Class** (inherits from FederatedClient):
- **Additional**: Proximal term in loss function
- `set_global_params()`: Store global model for proximal term
- `train()`: Modified to include proximal regularization

**Training Flow**:
1. Receive global parameters from server
2. Train locally for N epochs
3. Apply proximal term (FedProx only)
4. Return updated parameters to server

### `federated/server.py`
**Purpose**: Federated server coordinator.

**FederatedServer Class**:
- **Attributes**:
  - `global_model`: Central model
  - `method`: 'fedavg' or 'fedprox'
  - `server`: FedAvgServer or FedProxServer instance
  - `round_history`: Training history
- **Methods**:
  - `federated_training()`: Main training loop
    - Client selection
    - Local training coordination
    - Server aggregation
    - Global evaluation
  - `evaluate_global_model()`: Test on global test set
  - `save_global_model()` / `load_global_model()`: Persistence

**Federated Training Loop**:
```
FOR each round:
    1. Select fraction of clients
    2. Broadcast global model to clients
    3. Clients train locally
    4. Collect client updates
    5. Aggregate updates (FedAvg/FedProx)
    6. Update global model
    7. Evaluate and log metrics
```

### `federated/fedavg.py`
**Purpose**: FedAvg (Federated Averaging) algorithm.

**FedAvgServer Class**:
- **Method**: Weighted average of client parameters
- **Weighting**: Proportional to client dataset size
- **Formula**: `global_params = Σ(client_params_i * weight_i) / Σ(weight_i)`

**Algorithm**:
```python
def aggregate(client_params_list, client_weights):
    # Weighted averaging
    for each parameter:
        weighted_sum = 0
        for client_params, weight in zip(client_params_list, client_weights):
            weighted_sum += client_params * weight
        global_params = weighted_sum / total_weight
    return global_params
```

### `federated/fedprox.py`
**Purpose**: FedProx (Federated Proximal) algorithm.

**FedProxServer Class**:
- **Method**: FedAvg + proximal term in client training
- **Proximal Term**: `(mu/2) * ||w - w_global||^2`
- **Purpose**: Limit client drift from global model

**Key Features**:
- Aggregation same as FedAvg
- Proximal term coefficient: `mu` (default 0.01)
- Better for heterogeneous data distributions

**Proximal Loss Function**:
```python
def proximal_loss(local_params, global_params, mu):
    loss = 0
    for local_p, global_p in zip(local_params, global_params):
        loss += (mu/2) * torch.norm(local_p - global_p) ** 2
    return loss
```

---

## Centralized Training (`centralized/`)

### `centralized/__init__.py`
**Purpose**: Exports centralized training components.

### `centralized/train_global.py`
**Purpose**: Centralized training baseline (combines all datasets).

**CentralizedTrainer Class**:
- Similar to LocalTrainer but for combined dataset
- **Methods**:
  - `train_epoch()`: Training loop
  - `validate()`: Validation loop
  - `train()`: Full training with early stopping

**combine_datasets() Function**:
- Loads both CelebA and VGGFace2
- Remaps labels to avoid conflicts
- Creates combined train/val/test loaders
- Returns total number of classes

**Purpose**: Provides upper-bound baseline for federated learning comparison

**Usage**:
```bash
python centralized/train_global.py --model resnet18 --epochs 30
```

---

## Utility Functions (`utils/`)

### `utils/__init__.py`
**Purpose**: Exports utility functions.

### `utils/preprocessing.py`
**Purpose**: Image preprocessing and data augmentation.

**Functions**:
- `get_train_transforms(use_augmentation=True)`:
  - Resize to 128x128
  - Random horizontal flip (50%)
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation, hue)
  - Convert to tensor
  - Normalize (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

- `get_test_transforms()`:
  - Resize to 128x128
  - Convert to tensor
  - Normalize (same as training)

**Data Augmentation Rationale**:
- Horizontal flip: Faces can appear mirrored
- Rotation: Slight head tilts
- Color jitter: Lighting variations
- Normalization: Stabilize training

### `utils/metrics.py`
**Purpose**: Evaluation metrics and model evaluation.

**MetricsTracker Class**:
- Stores training history (losses, accuracies, learning rates)
- `update()`: Add new epoch metrics
- `get_latest()`: Retrieve latest metrics

**compute_metrics() Function**:
- **Input**: True labels, predictions, scores
- **Output**: Dictionary with:
  - Accuracy
  - Precision (macro, weighted)
  - Recall (macro, weighted)
  - F1-score (macro, weighted)
  - Confusion matrix
  - ROC curve data (FPR, TPR)
  - AUC score

**evaluate_model() Function**:
- **Input**: Model, test loader
- **Output**: Predictions, labels, scores, embeddings
- Handles batch processing
- Optional embedding extraction

**print_metrics() Function**:
- Pretty-prints evaluation results

### `utils/plotting.py`
**Purpose**: Visualization functions.

**Functions**:
- `plot_training_curves(metrics_tracker)`:
  - Loss and accuracy over epochs
  - Train vs validation comparison

- `plot_confusion_matrix(cm)`:
  - Heatmap of prediction errors
  - Normalized by true labels

- `plot_roc_curve(fpr, tpr, auc)`:
  - ROC curve visualization
  - AUC score annotation

- `plot_embeddings_tsne(embeddings, labels)`:
  - 2D t-SNE projection of embeddings
  - Color-coded by class

- `plot_embeddings_pca(embeddings, labels)`:
  - 2D PCA projection
  - Faster than t-SNE

- `plot_federated_convergence(history)`:
  - Round-by-round metrics
  - Loss and accuracy trends

- `plot_client_comparison(client_metrics)`:
  - Bar chart comparing clients
  - Multiple metrics side-by-side

- `plot_multi_roc_curves(results)`:
  - Multiple ROC curves on one plot
  - Compare different models

**All functions**:
- Save to file (PNG format)
- Return figure object for MLflow logging
- Use seaborn for styling

### `utils/mlflow_utils.py`
**Purpose**: MLflow integration helpers.

**Functions**:
- `setup_mlflow(experiment_name, tracking_uri, artifact_location)`:
  - Initialize MLflow experiment
  - Set tracking URI
  - Create or get experiment

- `log_params(params)`: Log hyperparameters
- `log_metrics(metrics, step)`: Log metrics with optional step
- `log_model(model, artifact_path, registered_model_name)`: Save PyTorch model
- `log_figure(fig, artifact_file)`: Save matplotlib figure
- `log_dict(dictionary, artifact_file)`: Save JSON data
- `log_artifacts(local_dir, artifact_path)`: Save directory
- `start_run(run_name, nested, tags)`: Context manager for runs
- `end_run()`: End current run
- `create_nested_run(run_name)`: Create child run

**Error Handling**:
- All functions wrapped in try-except
- Warnings instead of crashes
- Graceful degradation

---

## Documentation Files

### `README.md`
**Purpose**: Project overview and quick start guide.

**Contents**:
- Project description
- Features list
- Installation instructions
- Dataset setup
- Usage examples
- Expected results

### `CHANGES.md`
**Purpose**: Documents modifications made to the project.

**Contents**:
- LFW dataset removal
- Pretrained model removal
- Dataset structure adaptations
- Configuration updates

### `MLFLOW_GUIDE.md`
**Purpose**: Comprehensive MLflow usage guide.

**Contents**:
- MLflow overview
- Setup and configuration
- Usage for each training script
- Experiment organization
- UI navigation
- Advanced features
- Troubleshooting

### `PROJECT_STRUCTURE.md`
**Purpose**: This file - detailed structure documentation.

---

## Runtime-Generated Directories

### `data/`
**Purpose**: Dataset storage (user-provided).

**Expected Structure**:
```
data/
├── CelebA/
│   ├── Img/img_align_celeba/
│   ├── Eval/list_eval_partition.txt
│   └── Anno/identity_CelebA.txt
└── VGGFace2/
    ├── train/
    └── val/
```

**Note**: Not included in repository, must be downloaded separately.

### `outputs/`
**Purpose**: General training outputs.

**Created by**: Training scripts

**Contents**: Varies by experiment

### `checkpoints/`
**Purpose**: Saved model checkpoints.

**Structure**:
```
checkpoints/
├── local/
│   ├── celeba/
│   │   └── best_model.pth
│   └── vggface2/
│       └── best_model.pth
├── centralized/
│   └── best_model.pth
└── federated/
    ├── fedavg/
    │   └── resnet18_global_model.pth
    └── fedprox/
        └── resnet18_global_model.pth
```

**Checkpoint Contents**:
- `model_state_dict`: Model parameters
- `best_val_acc`: Best validation accuracy
- `metrics`: Training history
- Additional metadata

### `logs/`
**Purpose**: Training logs and tensorboard data.

**Created by**: Training scripts with logging enabled

### `plots/`
**Purpose**: Generated visualizations.

**Structure**:
```
plots/
├── local/
│   ├── celeba/
│   │   ├── resnet18_training_curves.png
│   │   ├── resnet18_confusion_matrix.png
│   │   ├── resnet18_roc_curve.png
│   │   ├── resnet18_embeddings_tsne.png
│   │   └── resnet18_embeddings_pca.png
│   └── vggface2/
│       └── ...
├── centralized/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── embeddings_tsne.png
├── federated/
│   ├── fedavg/
│   │   ├── resnet18_convergence.png
│   │   ├── resnet18_celeba_confusion_matrix.png
│   │   └── resnet18_client_comparison.png
│   └── fedprox/
│       └── ...
└── comparison/
    └── celeba_resnet18_roc_comparison.png
```

### `mlruns/`
**Purpose**: MLflow tracking data.

**Structure**:
```
mlruns/
├── .trash/                    # Deleted runs
├── 0/                        # Default experiment
├── 1/                        # facial_recognition_local_celeba
├── 2/                        # facial_recognition_local_vggface2
├── 3/                        # facial_recognition_centralized
├── 4/                        # facial_recognition_federated_fedavg
├── 5/                        # facial_recognition_federated_fedprox
└── 6/                        # facial_recognition_evaluation
```

**Each experiment contains**:
- Run directories with IDs
- Metrics (JSON/parquet)
- Parameters (YAML)
- Artifacts (models, plots, JSON)
- Metadata

**Access**: Browse via `mlflow ui` command

---

## Data Flow Diagrams

### Local Training Flow
```
CelebA/VGGFace2 Data
    ↓
get_client_data() [clients/__init__.py]
    ↓
DataLoader (train/val/test)
    ↓
create_model() [models/__init__.py]
    ↓
LocalTrainer [train_local.py]
    ↓
Training Loop (with MLflow logging)
    ↓
Checkpoints + Plots + Metrics
```

### Federated Learning Flow
```
Multiple Datasets (CelebA, VGGFace2)
    ↓
create_federated_clients() [run_federated.py]
    ↓
FederatedClient/FedProxClient [federated/client.py]
    ↓
FederatedServer [federated/server.py]
    ↓
FOR each round:
    Client Selection → Local Training → Aggregation
    ↓
Global Model Evaluation
    ↓
Checkpoints + Plots + Metrics (MLflow)
```

### Evaluation Flow
```
Trained Model Checkpoints
    ↓
load_*_model() [evaluate.py]
    ↓
evaluate_model() [utils/metrics.py]
    ↓
compute_metrics() [utils/metrics.py]
    ↓
Comparison Plots + JSON Results (MLflow)
```

---

## Key Design Patterns

### 1. Factory Pattern
- `create_model()`: Creates model instances based on string name
- `get_client_data()`: Creates data loaders based on client name

### 2. Strategy Pattern
- `FederatedServer`: Delegates to FedAvgServer or FedProxServer
- Different aggregation strategies encapsulated

### 3. Template Method
- `LocalTrainer` and `CentralizedTrainer`: Similar training loop structure
- `FederatedClient` and `FedProxClient`: Shared interface, different implementations

### 4. Singleton Pattern
- `config.py`: Global configuration accessed throughout project

### 5. Facade Pattern
- `utils/mlflow_utils.py`: Simplifies MLflow API
- `utils/plotting.py`: Unified plotting interface

---

## Configuration Flexibility

### Hyperparameter Tuning
All hyperparameters in `config.py` can be overridden via command-line arguments:

```bash
# Override local training settings
python train_local.py --client celeba --model resnet18 --epochs 50 --lr 0.0001 --batch-size 64

# Override federated settings
python run_federated.py --method fedavg --rounds 100 --epochs-per-round 10 --client-fraction 0.5
```

### Dataset Customization
- Modify `DATA_PATHS` in `config.py`
- Adjust `max_identities` parameter for subset training
- Change image size in `IMG_SIZE` (requires model adjustments)

### Model Customization
- Add new architectures to `models/` directory
- Register in `models/__init__.py` factory
- Use via `--model your_model_name`

---

## Best Practices

### File Organization
- **One concern per file**: Each module has single responsibility
- **Clear naming**: File names describe contents
- **Logical grouping**: Related files in same directory

### Code Reusability
- **Utils**: Common functions extracted to `utils/`
- **Base classes**: Shared functionality in parent classes
- **Configuration**: Centralized in `config.py`

### Experiment Tracking
- **MLflow**: All experiments automatically logged
- **Checkpoints**: Models saved at best validation performance
- **Visualizations**: Comprehensive plots for analysis

### Error Handling
- **Graceful degradation**: MLflow failures don't crash training
- **Informative messages**: Clear error messages and warnings
- **Validation**: Check data availability before training

---

## Extension Points

### Adding New Datasets
1. Create new dataset handler in `clients/`
2. Implement `__getitem__()`, `__len__()`, `get_num_classes()`
3. Add to `get_client_data()` factory
4. Update `config.DATA_PATHS`

### Adding New Models
1. Create new model file in `models/`
2. Inherit from `nn.Module`
3. Implement `forward()` with embedding return
4. Register in `create_model()` factory

### Adding New Federated Algorithms
1. Create new server in `federated/`
2. Implement `aggregate()` method
3. Optional: Create custom client class
4. Add to `FederatedServer` initialization

### Adding New Metrics
1. Add computation to `utils/metrics.py`
2. Add visualization to `utils/plotting.py`
3. Update MLflow logging in training scripts

---

## File Size and Complexity

### Large Files (>300 lines)
- `run_federated.py`: 380+ lines (orchestration logic)
- `train_local.py`: 450+ lines (comprehensive training script)
- `utils/plotting.py`: 400+ lines (multiple plotting functions)

### Critical Files (frequently modified)
- `config.py`: Central configuration
- `train_local.py`: Main training script
- `run_federated.py`: Federated orchestration

### Simple Files (<100 lines)
- All `__init__.py` files
- `federated/fedavg.py`
- `federated/fedprox.py`

---

## Summary

This project implements a complete federated learning system for facial recognition with:

- **3 Model Architectures**: Custom CNN, ResNet-18, MobileNetV2
- **2 Client Datasets**: CelebA, VGGFace2
- **3 Training Paradigms**: Local, Centralized, Federated
- **2 Federated Algorithms**: FedAvg, FedProx
- **Comprehensive Evaluation**: Metrics, visualizations, comparisons
- **Experiment Tracking**: Full MLflow integration
- **Extensible Design**: Easy to add datasets, models, algorithms

The structure prioritizes:
- **Modularity**: Clear separation of concerns
- **Reusability**: Shared utilities and base classes
- **Reproducibility**: Configuration management and experiment tracking
- **Maintainability**: Clear documentation and consistent patterns
