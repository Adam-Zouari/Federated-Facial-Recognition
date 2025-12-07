# MLflow Integration Guide

## Overview

MLflow has been integrated throughout the facial recognition federated learning project to track experiments, log metrics, parameters, models, and visualizations. This guide explains how to use MLflow effectively with the project.

## What is MLflow?

MLflow is an open-source platform for managing the machine learning lifecycle, including:
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Store and version models
- **Model Deployment**: Deploy models to various platforms
- **Reproducibility**: Track all aspects of model training

## Setup

### Installation

MLflow is already included in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Configuration

MLflow settings are configured in `config.py`:
```python
MLFLOW_TRACKING_URI = './mlruns'          # Where MLflow stores data
MLFLOW_EXPERIMENT_PREFIX = 'facial_recognition'  # Experiment name prefix
MLFLOW_ENABLE = True                      # Enable/disable MLflow globally
```

## Usage

### Viewing the MLflow UI

To view tracked experiments in the MLflow UI:

```bash
mlflow ui
```

Then open your browser to `http://localhost:5000`

### Training with MLflow

All training scripts now support MLflow tracking by default:

#### 1. Local Training
```bash
# Train with MLflow (default)
python train_local.py --client celeba --model resnet18

# Train without MLflow
python train_local.py --client celeba --model resnet18 --no-mlflow
```

**What gets logged:**
- Parameters: client name, model type, epochs, learning rate, batch size, dataset sizes
- Metrics per epoch: train_loss, train_acc, val_loss, val_acc, learning_rate
- Final test metrics: test_accuracy, test_precision, test_recall, test_f1, test_roc_auc
- Artifacts: training curves, confusion matrix, ROC curve, t-SNE plots, PCA plots
- Model: Best model saved to MLflow model registry

#### 2. Centralized Training
```bash
# Train with MLflow (default)
python centralized/train_global.py --model resnet18

# Train without MLflow
python centralized/train_global.py --model resnet18 --no-mlflow
```

**What gets logged:**
- Parameters: model type, epochs, learning rate, max identities, dataset sizes
- Metrics per epoch: train_loss, train_acc, val_loss, val_acc
- Final test metrics: test_accuracy, test_precision, test_recall, test_f1
- Artifacts: training curves, confusion matrix, embeddings visualization
- Model: Global model saved to registry

#### 3. Federated Learning
```bash
# Train with MLflow (default)
python run_federated.py --method fedavg --model resnet18

# Train without MLflow
python run_federated.py --method fedavg --model resnet18 --no-mlflow
```

**What gets logged:**
- Parameters: method (fedavg/fedprox), model, rounds, epochs per round, client info
- Metrics per round: round_train_loss, round_train_acc, round_test_loss, round_test_acc
- Client-specific test metrics: celeba_test_accuracy, vggface2_test_accuracy, etc.
- Artifacts: convergence plots, confusion matrices per client, client comparison
- Model: Global federated model saved to registry

#### 4. Model Evaluation
```bash
# Evaluate with MLflow (default)
python evaluate.py --mode single --client celeba --model resnet18

# Evaluate without MLflow
python evaluate.py --mode single --client celeba --model resnet18 --no-mlflow
```

**What gets logged:**
- Parameters: client, model, evaluation mode
- Metrics: Comparison of local, centralized, fedavg, fedprox models
- Artifacts: ROC comparison plots, comparison JSON files

## MLflow Experiments Organization

MLflow creates separate experiments for each training type:

1. **`facial_recognition_local_celeba`**: CelebA local training
2. **`facial_recognition_local_vggface2`**: VGGFace2 local training
3. **`facial_recognition_centralized`**: Centralized training
4. **`facial_recognition_federated_fedavg`**: FedAvg federated learning
5. **`facial_recognition_federated_fedprox`**: FedProx federated learning
6. **`facial_recognition_evaluation`**: Model evaluation and comparison

## Exploring Results in MLflow UI

### 1. Comparing Experiments

In the MLflow UI:
1. Select an experiment from the left sidebar
2. Check the runs you want to compare
3. Click "Compare" to see side-by-side metrics
4. Use the parallel coordinates plot to visualize parameter-metric relationships

### 2. Viewing Metrics

For each run:
- Click on a run to see details
- View metrics charts showing training progress
- Compare metrics across epochs/rounds
- Export metrics data for further analysis

### 3. Accessing Artifacts

Each run includes:
- **Plots**: Training curves, confusion matrices, ROC curves, t-SNE visualizations
- **Models**: Saved PyTorch models (can be loaded for inference)
- **Data**: JSON files with detailed metrics

### 4. Model Registry

Registered models can be accessed via:
```python
import mlflow.pytorch

# Load a registered model
model = mlflow.pytorch.load_model("models:/celeba_resnet18_local/1")
```

## Advanced Usage

### Programmatic Access

```python
from utils.mlflow_utils import setup_mlflow, start_run, log_params, log_metrics

# Setup experiment
setup_mlflow("my_experiment")

# Start a run
with start_run(run_name="my_run"):
    # Log parameters
    log_params({
        'learning_rate': 0.001,
        'batch_size': 32
    })
    
    # Log metrics (can specify step for time-series)
    log_metrics({'loss': 0.5, 'accuracy': 0.85}, step=1)
```

### Nested Runs

For federated learning, the main run tracks overall progress, while client updates could use nested runs:

```python
from utils.mlflow_utils import start_run, create_nested_run

with start_run(run_name="federated_training"):
    # Main federated run
    for round_num in range(10):
        # Server aggregation metrics logged here
        log_metrics({'round_loss': 0.3}, step=round_num)
```

### Custom Tracking URI

To use a remote MLflow server:

1. Update `config.py`:
```python
MLFLOW_TRACKING_URI = 'http://your-mlflow-server:5000'
```

2. Or set environment variable:
```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

## Disabling MLflow

To completely disable MLflow tracking:

1. **Globally**: Set in `config.py`:
```python
MLFLOW_ENABLE = False
```

2. **Per-script**: Use `--no-mlflow` flag:
```bash
python train_local.py --client celeba --model resnet18 --no-mlflow
```

## Best Practices

1. **Naming Conventions**: Use descriptive run names that include model type and key parameters
2. **Tagging**: Add tags to runs for easy filtering (can be added programmatically)
3. **Regular Cleanup**: Delete old experimental runs to keep the UI clean
4. **Backup**: Regularly backup the `mlruns` directory
5. **Remote Tracking**: Use a remote tracking server for team collaboration

## Troubleshooting

### Issue: MLflow UI not showing experiments
**Solution**: Ensure you're in the correct directory when running `mlflow ui`, or specify the tracking URI:
```bash
mlflow ui --backend-store-uri ./mlruns
```

### Issue: Import errors
**Solution**: Make sure mlflow is installed:
```bash
pip install mlflow>=2.9.0
```

### Issue: Models not logging
**Solution**: Check that `MLFLOW_ENABLE = True` in `config.py` and you're not using `--no-mlflow`

### Issue: Too much disk space usage
**Solution**: MLflow stores all artifacts locally. Consider:
- Using artifact storage backends (S3, Azure Blob, etc.)
- Regularly cleaning old experiments
- Reducing artifact logging (modify plotting functions to not save all figures)

## Example Workflow

```bash
# 1. Start MLflow UI in separate terminal
mlflow ui

# 2. Train local models
python train_local.py --client celeba --model resnet18 --epochs 20
python train_local.py --client vggface2 --model resnet18 --epochs 20

# 3. Train centralized model
python centralized/train_global.py --model resnet18 --epochs 30

# 4. Train federated models
python run_federated.py --method fedavg --model resnet18 --rounds 50
python run_federated.py --method fedprox --model resnet18 --rounds 50

# 5. Compare all models
python evaluate.py --mode single --client celeba --model resnet18

# 6. View results in browser at http://localhost:5000
```

## Integration Details

### Modified Files

All training and evaluation scripts have been updated:
- `train_local.py`: Local training with MLflow
- `centralized/train_global.py`: Centralized training with MLflow
- `run_federated.py`: Federated training orchestration with MLflow
- `federated/server.py`: Server-side metric logging
- `evaluate.py`: Model comparison with MLflow

### New Files

- `utils/mlflow_utils.py`: Helper functions for MLflow operations
- `MLFLOW_GUIDE.md`: This guide

### Configuration

- `config.py`: Added MLflow settings
- `requirements.txt`: Added mlflow>=2.9.0

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
