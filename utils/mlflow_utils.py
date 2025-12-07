"""
MLflow utilities for experiment tracking and model logging
"""
import mlflow
import mlflow.pytorch
import torch
import os
from typing import Dict, Any, Optional
from pathlib import Path


def setup_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None
):
    """
    Setup MLflow experiment
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI (default: ./mlruns)
        artifact_location: Path to store artifacts
    
    Returns:
        experiment_id: MLflow experiment ID
    """
    if tracking_uri is None:
        tracking_uri = "./mlruns"
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        if artifact_location:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_params(params: Dict[str, Any]):
    """
    Log parameters to MLflow
    
    Args:
        params: Dictionary of parameters to log
    """
    for key, value in params.items():
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            print(f"Warning: Could not log parameter {key}: {e}")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Log metrics to MLflow
    
    Args:
        metrics: Dictionary of metrics to log
        step: Step number (e.g., epoch, round)
    """
    for key, value in metrics.items():
        try:
            if step is not None:
                mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metric(key, value)
        except Exception as e:
            print(f"Warning: Could not log metric {key}: {e}")


def log_model(
    model: torch.nn.Module,
    artifact_path: str,
    registered_model_name: Optional[str] = None,
    conda_env: Optional[Dict] = None,
    code_paths: Optional[list] = None
):
    """
    Log PyTorch model to MLflow
    
    Args:
        model: PyTorch model to log
        artifact_path: Path within run's artifact directory
        registered_model_name: Name to register model under
        conda_env: Conda environment specification
        code_paths: List of code files to include
    """
    try:
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
            code_paths=code_paths
        )
    except Exception as e:
        print(f"Warning: Could not log model: {e}")


def log_artifacts(local_dir: str, artifact_path: Optional[str] = None):
    """
    Log local directory as artifacts
    
    Args:
        local_dir: Local directory to log
        artifact_path: Directory in artifact store (optional)
    """
    try:
        if artifact_path:
            mlflow.log_artifacts(local_dir, artifact_path)
        else:
            mlflow.log_artifacts(local_dir)
    except Exception as e:
        print(f"Warning: Could not log artifacts from {local_dir}: {e}")


def log_figure(fig, artifact_file: str):
    """
    Log matplotlib figure to MLflow
    
    Args:
        fig: Matplotlib figure
        artifact_file: Filename for the artifact
    """
    try:
        mlflow.log_figure(fig, artifact_file)
    except Exception as e:
        print(f"Warning: Could not log figure {artifact_file}: {e}")


def start_run(run_name: Optional[str] = None, nested: bool = False, tags: Optional[Dict[str, str]] = None):
    """
    Start MLflow run with context manager support
    
    Args:
        run_name: Name for the run
        nested: Whether this is a nested run
        tags: Dictionary of tags to add to the run
    
    Returns:
        MLflow active run context manager
    """
    return mlflow.start_run(run_name=run_name, nested=nested, tags=tags)


def end_run():
    """End the current MLflow run"""
    mlflow.end_run()


def log_checkpoint(model: torch.nn.Module, checkpoint_path: str, epoch: int):
    """
    Save and log model checkpoint
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch number
    """
    try:
        # Create checkpoint directory if needed
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, checkpoint_path)
        
        # Log to MLflow
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
    except Exception as e:
        print(f"Warning: Could not log checkpoint: {e}")


def log_dataset_info(dataset_name: str, dataset_size: int, num_classes: int, split: str = "train"):
    """
    Log dataset information as parameters
    
    Args:
        dataset_name: Name of the dataset
        dataset_size: Number of samples
        num_classes: Number of classes
        split: Dataset split (train/val/test)
    """
    params = {
        f"{split}_dataset": dataset_name,
        f"{split}_size": dataset_size,
        f"{split}_num_classes": num_classes
    }
    log_params(params)


def create_nested_run(run_name: str, parent_run_id: Optional[str] = None):
    """
    Create a nested run for hierarchical tracking (e.g., federated clients)
    
    Args:
        run_name: Name for the nested run
        parent_run_id: Parent run ID (uses current run if None)
    
    Returns:
        Nested run context manager
    """
    return mlflow.start_run(run_name=run_name, nested=True)


def log_text(text: str, artifact_file: str):
    """
    Log text content as artifact
    
    Args:
        text: Text content to log
        artifact_file: Filename for the artifact
    """
    try:
        mlflow.log_text(text, artifact_file)
    except Exception as e:
        print(f"Warning: Could not log text to {artifact_file}: {e}")


def log_dict(dictionary: Dict, artifact_file: str):
    """
    Log dictionary as JSON artifact
    
    Args:
        dictionary: Dictionary to log
        artifact_file: Filename for the artifact
    """
    try:
        mlflow.log_dict(dictionary, artifact_file)
    except Exception as e:
        print(f"Warning: Could not log dict to {artifact_file}: {e}")
