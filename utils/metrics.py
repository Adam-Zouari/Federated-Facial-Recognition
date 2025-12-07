"""
Metrics computation and evaluation utilities.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import config


def compute_metrics(y_true, y_pred, y_scores=None, num_classes=None):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities (for ROC)
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Multi-class metrics
    avg_method = 'weighted' if num_classes and num_classes > 2 else 'binary'
    metrics['precision'] = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=avg_method, zero_division=0)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # ROC curve data (if scores provided)
    if y_scores is not None and num_classes is not None:
        try:
            if num_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
                metrics['roc_auc'] = auc(fpr, tpr)
                metrics['fpr'] = fpr
                metrics['tpr'] = tpr
            else:
                # Multi-class ROC
                y_true_bin = label_binarize(y_true, classes=range(num_classes))
                metrics['roc_auc_per_class'] = {}
                metrics['fpr_per_class'] = {}
                metrics['tpr_per_class'] = {}
                
                for i in range(num_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                    metrics['roc_auc_per_class'][i] = auc(fpr, tpr)
                    metrics['fpr_per_class'][i] = fpr
                    metrics['tpr_per_class'][i] = tpr
                
                # Macro average
                metrics['roc_auc'] = np.mean(list(metrics['roc_auc_per_class'].values()))
        except Exception as e:
            print(f"Warning: Could not compute ROC curve: {e}")
    
    return metrics


def evaluate_model(model, data_loader, device=None, return_embeddings=False):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to use
        return_embeddings: Whether to return feature embeddings
        
    Returns:
        Dictionary with predictions, labels, scores, and optionally embeddings
    """
    device = device or config.DEVICE
    model.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    all_embeddings = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Handle different output types
            if isinstance(outputs, tuple):
                # If model returns (logits, embeddings)
                logits, embeddings = outputs
                if return_embeddings:
                    all_embeddings.append(embeddings.cpu().numpy())
            else:
                logits = outputs
            
            scores = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
    
    results = {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'scores': np.vstack(all_scores)
    }
    
    if return_embeddings and all_embeddings:
        results['embeddings'] = np.vstack(all_embeddings)
    
    return results


def print_metrics(metrics, title="Evaluation Metrics"):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"{'='*60}\n")


class MetricsTracker:
    """Track metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_latest(self, key):
        """Get the latest value for a metric."""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return None
    
    def get_best(self, key, mode='max'):
        """Get the best value for a metric."""
        if key in self.metrics and self.metrics[key]:
            if mode == 'max':
                return max(self.metrics[key])
            else:
                return min(self.metrics[key])
        return None
    
    def save(self, filepath):
        """Save metrics to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, filepath):
        """Load metrics from file."""
        import json
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
