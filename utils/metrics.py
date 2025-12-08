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
from itertools import combinations
import random


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
            'val_auc': [],
            'val_eer': [],
            'lr': []
        }
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
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


def extract_embeddings(model, data_loader, device=None):
    """
    Extract embeddings from a model for all samples in the dataset.
    
    Args:
        model: PyTorch model with return_embedding support
        data_loader: DataLoader for the dataset
        device: Device to use
        
    Returns:
        embeddings: numpy array of shape (N, embedding_dim)
        labels: numpy array of shape (N,)
    """
    device = device or config.DEVICE
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device)
            
            # Get embeddings from model
            _, embeddings = model(images, return_embedding=True)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    
    return embeddings, labels


def cosine_similarity(emb1, emb2):
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding (can be 1D or 2D array)
        emb2: Second embedding (can be 1D or 2D array)
        
    Returns:
        Cosine similarity score(s)
    """
    # Normalize embeddings
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=-1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=-1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    return np.sum(emb1_norm * emb2_norm, axis=-1)


def create_verification_pairs(embeddings, labels, num_pairs=None):
    """
    Create positive and negative pairs for verification.
    
    Args:
        embeddings: Embeddings array (N, embedding_dim)
        labels: Labels array (N,)
        num_pairs: Number of pairs to create (None = use all possible positive pairs)
        
    Returns:
        pairs: List of (emb1, emb2) tuples
        pair_labels: List of labels (1 for positive, 0 for negative)
        similarities: List of cosine similarities
    """
    # Group embeddings by identity
    identity_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in identity_to_indices:
            identity_to_indices[label] = []
        identity_to_indices[label].append(idx)
    
    positive_pairs = []
    negative_pairs = []
    
    # Create positive pairs (same identity)
    for identity, indices in identity_to_indices.items():
        if len(indices) >= 2:
            # Create all combinations of pairs for this identity
            for idx1, idx2 in combinations(indices, 2):
                positive_pairs.append((embeddings[idx1], embeddings[idx2]))
    
    # Limit number of positive pairs if specified
    if num_pairs is not None and len(positive_pairs) > num_pairs:
        positive_pairs = random.sample(positive_pairs, num_pairs)
    
    num_positive = len(positive_pairs)
    
    # Create equal number of negative pairs (different identities)
    identities = list(identity_to_indices.keys())
    attempts = 0
    max_attempts = num_positive * 10  # Prevent infinite loop
    
    while len(negative_pairs) < num_positive and attempts < max_attempts:
        # Randomly select two different identities
        id1, id2 = random.sample(identities, 2)
        
        # Randomly select one sample from each identity
        idx1 = random.choice(identity_to_indices[id1])
        idx2 = random.choice(identity_to_indices[id2])
        
        negative_pairs.append((embeddings[idx1], embeddings[idx2]))
        attempts += 1
    
    # Combine pairs and create labels
    all_pairs = positive_pairs + negative_pairs
    pair_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    
    # Compute similarities
    similarities = []
    for emb1, emb2 in all_pairs:
        sim = cosine_similarity(emb1, emb2)
        similarities.append(sim)
    
    return all_pairs, np.array(pair_labels), np.array(similarities)


def compute_eer(labels, scores):
    """
    Compute Equal Error Rate (EER).
    
    Args:
        labels: True labels (1 for positive, 0 for negative)
        scores: Similarity scores
        
    Returns:
        eer: Equal Error Rate
        threshold: Threshold at EER
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find the threshold where FPR and FNR are closest
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    threshold = thresholds[eer_idx]
    
    return eer, threshold


def evaluate_verification(model, data_loader, device=None, num_pairs=None):
    """
    Evaluate model using embedding-based verification.
    
    Args:
        model: PyTorch model with return_embedding support
        data_loader: DataLoader for evaluation
        device: Device to use
        num_pairs: Number of pairs to create (None = use all possible)
        
    Returns:
        Dictionary with AUC, EER, and other verification metrics
    """
    device = device or config.DEVICE
    
    # Extract embeddings
    embeddings, labels = extract_embeddings(model, data_loader, device)
    
    # Create verification pairs
    pairs, pair_labels, similarities = create_verification_pairs(
        embeddings, labels, num_pairs=num_pairs
    )
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(pair_labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Compute EER
    eer, eer_threshold = compute_eer(pair_labels, similarities)
    
    metrics = {
        'auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'num_positive_pairs': np.sum(pair_labels == 1),
        'num_negative_pairs': np.sum(pair_labels == 0),
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'similarities': similarities,
        'pair_labels': pair_labels
    }
    
    return metrics