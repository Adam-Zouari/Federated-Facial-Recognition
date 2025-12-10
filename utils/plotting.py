"""
Plotting and visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import config


sns.set_style('whitegrid')


def plot_training_curves(metrics_tracker, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        metrics_tracker: MetricsTracker object
        save_path: Path to save the plot
    """
    metrics = metrics_tracker.metrics
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    if metrics['train_loss']:
        axes[0].plot(metrics['train_loss'], label='Train Loss', linewidth=2)
    if metrics['val_loss']:
        axes[0].plot(metrics['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    if metrics['train_acc']:
        axes[1].plot(metrics['train_acc'], label='Train Acc', linewidth=2)
    if metrics['val_acc']:
        axes[1].plot(metrics['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.show()
    plt.close()


def plot_confusion_matrix(cm, class_names=None, save_path=None, normalize=False):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    
    # Limit display to top N classes if too many
    max_display = 20
    if cm.shape[0] > max_display:
        print(f"Too many classes ({cm.shape[0]}). Displaying top {max_display} only.")
        # Sum by row and column to find most common classes
        class_totals = cm.sum(axis=0) + cm.sum(axis=1)
        top_classes = np.argsort(class_totals)[-max_display:]
        cm = cm[np.ix_(top_classes, top_classes)]
        if class_names:
            class_names = [class_names[i] for i in top_classes]
        else:
            # Generate numeric labels for top classes
            class_names = [str(i) for i in top_classes]
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc=None, save_path=None, label=None):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        roc_auc: Area under curve
        save_path: Path to save the plot
        label: Label for the curve
    """
    plt.figure(figsize=(8, 6))
    
    if roc_auc is not None:
        label = f'{label} (AUC = {roc_auc:.3f})' if label else f'ROC curve (AUC = {roc_auc:.3f})'
    
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    
    plt.show()
    plt.close()


def plot_multi_roc_curves(metrics_dict, save_path=None):
    """
    Plot multiple ROC curves on the same plot.
    
    Args:
        metrics_dict: Dictionary of {name: metrics} pairs
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for name, metrics in metrics_dict.items():
        if 'fpr' in metrics and 'tpr' in metrics:
            roc_auc = metrics.get('roc_auc', None)
            label = f'{name} (AUC = {roc_auc:.3f})' if roc_auc else name
            plt.plot(metrics['fpr'], metrics['tpr'], linewidth=2, label=label)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    
    plt.show()
    plt.close()


def plot_embeddings_tsne(embeddings, labels, save_path=None, perplexity=30, n_components=2):
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Feature embeddings (N x D)
        labels: Labels for each embedding
        save_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        n_components: Number of dimensions (2 or 3)
    """
    print("Computing t-SNE projection...")
    
    # Sample if too many points
    max_points = 5000
    if len(embeddings) > max_points:
        indices = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Limit number of classes to display
    max_classes = 20
    if len(unique_labels) > max_classes:
        print(f"Too many classes ({len(unique_labels)}). Displaying {max_classes} most common.")
        label_counts = {label: np.sum(labels == label) for label in unique_labels}
        top_labels = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)[:max_classes]
        unique_labels = top_labels
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[color], label=f'Class {label}', alpha=0.6, s=20)
    
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title('t-SNE Visualization of Embeddings', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_embeddings_pca(embeddings, labels, save_path=None, n_components=2):
    """
    Visualize embeddings using PCA.
    
    Args:
        embeddings: Feature embeddings (N x D)
        labels: Labels for each embedding
        save_path: Path to save the plot
        n_components: Number of principal components
    """
    print("Computing PCA projection...")
    
    pca = PCA(n_components=n_components)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Limit number of classes to display
    max_classes = 20
    if len(unique_labels) > max_classes:
        print(f"Too many classes ({len(unique_labels)}). Displaying {max_classes} most common.")
        label_counts = {label: np.sum(labels == label) for label in unique_labels}
        top_labels = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)[:max_classes]
        unique_labels = top_labels
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[color], label=f'Class {label}', alpha=0.6, s=20)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title('PCA Visualization of Embeddings', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_federated_convergence(rounds_history, save_path=None):
    """
    Plot federated learning convergence curves.
    
    Args:
        rounds_history: List of dictionaries with round metrics
        save_path: Path to save the plot
    """
    rounds = [h['round'] for h in rounds_history]
    losses = [h['loss'] for h in rounds_history]
    accuracies = [h['accuracy'] for h in rounds_history]
    
    # Check if we have AUC and EER metrics (for verification-based evaluation)
    has_verification_metrics = any('test_auc' in h for h in rounds_history)
    
    if has_verification_metrics:
        # Extract verification metrics (only from rounds that have them)
        eval_rounds = [h['round'] for h in rounds_history if 'test_auc' in h]
        aucs = [h['test_auc'] for h in rounds_history if 'test_auc' in h]
        eers = [h['test_eer'] for h in rounds_history if 'test_eer' in h]
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training Loss
        axes[0, 0].plot(rounds, losses, marker='o', linewidth=2, markersize=6, color='blue')
        axes[0, 0].set_xlabel('Communication Round', fontsize=12)
        axes[0, 0].set_ylabel('Average Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training Accuracy
        axes[0, 1].plot(rounds, accuracies, marker='o', linewidth=2, markersize=6, color='green')
        axes[0, 1].set_xlabel('Communication Round', fontsize=12)
        axes[0, 1].set_ylabel('Average Accuracy', fontsize=12)
        axes[0, 1].set_title('Training Accuracy Convergence', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Evaluation AUC (Primary Metric)
        if aucs:
            axes[1, 0].plot(eval_rounds, aucs, marker='s', linewidth=2.5, markersize=7, 
                           color='purple', label='Test AUC')
            axes[1, 0].set_xlabel('Communication Round', fontsize=12)
            axes[1, 0].set_ylabel('AUC Score', fontsize=12)
            axes[1, 0].set_title('Evaluation AUC Convergence', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].legend(fontsize=10)
            
            # Add horizontal line at best AUC
            best_auc = max(aucs)
            best_round = eval_rounds[aucs.index(best_auc)]
            axes[1, 0].axhline(y=best_auc, color='purple', linestyle='--', alpha=0.5, 
                              label=f'Best: {best_auc:.4f} @ Round {best_round}')
            axes[1, 0].legend(fontsize=10)
        
        # Evaluation EER (Secondary Metric - Lower is Better)
        if eers:
            axes[1, 1].plot(eval_rounds, eers, marker='^', linewidth=2.5, markersize=7, 
                           color='red', label='Test EER')
            axes[1, 1].set_xlabel('Communication Round', fontsize=12)
            axes[1, 1].set_ylabel('EER Score', fontsize=12)
            axes[1, 1].set_title('Evaluation EER Convergence (Lower is Better)', 
                                fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].legend(fontsize=10)
            
            # Add horizontal line at best EER (minimum)
            best_eer = min(eers)
            best_round = eval_rounds[eers.index(best_eer)]
            axes[1, 1].axhline(y=best_eer, color='red', linestyle='--', alpha=0.5,
                              label=f'Best: {best_eer:.4f} @ Round {best_round}')
            axes[1, 1].legend(fontsize=10)
    else:
        # Original 1x2 subplot for classification metrics
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss convergence
        axes[0].plot(rounds, losses, marker='o', linewidth=2, markersize=6)
        axes[0].set_xlabel('Communication Round', fontsize=12)
        axes[0].set_ylabel('Average Loss', fontsize=12)
        axes[0].set_title('Federated Learning Loss Convergence', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy convergence
        axes[1].plot(rounds, accuracies, marker='o', linewidth=2, markersize=6, color='green')
        axes[1].set_xlabel('Communication Round', fontsize=12)
        axes[1].set_ylabel('Average Accuracy', fontsize=12)
        axes[1].set_title('Federated Learning Accuracy Convergence', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved federated convergence curves to {save_path}")
    
    return fig


def plot_client_comparison(client_metrics, save_path=None):
    """
    Plot comparison of metrics across clients.
    
    Args:
        client_metrics: Dictionary of {client_name: metrics}
        save_path: Path to save the plot
    """
    clients = list(client_metrics.keys())
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1_score']
    
    data = {key: [] for key in metrics_keys}
    for client in clients:
        for key in metrics_keys:
            data[key].append(client_metrics[client].get(key, 0))
    
    x = np.arange(len(clients))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, key in enumerate(metrics_keys):
        ax.bar(x + i * width, data[key], width, label=key.replace('_', ' ').title())
    
    ax.set_xlabel('Client', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Client Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(clients)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved client comparison to {save_path}")
    
    plt.show()
    plt.close()


def plot_client_evolution(history, save_path=None):
    """
    Plot evolution of all clients' performance over rounds.
    Shows loss and accuracy for each client on separate subplots.
    
    Args:
        history: Training history with per-client metrics
        save_path: Path to save the plot
        
    Returns:
        matplotlib figure object
    """
    if not history:
        return None
    
    # Extract client metrics from history
    num_rounds = len(history)
    client_data = {}
    
    for round_idx, round_info in enumerate(history):
        if 'clients' not in round_info:
            continue
            
        for client_stat in round_info['clients']:
            client_id = client_stat['client_id']
            if client_id not in client_data:
                client_data[client_id] = {
                    'rounds': [],
                    'loss': [],
                    'accuracy': []
                }
            
            client_data[client_id]['rounds'].append(round_idx + 1)
            client_data[client_id]['loss'].append(client_stat['loss'])
            client_data[client_id]['accuracy'].append(client_stat['accuracy'])
    
    if not client_data:
        print("No client data found in history")
        return None
    
    # Create color palette for clients
    num_clients = len(client_data)
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_clients, 10)))
    if num_clients > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, num_clients))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot loss for all clients
    for idx, (client_id, data) in enumerate(sorted(client_data.items())):
        # Extract client number for cleaner label
        client_num = client_id.split('_')[-1]
        ax1.plot(data['rounds'], data['loss'], 
                marker='o', markersize=3, linewidth=1.5,
                color=colors[idx], label=f'Client {client_num}', alpha=0.8)
    
    ax1.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Client Training Loss Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy for all clients
    for idx, (client_id, data) in enumerate(sorted(client_data.items())):
        client_num = client_id.split('_')[-1]
        ax2.plot(data['rounds'], data['accuracy'], 
                marker='o', markersize=3, linewidth=1.5,
                color=colors[idx], label=f'Client {client_num}', alpha=0.8)
    
    ax2.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Client Training Accuracy Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved client evolution plot to {save_path}")
    
    return fig
