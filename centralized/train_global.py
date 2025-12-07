"""
Centralized training baseline.
Combines all datasets and trains a single global model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import os
import argparse
from tqdm import tqdm
import config
from models import create_model
from clients import CelebADataset, VGGFace2Dataset
from utils.preprocessing import get_train_transforms, get_test_transforms
from utils.metrics import compute_metrics, evaluate_model, print_metrics, MetricsTracker
from utils.plotting import (plot_training_curves, plot_confusion_matrix, 
                           plot_roc_curve, plot_embeddings_tsne)
from utils.mlflow_utils import (setup_mlflow, start_run, log_params, log_metrics,
                                log_model, log_figure, log_dict, end_run)


class CentralizedTrainer:
    """Centralized training coordinator."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device=None, checkpoint_dir=None, use_mlflow=True):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            use_mlflow: Whether to use MLflow tracking
        """
        self.device = device or config.DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
        self.use_mlflow = use_mlflow and config.MLFLOW_ENABLE
        
        self.metrics_tracker = MetricsTracker()
        self.best_val_acc = 0.0
        
    def train_epoch(self, optimizer, criterion, epoch):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size
            
            pbar.set_postfix({
                'loss': f'{total_loss/total_samples:.4f}',
                'acc': f'{total_correct/total_samples:.4f}'
            })
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, criterion):
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_correct += correct
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def train(self, epochs=None, learning_rate=None, early_stopping_patience=None):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        epochs = epochs or config.CENTRAL_EPOCHS
        learning_rate = learning_rate or config.CENTRAL_LEARNING_RATE
        early_stopping_patience = early_stopping_patience or config.EARLY_STOPPING_PATIENCE
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nStarting Centralized Training")
        print(f"Epochs: {epochs}, LR: {learning_rate}")
        print("="*60)
        
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(criterion)
            
            # Update metrics
            self.metrics_tracker.update(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=learning_rate
            )
            
            # Log to MLflow
            if self.use_mlflow:
                log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': learning_rate
                }, step=epoch)
            
            print(f"Epoch {epoch}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(f'best_model.pth')
                patience_counter = 0
                print(f"  New best model! Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        
        return self.metrics_tracker.metrics
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, 'centralized', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'metrics': self.metrics_tracker.metrics
        }, filepath)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, 'centralized', filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        if 'metrics' in checkpoint:
            self.metrics_tracker.metrics = checkpoint['metrics']


def combine_datasets(max_identities_per_dataset=300):
    """
    Combine both datasets into a single dataset.
    
    Args:
        max_identities_per_dataset: Maximum identities to use from each dataset
        
    Returns:
        Combined train, val, test loaders and total number of classes
    """
    print("Loading and combining datasets...")
    
    datasets = {'train': [], 'val': [], 'test': []}
    total_classes = {'train': 0, 'val': 0, 'test': 0}
    
    # Load CelebA
    try:
        for split in ['train', 'val', 'test']:
            celeba_dataset = CelebADataset(
                root_dir=config.DATA_PATHS['celeba'],
                split=split,
                transform=get_train_transforms(use_augmentation=True) if split == 'train' else get_test_transforms(),
                max_identities=max_identities_per_dataset
            )
            celeba_num_classes = celeba_dataset.get_num_classes()
            datasets[split].append(celeba_dataset)
            print(f"CelebA {split}: {len(celeba_dataset)} samples, {celeba_num_classes} classes")
            total_classes[split] += celeba_num_classes
    except Exception as e:
        print(f"Warning: Could not load CelebA: {e}")
    
    # Load VGGFace2
    try:
        for split in ['train', 'val', 'test']:
            vggface2_dataset = VGGFace2Dataset(
                root_dir=config.DATA_PATHS['vggface2'],
                split=split,
                transform=get_train_transforms(use_augmentation=True) if split == 'train' else get_test_transforms(),
                max_identities=max_identities_per_dataset
            )
            # Remap labels to avoid conflicts with CelebA
            vggface2_num_classes = vggface2_dataset.get_num_classes()
            vggface2_dataset.labels = [l + total_classes[split] for l in vggface2_dataset.labels]
            datasets[split].append(vggface2_dataset)
            print(f"VGGFace2 {split}: {len(vggface2_dataset)} samples, {vggface2_num_classes} classes")
            total_classes[split] += vggface2_num_classes
    except Exception as e:
        print(f"Warning: Could not load VGGFace2: {e}")
    
    if not any(datasets['train']):
        raise ValueError("No datasets could be loaded!")
    
    # Combine datasets for each split
    from torch.utils.data import ConcatDataset
    
    train_combined = ConcatDataset(datasets['train']) if datasets['train'] else None
    val_combined = ConcatDataset(datasets['val']) if datasets['val'] else None
    test_combined = ConcatDataset(datasets['test']) if datasets['test'] else None
    
    print(f"\nCombined dataset:")
    print(f"  Train: {len(train_combined)} samples, {total_classes['train']} total classes")
    print(f"  Val: {len(val_combined)} samples, {total_classes['val']} total classes")
    print(f"  Test: {len(test_combined)} samples, {total_classes['test']} total classes")
    
    # Create data loaders
    train_loader = DataLoader(train_combined, batch_size=config.CENTRAL_BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_combined, batch_size=config.CENTRAL_BATCH_SIZE,
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_combined, batch_size=config.CENTRAL_BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Use max classes across all splits
    max_total_classes = max(total_classes.values())
    
    return train_loader, val_loader, test_loader, max_total_classes


def main():
    parser = argparse.ArgumentParser(description='Centralized Training')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['custom_cnn', 'resnet18', 'mobilenetv2'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=config.CENTRAL_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=config.CENTRAL_LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--max-identities', type=int, default=300,
                       help='Max identities per dataset')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    # Setup MLflow
    use_mlflow = config.MLFLOW_ENABLE and not args.no_mlflow
    if use_mlflow:
        experiment_name = f"{config.MLFLOW_EXPERIMENT_PREFIX}_centralized"
        setup_mlflow(experiment_name, tracking_uri=config.MLFLOW_TRACKING_URI)
    
    # Load combined data
    train_loader, val_loader, test_loader, num_classes = combine_datasets(
        max_identities_per_dataset=args.max_identities
    )
    
    # Create model
    print(f"\nCreating {args.model} model with {num_classes} classes...")
    model = create_model(args.model, num_classes=num_classes)
    
    # Start MLflow run
    run_name = f"centralized_{args.model}"
    if use_mlflow:
        with start_run(run_name=run_name):
            # Log parameters
            log_params({
                'model': args.model,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'max_identities': args.max_identities,
                'num_classes': num_classes,
                'train_size': len(train_loader.dataset),
                'val_size': len(val_loader.dataset),
                'test_size': len(test_loader.dataset),
                'batch_size': config.CENTRAL_BATCH_SIZE,
                'device': str(config.DEVICE)
            })
            
            # Create trainer
            trainer = CentralizedTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                use_mlflow=use_mlflow
            )
            
            # Train
            history = trainer.train(epochs=args.epochs, learning_rate=args.lr)
            
            # Save training curves
            plots_dir = os.path.join(config.PLOTS_DIR, 'centralized')
            os.makedirs(plots_dir, exist_ok=True)
            
            import matplotlib.pyplot as plt
            fig = plot_training_curves(trainer.metrics_tracker, 
                                save_path=os.path.join(plots_dir, 'training_curves.png'))
            if use_mlflow and fig is not None:
                log_figure(fig, "training_curves.png")
                plt.close(fig)
            
            # Evaluate on test set
            print("\nEvaluating on test set...")
            trainer.load_checkpoint('best_model.pth')
            
            results = evaluate_model(trainer.model, test_loader)
            metrics = compute_metrics(
                results['labels'],
                results['predictions'],
                results['scores'],
                num_classes=num_classes
            )
            
            print_metrics(metrics, "Centralized Model - Test Set")
            
            # Log test metrics
            if use_mlflow:
                log_metrics({
                    'test_accuracy': metrics['accuracy'],
                    'test_precision': metrics['precision'],
                    'test_recall': metrics['recall'],
                    'test_f1': metrics['f1'],
                })
                if 'roc_auc' in metrics:
                    log_metrics({'test_roc_auc': metrics['roc_auc']})
            
            # Save visualizations
            fig_cm = plot_confusion_matrix(metrics['confusion_matrix'],
                                 save_path=os.path.join(plots_dir, 'confusion_matrix.png'))
            if use_mlflow and fig_cm is not None:
                log_figure(fig_cm, "confusion_matrix.png")
                plt.close(fig_cm)
            
            if 'embeddings' in results:
                fig_tsne = plot_embeddings_tsne(results['embeddings'], results['labels'],
                                   save_path=os.path.join(plots_dir, 'embeddings_tsne.png'))
                if use_mlflow and fig_tsne is not None:
                    log_figure(fig_tsne, "embeddings_tsne.png")
                    plt.close(fig_tsne)
            
            # Log model
            if use_mlflow:
                log_model(trainer.model, artifact_path="model",
                         registered_model_name=f"centralized_{args.model}")
            
            print("\nCentralized training complete!")
    else:
        # Train without MLflow
        trainer = CentralizedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            use_mlflow=False
        )
        
        history = trainer.train(epochs=args.epochs, learning_rate=args.lr)
        
        plots_dir = os.path.join(config.PLOTS_DIR, 'centralized')
        os.makedirs(plots_dir, exist_ok=True)
        plot_training_curves(trainer.metrics_tracker, 
                            save_path=os.path.join(plots_dir, 'training_curves.png'))
        
        print("\nEvaluating on test set...")
        trainer.load_checkpoint('best_model.pth')
        
        results = evaluate_model(trainer.model, test_loader)
        metrics = compute_metrics(
            results['labels'],
            results['predictions'],
            results['scores'],
            num_classes=num_classes
        )
        
        print_metrics(metrics, "Centralized Model - Test Set")
        
        plot_confusion_matrix(metrics['confusion_matrix'],
                             save_path=os.path.join(plots_dir, 'confusion_matrix.png'))
        
        if 'embeddings' in results:
            plot_embeddings_tsne(results['embeddings'], results['labels'],
                               save_path=os.path.join(plots_dir, 'embeddings_tsne.png'))
        
        print("\nCentralized training complete!")


if __name__ == '__main__':
    main()
