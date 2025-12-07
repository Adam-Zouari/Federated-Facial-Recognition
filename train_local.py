"""
Local training script for individual clients.
Each client trains independently on their own dataset.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import config
from models import create_model
from clients import get_client_data
from utils.metrics import compute_metrics, evaluate_model, print_metrics, MetricsTracker
from utils.plotting import (plot_training_curves, plot_confusion_matrix,
                           plot_roc_curve, plot_embeddings_tsne, plot_embeddings_pca)
from utils.mlflow_utils import (setup_mlflow, start_run, log_params, log_metrics,
                                log_model, log_figure, log_dict, end_run)


class LocalTrainer:
    """Local trainer for a single client."""
    
    def __init__(self, client_name, model, train_loader, val_loader, test_loader,
                 device=None, checkpoint_dir=None, use_mlflow=True):
        """
        Args:
            client_name: Name of the client
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            use_mlflow: Whether to use MLflow tracking
        """
        self.client_name = client_name
        self.device = device or config.DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
        self.use_mlflow = use_mlflow and config.MLFLOW_ENABLE
        
        self.metrics_tracker = MetricsTracker()
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.optimizer = None
        self.scheduler = None
    
    def train_epoch(self, optimizer, criterion, epoch):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"{self.client_name} - Epoch {epoch}")
        
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
                'loss': f'{total_loss/total_samples:.6f}',
                'acc': f'{total_correct/total_samples:.6f}'
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
        epochs = epochs or config.LOCAL_EPOCHS
        learning_rate = learning_rate or config.LOCAL_LEARNING_RATE
        early_stopping_patience = early_stopping_patience or config.EARLY_STOPPING_PATIENCE
        
        # Create or reuse optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        optimizer = self.optimizer
        
        # Compute class weights for balanced loss (even with stratification)
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.extend(labels.tolist())
        
        unique_labels = torch.unique(torch.tensor(all_labels))
        num_classes = len(unique_labels)
        class_counts = torch.bincount(torch.tensor(all_labels), minlength=num_classes)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
        class_weights = class_weights.to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class-weighted loss (weights range: {class_weights.min():.3f} - {class_weights.max():.3f})")
        
        # Create or reuse learning rate scheduler
        if self.scheduler is None:
            self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        scheduler = self.scheduler
        
        print(f"\nStarting Local Training for {self.client_name}")
        print(f"Epochs: {epochs}, LR: {learning_rate}")
        if self.current_epoch > 0:
            print(f"Resuming from epoch {self.current_epoch + 1}")
        print("="*60)
        
        patience_counter = 0
        start_epoch = self.current_epoch + 1
        
        for epoch in range(start_epoch, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(criterion)
            
            # Update current epoch
            self.current_epoch = epoch
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update metrics
            self.metrics_tracker.update(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr
            )
            
            # Log to MLflow
            if self.use_mlflow:
                log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': current_lr
                }, step=epoch)
            
            print(f"Epoch {epoch}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}")
            
            # Save best model based on validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
                print(f"  New best model! Val Acc: {val_acc:.6f}")
            else:
                patience_counter += 1
            
            # Save latest checkpoint for resumption
            self.save_checkpoint('latest_checkpoint.pth')
            
            # Early stopping based on validation loss
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.6f}")
        
        return self.metrics_tracker.metrics
    
    def save_checkpoint(self, filename):
        """Save complete model checkpoint for training resumption."""
        filepath = os.path.join(self.checkpoint_dir, 'local', self.client_name, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'metrics': self.metrics_tracker.metrics
        }
        
        torch.save(checkpoint, filepath)
        print(f"  Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename, resume_training=False):
        """Load model checkpoint and optionally resume training state.
        
        Args:
            filename: Checkpoint filename
            resume_training: If True, restore optimizer, scheduler, and epoch state
        """
        filepath = os.path.join(self.checkpoint_dir, 'local', self.client_name, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: Checkpoint not found at {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Always load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'metrics' in checkpoint:
            self.metrics_tracker.metrics = checkpoint['metrics']
        
        # Optionally restore training state for resumption
        if resume_training:
            self.current_epoch = checkpoint.get('epoch', 0)
            
            if checkpoint.get('optimizer_state_dict') and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"  Optimizer state restored")
            
            if checkpoint.get('scheduler_state_dict') and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"  Scheduler state restored")
            
            print(f"  Resuming from epoch {self.current_epoch}")
        
        print(f"  Checkpoint loaded: {filename}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Local Model Training')
    parser.add_argument('--client', type=str, required=True,
                       choices=['celeba', 'vggface2'],
                       help='Client dataset')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['custom_cnn', 'resnet18', 'mobilenetv2'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=config.LOCAL_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=config.LOCAL_LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=config.LOCAL_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=config.EARLY_STOPPING_PATIENCE,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--resume', type=str, choices=['latest', 'best'], default=None,
                       help='Resume training from checkpoint (latest or best)')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    # Setup MLflow
    use_mlflow = config.MLFLOW_ENABLE and not args.no_mlflow
    if use_mlflow:
        experiment_name = f"{config.MLFLOW_EXPERIMENT_PREFIX}_local_{args.client}"
        setup_mlflow(experiment_name, tracking_uri=config.MLFLOW_TRACKING_URI)
    
    # Load client data
    print(f"\nLoading {args.client.upper()} dataset...")
    train_loader, val_loader, test_loader, num_classes = get_client_data(
        args.client, batch_size=args.batch_size
    )
    
    if train_loader is None:
        print(f"Error: Could not load {args.client} dataset!")
        return
    
    # Create model
    print(f"\nCreating {args.model} model with {num_classes} classes...")
    model = create_model(args.model, num_classes=num_classes)
    
    # Start MLflow run
    run_name = f"{args.client}_{args.model}"
    if use_mlflow:
        with start_run(run_name=run_name):
            # Log parameters
            log_params({
                'client': args.client,
                'model': args.model,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'num_classes': num_classes,
                'train_size': len(train_loader.dataset),
                'val_size': len(val_loader.dataset),
                'test_size': len(test_loader.dataset),
                'device': str(config.DEVICE)
            })
            
            # Create trainer
            trainer = LocalTrainer(
                client_name=args.client,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                use_mlflow=use_mlflow
            )
            
            # Resume from checkpoint if requested
            if args.resume:
                checkpoint_file = 'latest_checkpoint.pth' if args.resume == 'latest' else 'best_model.pth'
                print(f"\nAttempting to resume from {checkpoint_file}...")
                if trainer.load_checkpoint(checkpoint_file, resume_training=True):
                    print(f"Successfully resumed from {args.resume} checkpoint")
                else:
                    print(f"No checkpoint found, starting from scratch")
            
            # Train
            history = trainer.train(epochs=args.epochs, learning_rate=args.lr, early_stopping_patience=args.patience)
            
            # Save training curves
            plots_dir = os.path.join(config.PLOTS_DIR, 'local', args.client)
            os.makedirs(plots_dir, exist_ok=True)
            
            import matplotlib.pyplot as plt
            fig = plot_training_curves(trainer.metrics_tracker,
                                save_path=os.path.join(plots_dir, f'{args.model}_training_curves.png'))
            if use_mlflow and fig is not None:
                log_figure(fig, f"{args.model}_training_curves.png")
                plt.close(fig)
            
            # Evaluate on test set
            print("\nEvaluating on test set...")
            trainer.load_checkpoint('best_model.pth')
            
            results = evaluate_model(trainer.model, test_loader, return_embeddings=True)
            metrics = compute_metrics(
                results['labels'],
                results['predictions'],
                results['scores'],
                num_classes=num_classes
            )
            
            print_metrics(metrics, f"{args.client.upper()} - {args.model} - Test Set")
            
            # Log test metrics
            if use_mlflow:
                log_metrics({
                    'test_accuracy': metrics['accuracy'],
                    'test_precision': metrics['precision'],
                    'test_recall': metrics['recall'],
                    'test_f1': metrics['f1_score'],
                })
                if 'roc_auc' in metrics:
                    log_metrics({'test_roc_auc': metrics['roc_auc']})
            
            # Save visualizations
            fig_cm = plot_confusion_matrix(metrics['confusion_matrix'],
                                 save_path=os.path.join(plots_dir, f'{args.model}_confusion_matrix.png'))
            if use_mlflow and fig_cm is not None:
                log_figure(fig_cm, f"{args.model}_confusion_matrix.png")
                plt.close(fig_cm)
            
            if 'fpr' in metrics and 'tpr' in metrics:
                fig_roc = plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics.get('roc_auc'),
                              save_path=os.path.join(plots_dir, f'{args.model}_roc_curve.png'),
                              label=args.model)
                if use_mlflow and fig_roc is not None:
                    log_figure(fig_roc, f"{args.model}_roc_curve.png")
                    plt.close(fig_roc)
            
            if 'embeddings' in results:
                fig_tsne = plot_embeddings_tsne(results['embeddings'], results['labels'],
                                   save_path=os.path.join(plots_dir, f'{args.model}_embeddings_tsne.png'))
                if use_mlflow and fig_tsne is not None:
                    log_figure(fig_tsne, f"{args.model}_embeddings_tsne.png")
                    plt.close(fig_tsne)
                    
                fig_pca = plot_embeddings_pca(results['embeddings'], results['labels'],
                                  save_path=os.path.join(plots_dir, f'{args.model}_embeddings_pca.png'))
                if use_mlflow and fig_pca is not None:
                    log_figure(fig_pca, f"{args.model}_embeddings_pca.png")
                    plt.close(fig_pca)
            
            # Save metrics to file and log to MLflow
            import json
            metrics_file = os.path.join(plots_dir, f'{args.model}_metrics.json')
            with open(metrics_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                metrics_serializable = {}
                for key, value in metrics.items():
                    if isinstance(value, (list, dict, str, int, float)):
                        metrics_serializable[key] = value
                    elif hasattr(value, 'tolist'):
                        metrics_serializable[key] = value.tolist()
                json.dump(metrics_serializable, f, indent=2)
            
            if use_mlflow:
                log_dict(metrics_serializable, f"{args.model}_metrics.json")
                # Log the best model
                log_model(trainer.model, artifact_path="model", 
                         registered_model_name=f"{args.client}_{args.model}_local")
            
            print(f"\nLocal training complete for {args.client}!")
            print(f"Results saved to {plots_dir}")
    else:
        # Train without MLflow
        trainer = LocalTrainer(
            client_name=args.client,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            use_mlflow=False
        )
        
        # Resume from checkpoint if requested
        if args.resume:
            checkpoint_file = 'latest_checkpoint.pth' if args.resume == 'latest' else 'best_model.pth'
            print(f"\nAttempting to resume from {checkpoint_file}...")
            if trainer.load_checkpoint(checkpoint_file, resume_training=True):
                print(f"Successfully resumed from {args.resume} checkpoint")
            else:
                print(f"No checkpoint found, starting from scratch")
        
        history = trainer.train(epochs=args.epochs, learning_rate=args.lr, early_stopping_patience=args.patience)
        
        plots_dir = os.path.join(config.PLOTS_DIR, 'local', args.client)
        os.makedirs(plots_dir, exist_ok=True)
        plot_training_curves(trainer.metrics_tracker,
                            save_path=os.path.join(plots_dir, f'{args.model}_training_curves.png'))
        
        print("\nEvaluating on test set...")
        trainer.load_checkpoint('best_model.pth')
        
        results = evaluate_model(trainer.model, test_loader, return_embeddings=True)
        metrics = compute_metrics(
            results['labels'],
            results['predictions'],
            results['scores'],
            num_classes=num_classes
        )
        
        print_metrics(metrics, f"{args.client.upper()} - {args.model} - Test Set")
        
        plot_confusion_matrix(metrics['confusion_matrix'],
                             save_path=os.path.join(plots_dir, f'{args.model}_confusion_matrix.png'))
        
        if 'fpr' in metrics and 'tpr' in metrics:
            plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics.get('roc_auc'),
                          save_path=os.path.join(plots_dir, f'{args.model}_roc_curve.png'),
                          label=args.model)
        
        if 'embeddings' in results:
            plot_embeddings_tsne(results['embeddings'], results['labels'],
                               save_path=os.path.join(plots_dir, f'{args.model}_embeddings_tsne.png'))
            plot_embeddings_pca(results['embeddings'], results['labels'],
                              save_path=os.path.join(plots_dir, f'{args.model}_embeddings_pca.png'))
        
        import json
        metrics_file = os.path.join(plots_dir, f'{args.model}_metrics.json')
        with open(metrics_file, 'w') as f:
            metrics_serializable = {}
            for key, value in metrics.items():
                if isinstance(value, (list, dict, str, int, float)):
                    metrics_serializable[key] = value
                elif hasattr(value, 'tolist'):
                    metrics_serializable[key] = value.tolist()
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"\nLocal training complete for {args.client}!")
        print(f"Results saved to {plots_dir}")


if __name__ == '__main__':
    main()
