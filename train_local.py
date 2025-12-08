"""
Local training script for individual clients.
Each client trains independently on their own dataset.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import config
from models import create_model
from clients import get_client_data
from utils.metrics import (compute_metrics, evaluate_model, print_metrics, MetricsTracker,
                          evaluate_verification, extract_embeddings)
from utils.plotting import (plot_training_curves, plot_confusion_matrix,
                           plot_roc_curve)
from utils.mlflow_utils import (setup_mlflow, start_run, log_params, log_metrics,
                                log_model, log_figure, log_dict, end_run)


class LocalTrainer:
    """Local trainer for a single client."""
    
    def __init__(self, client_name, model, train_loader, val_loader, test_loader,
                 device=None, checkpoint_dir=None, use_mlflow=True, aug_level='strong'):
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
            aug_level: Augmentation level (none/weak/strong)
        """
        self.client_name = f"{client_name}_{aug_level}"
        self.client_name_base = client_name  # Base name without augmentation suffix
        self.device = device or config.DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
        self.use_mlflow = use_mlflow and config.MLFLOW_ENABLE
        
        self.metrics_tracker = MetricsTracker()
        self.best_val_auc = 0.0  # Changed from best_val_acc
        self.best_val_eer = float('inf')  # Track best EER
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
    
    def validate(self, criterion=None):
        """
        Validate the model using embedding-based verification.
        Returns verification metrics (AUC, EER).
        """
        self.model.eval()
        
        # Perform embedding-based verification evaluation
        verification_metrics = evaluate_verification(
            self.model, 
            self.val_loader, 
            device=self.device,
            num_pairs=5000  # Limit pairs for faster validation
        )
        
        return verification_metrics
    
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
        # Use caching to avoid recomputing on every run
        # Cache uses base client name (without augmentation suffix) since class distribution is identical
        cache_dir = Path("checkpoints") / "class_weights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.client_name_base}_class_weights.pth"
        
        if cache_file.exists():
            print(f"Loading cached class weights from {cache_file}")
            cache_data = torch.load(cache_file, map_location='cpu', weights_only=False)
            class_counts = cache_data['class_counts']
            num_classes = cache_data['num_classes']
        else:
            print(f"Computing class weights (first run - will be cached to {cache_file})...")
            all_labels = []
            for _, labels in self.train_loader:
                all_labels.extend(labels.tolist())
            
            unique_labels = torch.unique(torch.tensor(all_labels))
            num_classes = len(unique_labels)
            class_counts = torch.bincount(torch.tensor(all_labels), minlength=num_classes)
            
            # Cache for future runs
            torch.save({
                'num_classes': num_classes,
                'class_counts': class_counts,
                'dataset': self.client_name
            }, cache_file)
            print(f"Class weights cached to {cache_file}")
        
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
        class_weights = class_weights.to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class-weighted loss (weights range: {class_weights.min():.3f} - {class_weights.max():.3f})")
        
        # Create or reuse learning rate scheduler (monitors AUC - higher is better)
        if self.scheduler is None:
            self.scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
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
            
            # Validate using embedding-based verification
            val_metrics = self.validate()
            val_auc = val_metrics['auc']
            val_eer = val_metrics['eer']
            
            # Update current epoch
            self.current_epoch = epoch
            
            # Update learning rate based on validation AUC (higher is better)
            scheduler.step(val_auc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update metrics
            self.metrics_tracker.update(
                train_loss=train_loss,
                train_acc=train_acc,
                val_auc=val_auc,
                val_eer=val_eer,
                lr=current_lr
            )
            
            # Log to MLflow
            if self.use_mlflow:
                log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_auc': val_auc,
                    'val_eer': val_eer,
                    'num_positive_pairs': val_metrics['num_positive_pairs'],
                    'num_negative_pairs': val_metrics['num_negative_pairs'],
                    'learning_rate': current_lr
                }, step=epoch)
            
            print(f"Epoch {epoch}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, "
                  f"Val AUC: {val_auc:.6f}, Val EER: {val_eer:.6f}")
            
            # Save best model based on validation AUC (higher is better)
            improved = False
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_val_eer = val_eer
                improved = True
            # Also consider EER improvement (lower is better)
            elif val_auc == self.best_val_auc and val_eer < self.best_val_eer:
                self.best_val_eer = val_eer
                improved = True
            
            if improved:
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
                print(f"  New best model! Val AUC: {val_auc:.6f}, Val EER: {val_eer:.6f}")
            else:
                patience_counter += 1
            
            # Save latest checkpoint for resumption
            self.save_checkpoint('latest_checkpoint.pth')
            
            # Early stopping based on patience
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation AUC: {self.best_val_auc:.6f}")
        print(f"Best Validation EER: {self.best_val_eer:.6f}")
        
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
            'best_val_auc': self.best_val_auc,
            'best_val_eer': self.best_val_eer,
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
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        self.best_val_eer = checkpoint.get('best_val_eer', float('inf'))
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
    parser.add_argument('--augmentation', type=str, default='strong',
                       choices=['none', 'weak', 'strong'],
                       help='Augmentation level: none (no aug), weak (flip+small rotation+mild color), strong (all transforms)')
    
    args = parser.parse_args()
    
    # Map augmentation level to config
    aug_map = {
        'none': config.AUGMENTATION_NONE,
        'weak': config.AUGMENTATION_WEAK,
        'strong': config.AUGMENTATION_STRONG
    }
    aug_config = aug_map[args.augmentation]
    
    # Setup MLflow
    use_mlflow = config.MLFLOW_ENABLE and not args.no_mlflow
    if use_mlflow:
        experiment_name = f"{config.MLFLOW_EXPERIMENT_PREFIX}_local_{args.client}"
        setup_mlflow(experiment_name, tracking_uri=config.MLFLOW_TRACKING_URI)
    
    # Load client data
    print(f"\nLoading {args.client.upper()} dataset with '{args.augmentation}' augmentation...")
    train_loader, val_loader, test_loader, num_classes = get_client_data(
        args.client, batch_size=args.batch_size, aug_config=aug_config
    )
    
    if train_loader is None:
        print(f"Error: Could not load {args.client} dataset!")
        return
    
    # Create model
    print(f"\nCreating {args.model} model with {num_classes} classes...")
    model = create_model(args.model, num_classes=num_classes)
    
    # Start MLflow run
    run_name = f"{args.client}_{args.model}_{args.augmentation}"
    if use_mlflow:
        with start_run(run_name=run_name):
            # Log parameters
            log_params({
                'client': args.client,
                'model': args.model,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'augmentation': args.augmentation,
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
                use_mlflow=use_mlflow,
                aug_level=args.augmentation
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
            plots_dir = os.path.join(config.PLOTS_DIR, 'local', f"{args.client}_{args.augmentation}")
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
            
            # Final test evaluation using embedding-based verification
            print("\n" + "="*60)
            print(f"Final Test Evaluation - Embedding-Based Verification")
            print("="*60)
            
            test_metrics = evaluate_verification(
                trainer.model,
                test_loader,
                device=config.DEVICE,
                num_pairs=None  # Use all possible pairs for final test
            )
            
            print(f"\nTest Set Results:")
            print(f"  AUC: {test_metrics['auc']:.6f}")
            print(f"  EER: {test_metrics['eer']:.6f}")
            print(f"  EER Threshold: {test_metrics['eer_threshold']:.6f}")
            print(f"  Positive Pairs: {test_metrics['num_positive_pairs']}")
            print(f"  Negative Pairs: {test_metrics['num_negative_pairs']}")
            
            # Log test metrics
            if use_mlflow:
                log_metrics({
                    'test_auc': test_metrics['auc'],
                    'test_eer': test_metrics['eer'],
                    'test_eer_threshold': test_metrics['eer_threshold'],
                    'test_num_positive_pairs': test_metrics['num_positive_pairs'],
                    'test_num_negative_pairs': test_metrics['num_negative_pairs']
                })
            
            # Save visualizations
            # Plot ROC curve for verification
            fig_roc = plot_roc_curve(
                test_metrics['fpr'], 
                test_metrics['tpr'], 
                test_metrics['auc'],
                save_path=os.path.join(plots_dir, f'{args.model}_verification_roc_curve.png'),
                label=f'{args.model} Verification'
            )
            if use_mlflow and fig_roc is not None:
                log_figure(fig_roc, f"{args.model}_verification_roc_curve.png")
                plt.close(fig_roc)
            
            # Save metrics to file and log to MLflow
            import json
            metrics_file = os.path.join(plots_dir, f'{args.model}_verification_metrics.json')
            with open(metrics_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                metrics_serializable = {
                    'auc': float(test_metrics['auc']),
                    'eer': float(test_metrics['eer']),
                    'eer_threshold': float(test_metrics['eer_threshold']),
                    'num_positive_pairs': int(test_metrics['num_positive_pairs']),
                    'num_negative_pairs': int(test_metrics['num_negative_pairs'])
                }
                json.dump(metrics_serializable, f, indent=2)
            
            if use_mlflow:
                log_dict(metrics_serializable, f"{args.model}_verification_metrics.json")
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
        
        plots_dir = os.path.join(config.PLOTS_DIR, 'local', f"{args.client}_{args.augmentation}")
        os.makedirs(plots_dir, exist_ok=True)
        plot_training_curves(trainer.metrics_tracker,
                            save_path=os.path.join(plots_dir, f'{args.model}_training_curves.png'))
        
        print("\nEvaluating on test set...")
        trainer.load_checkpoint('best_model.pth')
        
        # Final test evaluation using embedding-based verification
        print("\n" + "="*60)
        print(f"Final Test Evaluation - Embedding-Based Verification")
        print("="*60)
        
        test_metrics = evaluate_verification(
            trainer.model,
            test_loader,
            device=config.DEVICE,
            num_pairs=None  # Use all possible pairs for final test
        )
        
        print(f"\nTest Set Results:")
        print(f"  AUC: {test_metrics['auc']:.6f}")
        print(f"  EER: {test_metrics['eer']:.6f}")
        print(f"  EER Threshold: {test_metrics['eer_threshold']:.6f}")
        print(f"  Positive Pairs: {test_metrics['num_positive_pairs']}")
        print(f"  Negative Pairs: {test_metrics['num_negative_pairs']}")
        
        # Plot verification ROC curve
        plot_roc_curve(
            test_metrics['fpr'], 
            test_metrics['tpr'], 
            test_metrics['auc'],
            save_path=os.path.join(plots_dir, f'{args.model}_verification_roc_curve.png'),
            label=f'{args.model} Verification'
        )
        
        import json
        metrics_file = os.path.join(plots_dir, f'{args.model}_verification_metrics.json')
        with open(metrics_file, 'w') as f:
            metrics_serializable = {
                'auc': float(test_metrics['auc']),
                'eer': float(test_metrics['eer']),
                'eer_threshold': float(test_metrics['eer_threshold']),
                'num_positive_pairs': int(test_metrics['num_positive_pairs']),
                'num_negative_pairs': int(test_metrics['num_negative_pairs'])
            }
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"\nLocal training complete for {args.client}!")
        print(f"Results saved to {plots_dir}")


if __name__ == '__main__':
    main()
