"""
Federated learning server implementation.
"""

import torch
import copy
import os
from tqdm import tqdm
import config
from .fedavg import FedAvgServer
from .fedprox import FedProxServer
from utils.mlflow_utils import log_metrics, log_params
from utils.metrics import evaluate_verification


class FederatedServer:
    """
    Main federated learning server coordinator.
    """
    
    def __init__(self, global_model, method='fedavg', device=None, use_mlflow=True):
        """
        Args:
            global_model: Initial global model
            method: Federated learning method ('fedavg' or 'fedprox')
            device: Device to use
            use_mlflow: Whether to use MLflow tracking
        """
        self.device = device or config.DEVICE
        self.global_model = global_model.to(self.device)
        self.method = method.lower()
        self.use_mlflow = use_mlflow and config.MLFLOW_ENABLE
        
        # Initialize appropriate server
        if self.method == 'fedavg':
            self.server = FedAvgServer(self.global_model)
        elif self.method == 'fedprox':
            self.server = FedProxServer(self.global_model)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.round_history = []
        
        # Early stopping state
        self.best_test_auc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        # Checkpoint paths
        self.checkpoint_dir = None
        self.best_checkpoint_path = None
        self.latest_checkpoint_path = None
    
    def federated_training(self, clients, num_rounds, epochs_per_round, 
                          client_fraction=1.0, test_loader=None, early_stopping_patience=None,
                          early_stopping_min_delta=0.001, checkpoint_dir=None):
        """
        Execute federated training with optional early stopping.
        
        Args:
            clients: List of FederatedClient objects
            num_rounds: Number of communication rounds
            epochs_per_round: Local epochs per round
            client_fraction: Fraction of clients to use per round
            test_loader: Global test data loader for evaluation
            early_stopping_patience: Number of rounds without improvement before stopping (None=disabled)
            early_stopping_min_delta: Minimum improvement to reset patience counter
            checkpoint_dir: Directory to save checkpoints (None=no checkpointing)
            
        Returns:
            Training history
        """
        # Setup checkpoint paths
        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.best_checkpoint_path = os.path.join(checkpoint_dir, 'best_global_model.pth')
            self.latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        
        print(f"\nStarting Federated Learning with {self.method.upper()}")
        print(f"Rounds: {num_rounds}, Clients: {len(clients)}, Epochs/Round: {epochs_per_round}")
        if early_stopping_patience is not None:
            print(f"Early Stopping: Enabled (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})")
        if checkpoint_dir is not None:
            print(f"Checkpointing: Enabled (dir={checkpoint_dir})")
        print("="*60)
        
        for round_idx in range(num_rounds):
            print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
            
            # Select clients for this round
            num_selected = max(1, int(len(clients) * client_fraction))
            if num_selected < len(clients):
                import random
                selected_clients = random.sample(clients, num_selected)
            else:
                selected_clients = clients
            
            # Get global parameters
            global_params = self.server.get_global_params()
            
            # Client updates
            client_params_list = []
            client_weights = []
            round_stats = {'clients': []}
            
            for client in selected_clients:
                # Update client model with global parameters
                client.set_model_params(global_params)
                
                # For FedProx, also set global params for proximal term
                if self.method == 'fedprox':
                    client.set_global_params(global_params)
                
                # Local training
                train_stats = client.train(epochs=epochs_per_round)
                
                # Collect updated parameters
                client_params_list.append(client.get_model_params())
                client_weights.append(train_stats['num_samples'])
                
                # Log client stats
                round_stats['clients'].append({
                    'client_id': client.client_id,
                    'loss': train_stats['loss'],
                    'accuracy': train_stats['accuracy'],
                    'num_samples': train_stats['num_samples']
                })
                
                print(f"  Client {client.client_id}: Loss={train_stats['loss']:.4f}, "
                      f"Acc={train_stats['accuracy']:.4f}")
            
            # Server aggregation
            aggregated_params = self.server.aggregate(client_params_list, client_weights)
            
            # Compute average metrics
            avg_loss = sum([s['loss'] * s['num_samples'] for s in round_stats['clients']]) / \
                      sum([s['num_samples'] for s in round_stats['clients']])
            avg_acc = sum([s['accuracy'] * s['num_samples'] for s in round_stats['clients']]) / \
                     sum([s['num_samples'] for s in round_stats['clients']])
            
            round_stats['round'] = round_idx + 1
            round_stats['loss'] = avg_loss
            round_stats['accuracy'] = avg_acc
            
            # Global evaluation
            if test_loader is not None:
                test_stats = self.evaluate_global_model(test_loader)
                round_stats['test_auc'] = test_stats['auc']
                round_stats['test_eer'] = test_stats['eer']
                
                print(f"\n  Global Test: AUC={test_stats['auc']:.4f}, "
                      f"EER={test_stats['eer']:.4f}")
                
                # Early stopping check (based on AUC as primary metric)
                if early_stopping_patience is not None:
                    current_auc = test_stats['auc']
                    
                    # Check if this is the best model (higher AUC is better)
                    if current_auc > self.best_test_auc + early_stopping_min_delta:
                        improvement = current_auc - self.best_test_auc
                        self.best_test_auc = current_auc
                        self.best_model_state = copy.deepcopy(self.global_model.state_dict())
                        self.patience_counter = 0
                        print(f"  ✓ New best AUC: {self.best_test_auc:.4f} (+{improvement:.4f}), "
                              f"EER: {test_stats['eer']:.4f}")
                        
                        # Save best checkpoint
                        if self.checkpoint_dir is not None:
                            self._save_checkpoint(
                                filepath=self.best_checkpoint_path,
                                round_idx=round_idx + 1,
                                is_best=True,
                                test_auc=current_auc,
                                test_eer=test_stats['eer']
                            )
                    else:
                        self.patience_counter += 1
                        print(f"  ⚠ No improvement for {self.patience_counter} rounds "
                              f"(best AUC: {self.best_test_auc:.4f})")
                        
                        # Check if we should stop
                        if self.patience_counter >= early_stopping_patience:
                            print(f"\n{'='*60}")
                            print(f"Early Stopping triggered after {round_idx + 1} rounds")
                            print(f"Best test AUC: {self.best_test_auc:.4f}")
                            print(f"{'='*60}")
                            
                            # Restore best model
                            if self.best_model_state is not None:
                                self.global_model.load_state_dict(self.best_model_state)
                                self.server.set_global_params(self.best_model_state)
                                print("Restored best model from earlier round")
                            
                            break  # Exit the training loop
                elif test_loader is not None:
                    # No early stopping, but still track best model for checkpointing
                    current_auc = test_stats['auc']
                    if current_auc > self.best_test_auc:
                        self.best_test_auc = current_auc
                        self.best_model_state = copy.deepcopy(self.global_model.state_dict())
                        print(f"  ✓ New best AUC: {self.best_test_auc:.4f}, EER: {test_stats['eer']:.4f}")
                        
                        # Save best checkpoint
                        if self.checkpoint_dir is not None:
                            self._save_checkpoint(
                                filepath=self.best_checkpoint_path,
                                round_idx=round_idx + 1,
                                is_best=True,
                                test_auc=current_auc,
                                test_eer=test_stats['eer']
                            )
            
            # Log to MLflow
            if self.use_mlflow:
                mlflow_metrics = {
                    'round_train_loss': avg_loss,
                    'round_train_acc': avg_acc,
                }
                if 'test_auc' in round_stats:
                    mlflow_metrics['round_test_auc'] = round_stats['test_auc']
                    mlflow_metrics['round_test_eer'] = round_stats['test_eer']
                
                log_metrics(mlflow_metrics, step=round_idx + 1)
            
            self.round_history.append(round_stats)
            
            print(f"  Round Avg: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
            
            # Save latest checkpoint after each round
            if self.checkpoint_dir is not None:
                self._save_checkpoint(
                    filepath=self.latest_checkpoint_path,
                    round_idx=round_idx + 1,
                    is_best=False,
                    test_auc=round_stats.get('test_auc', None),
                    test_eer=round_stats.get('test_eer', None)
                )
        
        print("\n" + "="*60)
        print("Federated Learning Complete!")
        
        return self.round_history
    
    def evaluate_global_model(self, test_loader):
        """
        Evaluate global model on test data using embedding-based verification.
        Uses AUC as primary metric and EER as secondary metric.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with AUC, EER, and other verification metrics
        """
        self.global_model.eval()
        
        # Use embedding-based verification evaluation (same as local models)
        metrics = evaluate_verification(
            self.global_model, 
            test_loader, 
            device=self.device,
            num_pairs=None  # Use all possible pairs
        )
        
        return {
            'auc': metrics['auc'],
            'eer': metrics['eer'],
            'eer_threshold': metrics['eer_threshold'],
            'num_positive_pairs': metrics['num_positive_pairs'],
            'num_negative_pairs': metrics['num_negative_pairs']
        }
    
    def save_global_model(self, filepath):
        """Save global model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'method': self.method,
            'round_history': self.round_history
        }, filepath)
        print(f"Global model saved to {filepath}")
    
    def load_global_model(self, filepath):
        """Load global model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.server.set_global_params(checkpoint['model_state_dict'])
        if 'round_history' in checkpoint:
            self.round_history = checkpoint['round_history']
        print(f"Global model loaded from {filepath}")
    
    def _save_checkpoint(self, filepath, round_idx, is_best=False, test_auc=None, test_eer=None):
        """
        Save checkpoint during training.
        
        Args:
            filepath: Path to save checkpoint
            round_idx: Current round index
            is_best: Whether this is the best model so far
            test_auc: Test AUC (if available)
            test_eer: Test EER (if available)
        """
        checkpoint = {
            'round': round_idx,
            'model_state_dict': self.global_model.state_dict(),
            'method': self.method,
            'round_history': self.round_history,
            'is_best': is_best,
            'best_test_auc': self.best_test_auc,
        }
        
        if test_auc is not None:
            checkpoint['test_auc'] = test_auc
        if test_eer is not None:
            checkpoint['test_eer'] = test_eer
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"  💾 Saved best checkpoint: {os.path.basename(filepath)}")
        else:
            print(f"  💾 Saved latest checkpoint: {os.path.basename(filepath)}")
