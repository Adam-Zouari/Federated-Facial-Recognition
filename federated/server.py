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
    
    def federated_training(self, clients, num_rounds, epochs_per_round, 
                          client_fraction=1.0, test_loader=None):
        """
        Execute federated training.
        
        Args:
            clients: List of FederatedClient objects
            num_rounds: Number of communication rounds
            epochs_per_round: Local epochs per round
            client_fraction: Fraction of clients to use per round
            test_loader: Global test data loader for evaluation
            
        Returns:
            Training history
        """
        print(f"\nStarting Federated Learning with {self.method.upper()}")
        print(f"Rounds: {num_rounds}, Clients: {len(clients)}, Epochs/Round: {epochs_per_round}")
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
                round_stats['test_loss'] = test_stats['loss']
                round_stats['test_accuracy'] = test_stats['accuracy']
                
                print(f"\n  Global Test: Loss={test_stats['loss']:.4f}, "
                      f"Acc={test_stats['accuracy']:.4f}")
            
            # Log to MLflow
            if self.use_mlflow:
                mlflow_metrics = {
                    'round_train_loss': avg_loss,
                    'round_train_acc': avg_acc,
                }
                if 'test_loss' in round_stats:
                    mlflow_metrics['round_test_loss'] = round_stats['test_loss']
                    mlflow_metrics['round_test_acc'] = round_stats['test_accuracy']
                
                log_metrics(mlflow_metrics, step=round_idx + 1)
            
            self.round_history.append(round_stats)
            
            print(f"  Round Avg: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
        
        print("\n" + "="*60)
        print("Federated Learning Complete!")
        
        return self.round_history
    
    def evaluate_global_model(self, test_loader):
        """
        Evaluate global model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_correct += correct
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc
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
