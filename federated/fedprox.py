"""
FedProx (Federated Proximal) implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from collections import OrderedDict
from tqdm import tqdm
import config
from utils.metrics import AverageMeter
from .fedavg import fedavg_aggregate


class FedProxClient:
    """
    Client for FedProx federated learning.
    Includes proximal term in local training.
    """
    
    def __init__(self, client_id, model, train_loader, val_loader=None, 
                 device=None, mu=None):
        """
        Args:
            client_id: Unique identifier for the client
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to use for training
            mu: Proximal term coefficient
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or config.DEVICE
        self.mu = mu or config.FEDPROX_MU
        self.model.to(self.device)
        self.global_params = None
    
    def get_model_params(self):
        """Get current model parameters."""
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_params(self, params):
        """Set model parameters."""
        self.model.load_state_dict(copy.deepcopy(params))
    
    def set_global_params(self, params):
        """Set global model parameters (for proximal term)."""
        self.global_params = copy.deepcopy(params)
    
    def proximal_loss(self):
        """
        Compute proximal term: (mu/2) * ||w - w_global||^2
        
        Returns:
            Proximal loss value
        """
        if self.global_params is None:
            return 0.0
        
        prox_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.global_params:
                prox_loss += torch.sum((param - self.global_params[name].to(self.device)) ** 2)
        
        return (self.mu / 2.0) * prox_loss
    
    def train(self, epochs=1, learning_rate=None, optimizer=None):
        """
        Train the model locally with proximal term.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            optimizer: Optimizer instance (if None, creates Adam optimizer)
            
        Returns:
            Dictionary with training statistics
        """
        learning_rate = learning_rate or config.FED_LEARNING_RATE
        
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = AverageMeter()
            epoch_acc = AverageMeter()
            
            pbar = tqdm(self.train_loader, desc=f"Client {self.client_id} - Epoch {epoch+1}/{epochs}")
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                
                # Standard cross-entropy loss
                ce_loss = criterion(outputs, labels)
                
                # Add proximal term
                prox_term = self.proximal_loss()
                loss = ce_loss + prox_term
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                
                # Update metrics
                batch_size = images.size(0)
                epoch_loss.update(loss.item(), batch_size)
                epoch_acc.update(correct / batch_size, batch_size)
                
                total_loss += loss.item() * batch_size
                total_correct += correct
                total_samples += batch_size
                
                pbar.set_postfix({
                    'loss': f'{epoch_loss.avg:.4f}',
                    'acc': f'{epoch_acc.avg:.4f}'
                })
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'num_samples': total_samples
        }
    
    def evaluate(self):
        """
        Evaluate the model on validation data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.val_loader is None:
            return None
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
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
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'num_samples': total_samples
        }


class FedProxServer:
    """
    Server for FedProx federated learning.
    Uses same aggregation as FedAvg.
    """
    
    def __init__(self, global_model):
        """
        Args:
            global_model: Initial global model
        """
        self.global_model = global_model
        self.global_params = copy.deepcopy(global_model.state_dict())
    
    def aggregate(self, client_params_list, client_weights=None):
        """
        Aggregate client model parameters.
        
        Args:
            client_params_list: List of client model parameters
            client_weights: List of weights for each client
            
        Returns:
            Aggregated global model parameters
        """
        self.global_params = fedavg_aggregate(client_params_list, client_weights)
        self.global_model.load_state_dict(self.global_params)
        return copy.deepcopy(self.global_params)
    
    def get_global_params(self):
        """Get current global model parameters."""
        return copy.deepcopy(self.global_params)
    
    def set_global_params(self, params):
        """Set global model parameters."""
        self.global_params = copy.deepcopy(params)
        self.global_model.load_state_dict(self.global_params)
