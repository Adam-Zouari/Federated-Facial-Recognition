"""
Federated client implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import config
from utils.metrics import AverageMeter


class FederatedClient:
    """
    Client for federated learning.
    Handles local training and model updates.
    """
    
    def __init__(self, client_id, model, train_loader, val_loader=None, device=None):
        """
        Args:
            client_id: Unique identifier for the client
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to use for training
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or config.DEVICE
        self.model.to(self.device)
        
    def get_model_params(self):
        """Get current model parameters."""
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_params(self, params):
        """Set model parameters."""
        self.model.load_state_dict(copy.deepcopy(params))
    
    def train(self, epochs=1, learning_rate=None, optimizer=None):
        """
        Train the model locally.
        
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
            
            pbar = tqdm(self.train_loader, 
                       desc=f"Client {self.client_id} - Epoch {epoch+1}/{epochs}",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
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
