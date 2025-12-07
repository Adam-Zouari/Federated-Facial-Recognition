"""
Custom CNN architecture for facial recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for facial recognition.
    Relatively small and efficient for local training.
    """
    
    def __init__(self, num_classes=1000, embedding_dim=128):
        """
        Args:
            num_classes: Number of identity classes
            embedding_dim: Dimension of the embedding layer
        """
        super(CustomCNN, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Classification layer
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x, return_embedding=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
            return_embedding: If True, return (logits, embeddings)
            
        Returns:
            logits or (logits, embeddings)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Get embeddings
        embeddings = self.fc(x)
        
        # Get logits
        logits = self.classifier(embeddings)
        
        if return_embedding:
            return logits, embeddings
        return logits
    
    def get_embedding_dim(self):
        """Return the embedding dimension."""
        return self.embedding_dim


def create_custom_cnn(num_classes, embedding_dim=128):
    """
    Factory function to create a custom CNN.
    
    Args:
        num_classes: Number of identity classes
        embedding_dim: Dimension of the embedding layer
        
    Returns:
        CustomCNN model
    """
    return CustomCNN(num_classes=num_classes, embedding_dim=embedding_dim)
