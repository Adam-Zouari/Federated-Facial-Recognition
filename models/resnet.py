"""
ResNet-18 architecture adapted for facial recognition.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 adapted for facial recognition with embedding extraction.
    Training from scratch (no pretrained weights).
    """
    
    def __init__(self, num_classes=1000, embedding_dim=128):
        """
        Args:
            num_classes: Number of identity classes
            embedding_dim: Dimension of the embedding layer
        """
        super(ResNet18Classifier, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Load ResNet-18 without pretrained weights
        self.resnet = models.resnet18(weights=None)
        
        # Get the number of features from the last layer
        num_features = self.resnet.fc.in_features
        
        # Replace the final fully connected layer
        self.resnet.fc = nn.Identity()
        
        # Add embedding layer with dropout for regularization
        self.embedding = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
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
        # Extract features
        features = self.resnet(x)
        
        # Get embeddings
        embeddings = self.embedding(features)
        
        # Get logits
        logits = self.classifier(embeddings)
        
        if return_embedding:
            return logits, embeddings
        return logits
    
    def get_embedding_dim(self):
        """Return the embedding dimension."""
        return self.embedding_dim


def create_resnet18(num_classes, embedding_dim=128):
    """
    Factory function to create a ResNet-18 classifier.
    
    Args:
        num_classes: Number of identity classes
        embedding_dim: Dimension of the embedding layer
        
    Returns:
        ResNet18Classifier model
    """
    return ResNet18Classifier(
        num_classes=num_classes,
        embedding_dim=embedding_dim
    )
