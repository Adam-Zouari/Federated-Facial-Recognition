"""
MobileNetV2 architecture adapted for facial recognition.
"""

import torch
import torch.nn as nn
from torchvision import models


class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 adapted for facial recognition with embedding extraction.
    Efficient architecture suitable for resource-constrained environments.
    Training from scratch (no pretrained weights).
    """
    
    def __init__(self, num_classes=1000, embedding_dim=128):
        """
        Args:
            num_classes: Number of identity classes
            embedding_dim: Dimension of the embedding layer
        """
        super(MobileNetV2Classifier, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Load MobileNetV2 without pretrained weights
        self.mobilenet = models.mobilenet_v2(weights=None)
        
        # Get the number of features from the last layer
        num_features = self.mobilenet.classifier[1].in_features
        
        # Replace the final classifier
        self.mobilenet.classifier = nn.Identity()
        
        # Add embedding layer
        self.embedding = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, embedding_dim),
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
        # Extract features
        features = self.mobilenet(x)
        
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


def create_mobilenetv2(num_classes, embedding_dim=128):
    """
    Factory function to create a MobileNetV2 classifier.
    
    Args:
        num_classes: Number of identity classes
        embedding_dim: Dimension of the embedding layer
        
    Returns:
        MobileNetV2Classifier model
    """
    return MobileNetV2Classifier(
        num_classes=num_classes,
        embedding_dim=embedding_dim
    )
