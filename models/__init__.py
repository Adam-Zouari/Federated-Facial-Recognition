"""Model architectures package."""

from .cnn_custom import CustomCNN, create_custom_cnn
from .resnet import ResNet18Classifier, create_resnet18
from .mobilenet import MobileNetV2Classifier, create_mobilenetv2


def create_model(model_name, num_classes, embedding_dim=128):
    """
    Factory function to create any model by name.
    All models train from scratch (no pretrained weights).
    
    Args:
        model_name: Name of the model ('custom_cnn', 'resnet18', 'mobilenetv2')
        num_classes: Number of identity classes
        embedding_dim: Dimension of the embedding layer
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'custom_cnn':
        return create_custom_cnn(num_classes, embedding_dim)
    elif model_name == 'resnet18':
        return create_resnet18(num_classes, embedding_dim)
    elif model_name == 'mobilenetv2':
        return create_mobilenetv2(num_classes, embedding_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


__all__ = [
    'CustomCNN',
    'ResNet18Classifier',
    'MobileNetV2Classifier',
    'create_custom_cnn',
    'create_resnet18',
    'create_mobilenetv2',
    'create_model'
]
