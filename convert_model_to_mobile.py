"""
Convert PyTorch model to TorchScript for mobile deployment.

This script converts your trained PyTorch model to TorchScript format,
which can be loaded and run on Android devices using PyTorch Mobile.
"""

import torch
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import config and models
sys.path.append(str(Path(__file__).parent))

import config
from models import create_model


class MobileModel(torch.nn.Module):
    """
    Wrapper model that returns only embeddings for mobile deployment.
    This simplifies the model interface for mobile apps.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        """
        Forward pass that returns only normalized embeddings.
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
            
        Returns:
            Normalized embeddings [batch_size, embedding_dim]
        """
        # Get embeddings from model
        _, embeddings = self.model(x, return_embedding=True)
        
        # Normalize embeddings (L2 normalization)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


def convert_model_to_mobile(model_path, output_path, num_classes=None):
    """
    Convert PyTorch model to TorchScript for mobile.
    
    Args:
        model_path: Path to trained model checkpoint (.pth)
        output_path: Path to save TorchScript model (.pt)
        num_classes: Number of classes (if None, will auto-detect from checkpoint)
    """
    print(f"Converting model: {model_path}")
    print(f"Output: {output_path}")
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint
    
    # Auto-detect num_classes from checkpoint
    if num_classes is None:
        # Find classifier weight shape
        classifier_weight_key = None
        for key in model_state.keys():
            if 'classifier.weight' in key or 'fc.weight' in key:
                classifier_weight_key = key
                break
        
        if classifier_weight_key:
            num_classes = model_state[classifier_weight_key].shape[0]
            print(f"Auto-detected num_classes: {num_classes}")
        else:
            # Default for VGGFace2
            num_classes = 480
            print(f"Could not auto-detect num_classes, using default: {num_classes}")
    
    # Create model
    print("Loading model architecture...")
    model = create_model('mobilenetv2', num_classes=num_classes)
    model.load_state_dict(model_state)
    model.eval()
    
    # Wrap model to return only embeddings
    mobile_model = MobileModel(model)
    mobile_model.eval()
    
    print("Converting to TorchScript...")
    
    # Create example input
    example_input = torch.rand(1, 3, config.IMG_SIZE[0], config.IMG_SIZE[1])
    
    # Trace model
    try:
        traced_model = torch.jit.trace(mobile_model, example_input)
        
        # Verify traced model works
        with torch.no_grad():
            original_output = mobile_model(example_input)
            traced_output = traced_model(example_input)
            
            # Check outputs match
            if not torch.allclose(original_output, traced_output, atol=1e-5):
                print("Warning: Traced model output differs from original!")
            else:
                print("✓ Traced model verified successfully")
        
        # Save model
        traced_model.save(output_path)
        print(f"✓ Model saved to: {output_path}")
        
        # Check file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Model size: {file_size_mb:.2f} MB")
        
        print("\n" + "="*60)
        print("Conversion successful!")
        print("="*60)
        print(f"\nNext steps:")
        print(f"1. Copy {output_path} to Android_App/app/src/main/assets/model.pt")
        print(f"2. Build and run the Android app")
        print(f"\nModel details:")
        print(f"  - Input shape: [1, 3, {config.IMG_SIZE[0]}, {config.IMG_SIZE[1]}]")
        print(f"  - Output shape: [1, embedding_dim]")
        print(f"  - Normalization: mean={config.NORMALIZE_MEAN}, std={config.NORMALIZE_STD}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TorchScript for mobile')
    parser.add_argument('--model', type=str, 
                       default='checkpoints/local/vggface2_weak/best_model.pth',
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--output', type=str, 
                       default='Android_App/app/src/main/assets/model.pt',
                       help='Output path for TorchScript model (.pt)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Convert model
    convert_model_to_mobile(
        model_path=args.model,
        output_path=args.output,
        num_classes=args.num_classes
    )


if __name__ == '__main__':
    main()
