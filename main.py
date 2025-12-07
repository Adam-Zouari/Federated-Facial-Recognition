"""
Main entry point for the project.
Dataset exploration and high-level orchestration.
"""

import argparse
import os
import config
from clients import explore_client_data


def explore_datasets():
    """Explore all datasets and print statistics."""
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    
    datasets = ['celeba', 'vggface2']
    
    for dataset_name in datasets:
        try:
            print(f"\nExploring {dataset_name.upper()} dataset...")
            explore_client_data(dataset_name)
        except Exception as e:
            print(f"Error exploring {dataset_name}: {e}")
    
    print("\n" + "="*60)
    print("Exploration complete!")
    print("="*60)


def setup_directories():
    """Create necessary output directories."""
    directories = [
        config.OUTPUT_DIR,
        config.CHECKPOINT_DIR,
        config.LOGS_DIR,
        config.PLOTS_DIR,
        os.path.join(config.CHECKPOINT_DIR, 'local'),
        os.path.join(config.CHECKPOINT_DIR, 'centralized'),
        os.path.join(config.CHECKPOINT_DIR, 'federated'),
        os.path.join(config.PLOTS_DIR, 'local'),
        os.path.join(config.PLOTS_DIR, 'centralized'),
        os.path.join(config.PLOTS_DIR, 'federated'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created successfully!")


def print_project_info():
    """Print project information."""
    print("\n" + "="*60)
    print("FEDERATED LEARNING FOR FACIAL RECOGNITION")
    print("="*60)
    print("\nProject Structure:")
    print("  - 2 Clients: CelebA, VGGFace2")
    print("  - 3 Models: Custom CNN, ResNet-18, MobileNetV2")
    print("  - Training: From scratch (no pretrained weights)")
    print("  - Federated Methods: FedAvg, FedProx")
    print("  - Centralized Baseline: Combined dataset training")
    print("\nConfiguration:")
    print(f"  - Image Size: {config.IMG_SIZE}")
    print(f"  - Device: {config.DEVICE}")
    print(f"  - Local Epochs: {config.LOCAL_EPOCHS}")
    print(f"  - Federated Rounds: {config.FED_ROUNDS}")
    print(f"  - Batch Size: {config.LOCAL_BATCH_SIZE}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Federated Learning for Facial Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore datasets
  python main.py --mode explore
  
  # Train local models
  python train_local.py --client celeba --model resnet18
  
  # Run federated learning
  python run_federated.py --method fedavg --rounds 50
  
  # Train centralized model
  python centralized/train_global.py --model resnet18
  
  # Evaluate all models
  python evaluate.py --mode all
        """
    )
    
    parser.add_argument('--mode', type=str, default='info',
                       choices=['info', 'explore', 'setup'],
                       help='Operation mode')
    
    args = parser.parse_args()
    
    if args.mode == 'info':
        print_project_info()
        print("\nUse --mode explore to explore datasets")
        print("Use --mode setup to create directory structure")
        print("\nFor training, see the help message: python main.py --help")
        
    elif args.mode == 'explore':
        setup_directories()
        explore_datasets()
        
    elif args.mode == 'setup':
        setup_directories()
        print("Setup complete! You can now train models.")


if __name__ == '__main__':
    main()
