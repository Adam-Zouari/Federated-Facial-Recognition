"""
VGGFace2 Federated Learning Training Script.
Trains federated learning models on VGGFace2 dataset with multiple clients.
"""

import argparse
import os
import torch
import json
import config
from models import create_model
from clients.vggface_client import load_vggface2_data
from federated import FederatedClient, FedProxClient, FederatedServer
from utils.federated_partitioner import VGGFace2Partitioner
from utils.metrics import evaluate_model, compute_metrics, print_metrics
from utils.plotting import (plot_federated_convergence, plot_confusion_matrix,
                           plot_client_comparison, plot_client_evolution)
from utils.mlflow_utils import (setup_mlflow, start_run, log_params, log_metrics,
                                log_model, log_figure, log_dict, end_run)


def create_vggface2_federated_clients(num_clients=10, partition_strategy='iid', alpha=0.5,
                                     method='fedavg', model_name='resnet18', aug_config=None):
    """
    Create federated clients for VGGFace2 dataset.
    
    Args:
        num_clients: Number of federated clients
        partition_strategy: 'iid' or 'non-iid'
        alpha: Dirichlet concentration parameter for non-iid
        method: Federated learning method ('fedavg' or 'fedprox')
        model_name: Model architecture name
        aug_config: Data augmentation configuration
        
    Returns:
        List of clients, global test loader, num_classes, partitioner
    """
    print("\n" + "="*70)
    print("CREATING VGGFACE2 FEDERATED CLIENTS")
    print("="*70)
    
    # Create data partitioner
    partitioner = VGGFace2Partitioner(
        root_dir=config.DATA_PATHS['vggface2'],
        num_clients=num_clients,
        partition_strategy=partition_strategy,
        alpha=alpha
    )
    
    clients = []
    total_identities = set()  # Track all unique identities across clients
    client_data_info = []  # Store client data info temporarily
    
    print("\n" + "-"*70)
    print("Loading client datasets...")
    print("-"*70)
    
    # First pass: Load all client data and determine total number of classes
    for client_id in range(num_clients):
        # Get identities for this client
        client_identities = partitioner.get_client_identities(client_id)
        
        if len(client_identities) == 0:
            print(f"\nClient {client_id}: No identities assigned, skipping...")
            continue
        
        print(f"\nClient {client_id}:")
        print(f"  Assigned identities: {len(client_identities)}")
        
        # Load data for this client (only training data needed)
        train_loader, _, _, num_classes = load_vggface2_data(
            batch_size=config.FED_BATCH_SIZE,
            allowed_identities=client_identities,
            aug_config=aug_config
        )
        
        if train_loader is None:
            print(f"  Warning: No data loaded for client {client_id}, skipping...")
            continue
        
        # Track all identities
        total_identities.update(client_identities)
        
        # Store client info
        client_data_info.append({
            'client_id': client_id,
            'train_loader': train_loader,
            'num_classes': num_classes
        })
        
        print(f"  Loaded: {num_classes} classes, {len(train_loader.dataset)} train samples")
    
    if not client_data_info:
        raise ValueError("No clients could be created!")
    
    # Total number of classes = total unique identities across all clients
    total_num_classes = len(total_identities)
    
    print(f"\n" + "-"*70)
    print(f"Total unique identities across all clients: {total_num_classes}")
    print(f"Creating client models with {total_num_classes} classes...")
    print("-"*70)
    
    # Second pass: Create clients with models that all have the same number of classes
    for info in client_data_info:
        client_id = info['client_id']
        train_loader = info['train_loader']
        
        # Create client model with TOTAL number of classes (same for all clients)
        client_model = create_model(model_name, num_classes=total_num_classes)
        # Create client model with TOTAL number of classes (same for all clients)
        client_model = create_model(model_name, num_classes=total_num_classes)
        
        # Create federated client (no val_loader needed)
        if method == 'fedprox':
            client = FedProxClient(
                client_id=f"vggface2_client_{client_id}",
                model=client_model,
                train_loader=train_loader,
                val_loader=None
            )
        else:
            client = FederatedClient(
                client_id=f"vggface2_client_{client_id}",
                model=client_model,
                train_loader=train_loader,
                val_loader=None
            )
        
        clients.append(client)
    
    if not clients:
        raise ValueError("No clients could be created!")
    
    print(f"\n" + "="*70)
    print(f"Successfully created {len(clients)} VGGFace2 federated clients")
    print(f"All clients use models with {total_num_classes} classes")
    print("="*70 + "\n")
    
    # Load global test set (all identities)
    print("Loading global test set...")
    _, _, global_test_loader, _ = load_vggface2_data(
        batch_size=config.FED_BATCH_SIZE,
        max_identities=None  # Use all identities for testing
    )
    
    return clients, global_test_loader, total_num_classes, partitioner


def main():
    parser = argparse.ArgumentParser(description='VGGFace2 Federated Learning Training')
    
    # Federated learning parameters
    parser.add_argument('--num-clients', type=int, default=10,
                       help='Number of federated clients (default: 10)')
    parser.add_argument('--partition', type=str, default='iid',
                       choices=['iid', 'non-iid'],
                       help='Data partition strategy')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha for non-iid partition (lower=more skewed)')
    parser.add_argument('--method', type=str, default='fedavg',
                       choices=['fedavg', 'fedprox'],
                       help='Federated learning method')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['custom_cnn', 'resnet18', 'mobilenetv2'],
                       help='Model architecture')
    
    # Training parameters
    parser.add_argument('--rounds', type=int, default=50,
                       help='Number of communication rounds')
    parser.add_argument('--epochs-per-round', type=int, default=5,
                       help='Local epochs per round')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--client-fraction', type=float, default=1.0,
                       help='Fraction of clients to sample per round')
    
    # Early stopping
    parser.add_argument('--early-stopping', type=int, default=15,
                       help='Early stopping patience (rounds without improvement, 0=disabled)')
    parser.add_argument('--min-delta', type=float, default=0.001,
                       help='Minimum improvement to reset early stopping patience')
    
    # Data augmentation
    parser.add_argument('--augmentation', type=str, default='weak',
                       choices=['none', 'weak', 'strong'],
                       help='Data augmentation level')
    
    # MLflow
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
    
    # Get augmentation config
    aug_config = None
    if args.augmentation == 'weak':
        aug_config = config.AUGMENTATION_WEAK
    elif args.augmentation == 'strong':
        aug_config = config.AUGMENTATION_STRONG
    
    # Setup MLflow
    use_mlflow = config.MLFLOW_ENABLE and not args.no_mlflow
    if use_mlflow:
        experiment_name = f"{config.MLFLOW_EXPERIMENT_PREFIX}_vggface2_federated_{args.method}"
        setup_mlflow(experiment_name, tracking_uri=config.MLFLOW_TRACKING_URI)
    
    # Create clients
    print("\n" + "="*70)
    print("VGGFace2 FEDERATED LEARNING")
    print("="*70)
    print(f"Configuration:")
    print(f"  Number of clients: {args.num_clients}")
    print(f"  Partition strategy: {args.partition}")
    if args.partition == 'non-iid':
        print(f"  Dirichlet alpha: {args.alpha}")
    print(f"  Method: {args.method}")
    print(f"  Model: {args.model}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Epochs per round: {args.epochs_per_round}")
    print(f"  Client fraction: {args.client_fraction}")
    print(f"  Augmentation: {args.augmentation}")
    print("="*70)
    
    clients, global_test_loader, num_classes, partitioner = create_vggface2_federated_clients(
        num_clients=args.num_clients,
        partition_strategy=args.partition,
        alpha=args.alpha,
        method=args.method,
        model_name=args.model,
        aug_config=aug_config
    )
    
    # Create global model
    print(f"\nInitializing global {args.model} model with {num_classes} classes...")
    global_model = create_model(args.model, num_classes=num_classes)
    
    # Start MLflow run
    run_name = f"vggface2_{args.method}_{args.model}_{args.num_clients}clients_{args.partition}"
    if use_mlflow:
        with start_run(run_name=run_name):
            # Log parameters
            log_params({
                'dataset': 'vggface2',
                'num_clients': args.num_clients,
                'partition_strategy': args.partition,
                'alpha': args.alpha if args.partition == 'non-iid' else None,
                'method': args.method,
                'model': args.model,
                'num_rounds': args.rounds,
                'epochs_per_round': args.epochs_per_round,
                'learning_rate': args.lr,
                'client_fraction': args.client_fraction,
                'num_classes': num_classes,
                'batch_size': config.FED_BATCH_SIZE,
                'augmentation': args.augmentation,
                'early_stopping_patience': args.early_stopping if args.early_stopping > 0 else None,
                'early_stopping_min_delta': args.min_delta,
                'device': str(config.DEVICE)
            })
            
            # Create federated server
            server = FederatedServer(
                global_model=global_model,
                method=args.method,
                use_mlflow=use_mlflow
            )
            
            # Run federated training
            print("\n" + "="*70)
            print("STARTING FEDERATED TRAINING")
            print("="*70 + "\n")
            
            # Prepare early stopping parameters
            early_stopping_patience = args.early_stopping if args.early_stopping > 0 else None
            
            # Prepare checkpoint directory
            checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'federated', 'vggface2', args.method)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            history = server.federated_training(
                clients=clients,
                num_rounds=args.rounds,
                epochs_per_round=args.epochs_per_round,
                client_fraction=args.client_fraction,
                test_loader=global_test_loader,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=args.min_delta,
                checkpoint_dir=checkpoint_dir,
                resume_from_checkpoint=args.resume
            )
            
            # Save final global model
            model_path = os.path.join(checkpoint_dir, 
                                     f'{args.model}_{args.num_clients}clients_{args.partition}_final.pth')
            server.save_global_model(model_path)
            print(f"\nFinal global model saved to: {model_path}")
            
            # Create plots directory
            plots_dir = os.path.join(config.PLOTS_DIR, 'federated', 'vggface2', args.method)
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot convergence curves
            import matplotlib.pyplot as plt
            fig = plot_federated_convergence(
                history,
                save_path=os.path.join(plots_dir, 
                                      f'{args.model}_{args.num_clients}clients_convergence.png')
            )
            if use_mlflow and fig is not None:
                log_figure(fig, f"{args.model}_convergence.png")
                plt.close(fig)
            
            # Plot client evolution
            fig_clients = plot_client_evolution(
                history,
                save_path=os.path.join(plots_dir,
                                      f'{args.model}_{args.num_clients}clients_evolution.png')
            )
            if use_mlflow and fig_clients is not None:
                log_figure(fig_clients, f"{args.model}_client_evolution.png")
                plt.close(fig_clients)
            
            # Save history to file
            history_file = os.path.join(plots_dir, 
                                       f'{args.model}_{args.num_clients}clients_history.json')
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            if use_mlflow:
                log_dict(history, f"{args.model}_history.json")
                # Log the global model
                log_model(server.global_model, artifact_path="model",
                         registered_model_name=f"vggface2_{args.method}_{args.model}_global")
            
            print("\n" + "="*70)
            print("FEDERATED LEARNING COMPLETE!")
            print("="*70)
            print(f"Results saved to: {plots_dir}")
            print(f"Model saved to: {model_path}")
            print("="*70 + "\n")
    else:
        # Run without MLflow
        # Create federated server
        server = FederatedServer(
            global_model=global_model,
            method=args.method,
            use_mlflow=False
        )
        
        # Run federated training
        print("\n" + "="*70)
        print("STARTING FEDERATED TRAINING")
        print("="*70 + "\n")
        
        history = server.federated_training(
            clients=clients,
            num_rounds=args.rounds,
            epochs_per_round=args.epochs_per_round,
            client_fraction=args.client_fraction,
            test_loader=global_test_loader
        )
        
        # Save global model
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'federated', 'vggface2', args.method)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, 
                                 f'{args.model}_{args.num_clients}clients_{args.partition}_global.pth')
        server.save_global_model(model_path)
        
        print("\n" + "="*70)
        print("FEDERATED LEARNING COMPLETE!")
        print(f"Model saved to: {model_path}")
        print("="*70 + "\n")


if __name__ == '__main__':
    main()
