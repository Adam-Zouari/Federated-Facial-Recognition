"""
Federated learning training script.
Coordinates multiple clients for collaborative learning.
"""

import argparse
import os
import torch
import copy
import config
from models import create_model
from clients import get_client_data
from federated import FederatedClient, FedProxClient, FederatedServer
from utils.metrics import compute_metrics, evaluate_model, print_metrics
from utils.plotting import (plot_federated_convergence, plot_confusion_matrix,
                           plot_roc_curve, plot_client_comparison)
from utils.mlflow_utils import (setup_mlflow, start_run, log_params, log_metrics,
                                log_model, log_figure, log_dict, end_run)


def create_federated_clients(method='fedavg', model_name='resnet18', num_classes=None):
    """
    Create federated clients for all datasets.
    
    Args:
        method: Federated learning method ('fedavg' or 'fedprox')
        model_name: Model architecture name
        num_classes: Number of classes (if None, uses max from all clients)
        
    Returns:
        List of clients, global test loader, num_classes
    """
    client_configs = [
        {'name': 'celeba', 'max_identities': 500},
        {'name': 'vggface2', 'max_identities': 500}
    ]
    
    clients = []
    all_test_loaders = []
    max_classes = 0
    
    print("\nLoading client datasets...")
    
    for client_config in client_configs:
        client_name = client_config['name']
        
        print(f"\n  Loading {client_name.upper()}...")
        
        # Load data
        train_loader, val_loader, test_loader, client_num_classes = get_client_data(
            client_name, batch_size=config.FED_BATCH_SIZE,
            max_identities=client_config.get('max_identities', 500)
        )
        
        if train_loader is None:
            print(f"    Warning: Skipping {client_name} (no data)")
            continue
        
        max_classes = max(max_classes, client_num_classes)
        all_test_loaders.append((client_name, test_loader))
        
        # Create client model
        client_model = create_model(model_name, num_classes=client_num_classes)
        
        # Create client
        if method == 'fedprox':
            client = FedProxClient(
                client_id=client_name,
                model=client_model,
                train_loader=train_loader,
                val_loader=val_loader
            )
        else:
            client = FederatedClient(
                client_id=client_name,
                model=client_model,
                train_loader=train_loader,
                val_loader=val_loader
            )
        
        clients.append(client)
        print(f"    {client_name}: {client_num_classes} classes")
    
    if not clients:
        raise ValueError("No clients could be created!")
    
    # Use provided num_classes or max from clients
    final_num_classes = num_classes or max_classes
    
    return clients, all_test_loaders, final_num_classes


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--method', type=str, default='fedavg',
                       choices=['fedavg', 'fedprox'],
                       help='Federated learning method')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['custom_cnn', 'resnet18', 'mobilenetv2'],
                       help='Model architecture')
    parser.add_argument('--rounds', type=int, default=config.FED_ROUNDS,
                       help='Number of communication rounds')
    parser.add_argument('--epochs-per-round', type=int, default=config.FED_EPOCHS_PER_ROUND,
                       help='Local epochs per round')
    parser.add_argument('--lr', type=float, default=config.FED_LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--client-fraction', type=float, default=config.FED_CLIENT_FRACTION,
                       help='Fraction of clients per round')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    # Setup MLflow
    use_mlflow = config.MLFLOW_ENABLE and not args.no_mlflow
    if use_mlflow:
        experiment_name = f"{config.MLFLOW_EXPERIMENT_PREFIX}_federated_{args.method}"
        setup_mlflow(experiment_name, tracking_uri=config.MLFLOW_TRACKING_URI)
    
    # Create clients
    clients, test_loaders, num_classes = create_federated_clients(
        method=args.method,
        model_name=args.model
    )
    
    print(f"\nCreated {len(clients)} clients with max {num_classes} classes")
    
    # Create global model
    print(f"\nInitializing global {args.model} model...")
    global_model = create_model(args.model, num_classes=num_classes)
    
    # Start MLflow run
    run_name = f"{args.method}_{args.model}"
    if use_mlflow:
        with start_run(run_name=run_name):
            # Log parameters
            log_params({
                'method': args.method,
                'model': args.model,
                'num_rounds': args.rounds,
                'epochs_per_round': args.epochs_per_round,
                'learning_rate': args.lr,
                'client_fraction': args.client_fraction,
                'num_clients': len(clients),
                'num_classes': num_classes,
                'batch_size': config.FED_BATCH_SIZE,
                'device': str(config.DEVICE)
            })
            
            # Create federated server
            server = FederatedServer(
                global_model=global_model,
                method=args.method,
                use_mlflow=use_mlflow
            )
            
            # Run federated training
            history = server.federated_training(
                clients=clients,
                num_rounds=args.rounds,
                epochs_per_round=args.epochs_per_round,
                client_fraction=args.client_fraction,
                test_loader=None
            )
            
            # Save global model
            checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'federated', args.method)
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = os.path.join(checkpoint_dir, f'{args.model}_global_model.pth')
            server.save_global_model(model_path)
            
            # Plot convergence curves
            plots_dir = os.path.join(config.PLOTS_DIR, 'federated', args.method)
            os.makedirs(plots_dir, exist_ok=True)
            
            import matplotlib.pyplot as plt
            fig = plot_federated_convergence(
                history,
                save_path=os.path.join(plots_dir, f'{args.model}_convergence.png')
            )
            if use_mlflow and fig is not None:
                log_figure(fig, f"{args.model}_convergence.png")
                plt.close(fig)
            
            # Evaluate global model on each client's test set
            print("\n" + "="*60)
            print("EVALUATING GLOBAL MODEL ON CLIENT TEST SETS")
            print("="*60)
            
            client_metrics = {}
            
            for client_name, test_loader in test_loaders:
                print(f"\nEvaluating on {client_name.upper()} test set...")
                
                results = evaluate_model(server.global_model, test_loader)
                unique_labels = len(set(results['labels']))
                
                metrics = compute_metrics(
                    results['labels'],
                    results['predictions'],
                    results['scores'],
                    num_classes=unique_labels
                )
                
                print_metrics(metrics, f"{client_name.upper()} Test Set")
                client_metrics[client_name] = metrics
                
                # Log client-specific test metrics
                if use_mlflow:
                    log_metrics({
                        f'{client_name}_test_accuracy': metrics['accuracy'],
                        f'{client_name}_test_precision': metrics['precision'],
                        f'{client_name}_test_recall': metrics['recall'],
                        f'{client_name}_test_f1': metrics['f1'],
                    })
                
                # Save confusion matrix for each client
                fig_cm = plot_confusion_matrix(
                    metrics['confusion_matrix'],
                    save_path=os.path.join(plots_dir, f'{args.model}_{client_name}_confusion_matrix.png')
                )
                if use_mlflow and fig_cm is not None:
                    log_figure(fig_cm, f"{args.model}_{client_name}_confusion_matrix.png")
                    plt.close(fig_cm)
            
            # Plot client comparison
            fig_comp = plot_client_comparison(
                client_metrics,
                save_path=os.path.join(plots_dir, f'{args.model}_client_comparison.png')
            )
            if use_mlflow and fig_comp is not None:
                log_figure(fig_comp, f"{args.model}_client_comparison.png")
                plt.close(fig_comp)
            
            # Save history to file
            import json
            history_file = os.path.join(plots_dir, f'{args.model}_history.json')
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            if use_mlflow:
                log_dict(history, f"{args.model}_history.json")
                # Log the global model
                log_model(server.global_model, artifact_path="model",
                         registered_model_name=f"{args.method}_{args.model}_global")
            
            print("\n" + "="*60)
            print("Federated Learning Complete!")
            print(f"Results saved to {plots_dir}")
            print("="*60)
    else:
        # Train without MLflow
        server = FederatedServer(
            global_model=global_model,
            method=args.method,
            use_mlflow=False
        )
        
        history = server.federated_training(
            clients=clients,
            num_rounds=args.rounds,
            epochs_per_round=args.epochs_per_round,
            client_fraction=args.client_fraction,
            test_loader=None
        )
        
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'federated', args.method)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f'{args.model}_global_model.pth')
        server.save_global_model(model_path)
        
        plots_dir = os.path.join(config.PLOTS_DIR, 'federated', args.method)
        os.makedirs(plots_dir, exist_ok=True)
        plot_federated_convergence(
            history,
            save_path=os.path.join(plots_dir, f'{args.model}_convergence.png')
        )
        
        print("\n" + "="*60)
        print("EVALUATING GLOBAL MODEL ON CLIENT TEST SETS")
        print("="*60)
        
        client_metrics = {}
        
        for client_name, test_loader in test_loaders:
            print(f"\nEvaluating on {client_name.upper()} test set...")
            
            results = evaluate_model(server.global_model, test_loader)
            unique_labels = len(set(results['labels']))
            
            metrics = compute_metrics(
                results['labels'],
                results['predictions'],
                results['scores'],
                num_classes=unique_labels
            )
            
            print_metrics(metrics, f"{client_name.upper()} Test Set")
            client_metrics[client_name] = metrics
            
            plot_confusion_matrix(
                metrics['confusion_matrix'],
                save_path=os.path.join(plots_dir, f'{args.model}_{client_name}_confusion_matrix.png')
            )
        
        plot_client_comparison(
            client_metrics,
            save_path=os.path.join(plots_dir, f'{args.model}_client_comparison.png')
        )
        
        import json
        history_file = os.path.join(plots_dir, f'{args.model}_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "="*60)
        print("Federated Learning Complete!")
        print(f"Results saved to {plots_dir}")
        print("="*60)


if __name__ == '__main__':
    main()
