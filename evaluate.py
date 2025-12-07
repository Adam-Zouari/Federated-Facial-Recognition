"""
Evaluation script for all trained models.
Compares local, centralized, and federated models.
"""

import argparse
import os
import torch
import json
import config
from models import create_model
from clients import get_client_data
from utils.metrics import compute_metrics, evaluate_model, print_metrics
from utils.plotting import plot_multi_roc_curves, plot_client_comparison
from centralized.train_global import combine_datasets
from utils.mlflow_utils import (setup_mlflow, start_run, log_params, log_metrics,
                                log_figure, log_dict, end_run)


def load_local_model(client_name, model_name, num_classes):
    """Load a local model checkpoint."""
    model = create_model(model_name, num_classes=num_classes)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'local', client_name, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Local model not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    
    return model


def load_centralized_model(model_name, num_classes):
    """Load a centralized model checkpoint."""
    model = create_model(model_name, num_classes=num_classes)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'centralized', 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Centralized model not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    
    return model


def load_federated_model(method, model_name, num_classes):
    """Load a federated model checkpoint."""
    model = create_model(model_name, num_classes=num_classes)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'federated', method, 
                                  f'{model_name}_global_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Federated model not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    
    return model


def evaluate_all_models(client_name='celeba', model_name='resnet18', use_mlflow=True):
    """
    Evaluate and compare all model types on a client's test set.
    
    Args:
        client_name: Client to evaluate on
        model_name: Model architecture
        use_mlflow: Whether to log to MLflow
    """
    print("\n" + "="*60)
    print(f"EVALUATING ALL MODELS ON {client_name.upper()} TEST SET")
    print("="*60)
    
    # Load test data
    print(f"\nLoading {client_name} test data...")
    _, _, test_loader, num_classes = get_client_data(client_name)
    
    if test_loader is None:
        print(f"Error: Could not load {client_name} data")
        return
    
    results = {}
    
    # Evaluate local model
    print(f"\n1. Evaluating LOCAL model...")
    local_model = load_local_model(client_name, model_name, num_classes)
    if local_model is not None:
        local_results = evaluate_model(local_model, test_loader)
        local_metrics = compute_metrics(
            local_results['labels'],
            local_results['predictions'],
            local_results['scores'],
            num_classes=num_classes
        )
        print_metrics(local_metrics, f"Local Model ({client_name})")
        results['local'] = local_metrics
    
    # Evaluate centralized model
    print(f"\n2. Evaluating CENTRALIZED model...")
    # Note: Centralized model may have different number of classes
    centralized_model = load_centralized_model(model_name, num_classes=1000)  # Approximate
    if centralized_model is not None:
        try:
            central_results = evaluate_model(centralized_model, test_loader)
            central_metrics = compute_metrics(
                central_results['labels'],
                central_results['predictions'],
                central_results['scores'],
                num_classes=num_classes
            )
            print_metrics(central_metrics, "Centralized Model")
            results['centralized'] = central_metrics
        except Exception as e:
            print(f"Could not evaluate centralized model: {e}")
    
    # Evaluate FedAvg model
    print(f"\n3. Evaluating FEDAVG model...")
    fedavg_model = load_federated_model('fedavg', model_name, num_classes)
    if fedavg_model is not None:
        fedavg_results = evaluate_model(fedavg_model, test_loader)
        fedavg_metrics = compute_metrics(
            fedavg_results['labels'],
            fedavg_results['predictions'],
            fedavg_results['scores'],
            num_classes=num_classes
        )
        print_metrics(fedavg_metrics, "FedAvg Model")
        results['fedavg'] = fedavg_metrics
    
    # Evaluate FedProx model
    print(f"\n4. Evaluating FEDPROX model...")
    fedprox_model = load_federated_model('fedprox', model_name, num_classes)
    if fedprox_model is not None:
        fedprox_results = evaluate_model(fedprox_model, test_loader)
        fedprox_metrics = compute_metrics(
            fedprox_results['labels'],
            fedprox_results['predictions'],
            fedprox_results['scores'],
            num_classes=num_classes
        )
        print_metrics(fedprox_metrics, "FedProx Model")
        results['fedprox'] = fedprox_metrics
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    
    for model_type, metrics in results.items():
        print(f"{model_type:<20} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")
    
    # Save comparison plot
    plots_dir = os.path.join(config.PLOTS_DIR, 'comparison')
    os.makedirs(plots_dir, exist_ok=True)
    
    import matplotlib.pyplot as plt
    if len(results) > 1:
        fig = plot_multi_roc_curves(
            results,
            save_path=os.path.join(plots_dir, f'{client_name}_{model_name}_roc_comparison.png')
        )
        if use_mlflow and fig is not None:
            log_figure(fig, f"{client_name}_{model_name}_roc_comparison.png")
            plt.close(fig)
    
    # Save results to JSON
    results_file = os.path.join(plots_dir, f'{client_name}_{model_name}_comparison.json')
    results_serializable = {}
    for model_type, metrics in results.items():
        results_serializable[model_type] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score'])
        }
        if 'roc_auc' in metrics:
            results_serializable[model_type]['roc_auc'] = float(metrics['roc_auc'])
        
        # Log individual model metrics to MLflow
        if use_mlflow:
            log_metrics({
                f'{model_type}_accuracy': float(metrics['accuracy']),
                f'{model_type}_precision': float(metrics['precision']),
                f'{model_type}_recall': float(metrics['recall']),
                f'{model_type}_f1': float(metrics['f1_score'])
            })
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    if use_mlflow:
        log_dict(results_serializable, f"{client_name}_{model_name}_comparison.json")
    
    print(f"\nComparison saved to {results_file}")


def generate_full_report():
    """Generate a comprehensive evaluation report for all models and clients."""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE EVALUATION REPORT")
    print("="*60)
    
    clients = ['celeba', 'vggface2']
    models = ['custom_cnn', 'resnet18', 'mobilenetv2']
    
    for client in clients:
        for model in models:
            try:
                print(f"\n\n{'='*60}")
                print(f"Client: {client.upper()}, Model: {model}")
                print('='*60)
                evaluate_all_models(client_name=client, model_name=model)
            except Exception as e:
                print(f"Error evaluating {client}/{model}: {e}")
    
    print("\n\n" + "="*60)
    print("FULL REPORT GENERATION COMPLETE")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation and Comparison')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'all'],
                       help='Evaluation mode: single client or full report')
    parser.add_argument('--client', type=str, default='celeba',
                       choices=['celeba', 'vggface2'],
                       help='Client to evaluate (for single mode)')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['custom_cnn', 'resnet18', 'mobilenetv2'],
                       help='Model architecture (for single mode)')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    # Setup MLflow
    use_mlflow = config.MLFLOW_ENABLE and not args.no_mlflow
    if use_mlflow:
        experiment_name = f"{config.MLFLOW_EXPERIMENT_PREFIX}_evaluation"
        setup_mlflow(experiment_name, tracking_uri=config.MLFLOW_TRACKING_URI)
    
    if args.mode == 'single':
        if use_mlflow:
            run_name = f"compare_{args.client}_{args.model}"
            with start_run(run_name=run_name):
                log_params({
                    'client': args.client,
                    'model': args.model,
                    'mode': 'single'
                })
                evaluate_all_models(client_name=args.client, model_name=args.model, use_mlflow=use_mlflow)
        else:
            evaluate_all_models(client_name=args.client, model_name=args.model, use_mlflow=False)
    elif args.mode == 'all':
        if use_mlflow:
            run_name = "full_comparison_report"
            with start_run(run_name=run_name):
                log_params({'mode': 'all'})
                generate_full_report()
        else:
            generate_full_report()


if __name__ == '__main__':
    main()
