"""
FedAvg (Federated Averaging) implementation.
"""

import torch
import copy
from collections import OrderedDict


def fedavg_aggregate(client_models, client_weights=None):
    """
    Aggregate client models using FedAvg algorithm.
    
    Args:
        client_models: List of client model state dictionaries
        client_weights: List of weights for each client (e.g., based on data size)
                       If None, simple averaging is used
                       
    Returns:
        Aggregated model state dictionary
    """
    if not client_models:
        raise ValueError("No client models provided for aggregation")
    
    # If no weights provided, use equal weights
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    else:
        # Normalize weights to sum to 1
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
    
    # Initialize aggregated model with zeros
    aggregated_model = OrderedDict()
    
    # Get the structure from the first model
    first_model = client_models[0]
    
    for key in first_model.keys():
        aggregated_model[key] = torch.zeros_like(first_model[key], dtype=torch.float32)
    
    # Weighted sum of all client models
    for client_model, weight in zip(client_models, client_weights):
        for key in aggregated_model.keys():
            aggregated_model[key] += weight * client_model[key].float()
    
    return aggregated_model


class FedAvgServer:
    """
    Server for FedAvg federated learning.
    """
    
    def __init__(self, global_model):
        """
        Args:
            global_model: Initial global model
        """
        self.global_model = global_model
        self.global_params = copy.deepcopy(global_model.state_dict())
    
    def aggregate(self, client_params_list, client_weights=None):
        """
        Aggregate client model parameters.
        
        Args:
            client_params_list: List of client model parameters
            client_weights: List of weights for each client
            
        Returns:
            Aggregated global model parameters
        """
        self.global_params = fedavg_aggregate(client_params_list, client_weights)
        self.global_model.load_state_dict(self.global_params)
        return copy.deepcopy(self.global_params)
    
    def get_global_params(self):
        """Get current global model parameters."""
        return copy.deepcopy(self.global_params)
    
    def set_global_params(self, params):
        """Set global model parameters."""
        self.global_params = copy.deepcopy(params)
        self.global_model.load_state_dict(self.global_params)
