"""Federated learning package."""

from .client import FederatedClient
from .fedavg import FedAvgServer, fedavg_aggregate
from .fedprox import FedProxClient, FedProxServer
from .server import FederatedServer


__all__ = [
    'FederatedClient',
    'FedAvgServer',
    'FedProxClient',
    'FedProxServer',
    'FederatedServer',
    'fedavg_aggregate'
]
