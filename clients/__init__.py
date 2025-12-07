"""Client dataset handlers package."""

from .celebA_client import load_celeba_data, explore_celeba, CelebADataset
from .vggface_client import load_vggface2_data, explore_vggface2, VGGFace2Dataset


def get_client_data(client_name, **kwargs):
    """
    Get data loaders for a specific client.
    
    Args:
        client_name: 'celeba' or 'vggface2'
        **kwargs: Additional arguments to pass to data loader
        
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    client_name = client_name.lower()
    
    if client_name == 'celeba':
        return load_celeba_data(**kwargs)
    elif client_name in ['vggface2', 'vggface']:
        return load_vggface2_data(**kwargs)
    else:
        raise ValueError(f"Unknown client: {client_name}. Must be 'celeba' or 'vggface2'")


def explore_client_data(client_name):
    """
    Explore dataset for a specific client.
    
    Args:
        client_name: 'celeba' or 'vggface2'
    """
    client_name = client_name.lower()
    
    if client_name == 'celeba':
        return explore_celeba()
    elif client_name in ['vggface2', 'vggface']:
        return explore_vggface2()
    else:
        raise ValueError(f"Unknown client: {client_name}. Must be 'celeba' or 'vggface2'")


__all__ = [
    'CelebADataset',
    'VGGFace2Dataset',
    'load_celeba_data',
    'load_vggface2_data',
    'explore_celeba',
    'explore_vggface2',
    'get_client_data',
    'explore_client_data'
]
