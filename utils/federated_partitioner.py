"""
Federated dataset partitioner for splitting datasets across multiple clients.
Supports IID and Non-IID (Dirichlet) partitioning strategies.
"""

import os
import numpy as np
from collections import defaultdict


class VGGFace2Partitioner:
    """
    Partition VGGFace2 dataset across multiple federated clients.
    
    Strategies:
    - IID: Each client gets a balanced subset of identities
    - Non-IID (Dirichlet): Identities distributed following Dirichlet distribution
    """
    
    def __init__(self, root_dir, num_clients=10, partition_strategy='iid', alpha=0.5, seed=42):
        """
        Args:
            root_dir: Root directory of VGGFace2 (contains train/ and val/ folders)
            num_clients: Number of federated clients
            partition_strategy: 'iid' or 'non-iid'
            alpha: Dirichlet concentration parameter (for non-iid), lower = more skewed
                   Note: Actual alpha used is alpha * 100 (e.g., 0.01 → 1.0)
            seed: Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.num_clients = num_clients
        self.partition_strategy = partition_strategy
        self.alpha = alpha * 100  # Scale alpha by 100
        self.seed = seed
        
        np.random.seed(seed)
        
        # Scan the dataset
        self.train_dir = os.path.join(root_dir, 'train')
        self.val_dir = os.path.join(root_dir, 'val')
        
        if not os.path.exists(self.train_dir):
            raise ValueError(f"Training directory not found: {self.train_dir}")
        
        # Get all identities
        self.identities = sorted([d for d in os.listdir(self.train_dir)
                                 if os.path.isdir(os.path.join(self.train_dir, d))])
        
        print(f"\nVGGFace2 Federated Partitioner:")
        print(f"  Total identities: {len(self.identities)}")
        print(f"  Number of clients: {num_clients}")
        print(f"  Partition strategy: {partition_strategy}")
        
        # Create partitions
        self.client_partitions = self._create_partitions()
        self._print_partition_stats()
    
    def _create_partitions(self):
        """Create client partitions based on strategy."""
        if self.partition_strategy == 'iid':
            return self._iid_partition()
        elif self.partition_strategy == 'non-iid':
            return self._non_iid_partition()
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition_strategy}")
    
    def _iid_partition(self):
        """
        IID partitioning: Randomly distribute identities evenly across clients.
        Each client gets approximately the same number of identities.
        """
        partitions = defaultdict(list)
        
        # Shuffle identities
        shuffled_identities = self.identities.copy()
        np.random.shuffle(shuffled_identities)
        
        # Distribute evenly
        identities_per_client = len(self.identities) // self.num_clients
        
        for client_id in range(self.num_clients):
            start_idx = client_id * identities_per_client
            end_idx = start_idx + identities_per_client
            
            # Last client gets remaining identities
            if client_id == self.num_clients - 1:
                end_idx = len(shuffled_identities)
            
            partitions[client_id] = shuffled_identities[start_idx:end_idx]
        
        return dict(partitions)
    
    def _non_iid_partition(self):
        """
        Non-IID partitioning using Dirichlet distribution.
        Creates heterogeneous data distribution across clients.
        
        Lower alpha (e.g., 0.1) = more skewed, higher alpha (e.g., 10) = more balanced
        """
        partitions = defaultdict(list)
        
        # Sample a single Dirichlet distribution to split ALL identities
        # This creates correlated client sizes based on alpha
        proportions = np.random.dirichlet([self.alpha] * self.num_clients)
        
        # Calculate number of identities per client based on proportions
        num_identities = len(self.identities)
        identities_per_client = (proportions * num_identities).astype(int)
        
        # Adjust to ensure we assign all identities (handle rounding)
        diff = num_identities - identities_per_client.sum()
        if diff > 0:
            # Add remaining identities to clients with largest proportions
            indices = np.argsort(proportions)[::-1][:diff]
            identities_per_client[indices] += 1
        elif diff < 0:
            # Remove excess identities from clients with smallest proportions
            indices = np.argsort(proportions)[:abs(diff)]
            identities_per_client[indices] -= 1
        
        # Shuffle identities for random assignment
        shuffled_identities = self.identities.copy()
        np.random.shuffle(shuffled_identities)
        
        # Assign identities to clients based on calculated counts
        start_idx = 0
        for client_id in range(self.num_clients):
            num_for_client = identities_per_client[client_id]
            end_idx = start_idx + num_for_client
            partitions[client_id] = shuffled_identities[start_idx:end_idx]
            start_idx = end_idx
        
        return dict(partitions)
    
    def _print_partition_stats(self):
        """Print statistics about the partitions."""
        print(f"\nPartition Statistics:")
        
        identities_per_client = []
        images_per_client = []
        
        for client_id in range(self.num_clients):
            identities = self.client_partitions.get(client_id, [])
            num_identities = len(identities)
            identities_per_client.append(num_identities)
            
            # Count images for this client
            num_images = 0
            for identity in identities:
                identity_dir = os.path.join(self.train_dir, identity)
                if os.path.exists(identity_dir):
                    num_images += len([f for f in os.listdir(identity_dir)
                                      if f.endswith(('.jpg', '.png', '.jpeg'))])
            images_per_client.append(num_images)
            
            print(f"  Client {client_id}: {num_identities} identities, ~{num_images} images")
        
        print(f"\nIdentities Distribution:")
        print(f"  Mean: {np.mean(identities_per_client):.1f}")
        print(f"  Std: {np.std(identities_per_client):.1f}")
        print(f"  Min: {np.min(identities_per_client)}")
        print(f"  Max: {np.max(identities_per_client)}")
        
        print(f"\nImages Distribution:")
        print(f"  Mean: {np.mean(images_per_client):.1f}")
        print(f"  Std: {np.std(images_per_client):.1f}")
        print(f"  Min: {np.min(images_per_client)}")
        print(f"  Max: {np.max(images_per_client)}")
    
    def get_client_identities(self, client_id):
        """
        Get list of identities assigned to a specific client.
        
        Args:
            client_id: Client ID (0 to num_clients-1)
            
        Returns:
            List of identity names (folder names)
        """
        if client_id < 0 or client_id >= self.num_clients:
            raise ValueError(f"Invalid client_id: {client_id}. Must be 0-{self.num_clients-1}")
        
        return self.client_partitions.get(client_id, [])
    
    def save_partitions(self, save_path):
        """
        Save partition information to file.
        
        Args:
            save_path: Path to save partition file
        """
        import json
        
        partition_info = {
            'num_clients': self.num_clients,
            'partition_strategy': self.partition_strategy,
            'alpha': self.alpha if self.partition_strategy == 'non-iid' else None,
            'seed': self.seed,
            'total_identities': len(self.identities),
            'partitions': {str(k): v for k, v in self.client_partitions.items()}
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(partition_info, f, indent=2)
        
        print(f"\nPartition info saved to: {save_path}")
    
    @staticmethod
    def load_partitions(load_path):
        """
        Load partition information from file.
        
        Args:
            load_path: Path to partition file
            
        Returns:
            Dictionary with partition information
        """
        import json
        
        with open(load_path, 'r') as f:
            partition_info = json.load(f)
        
        # Convert string keys back to integers
        partition_info['partitions'] = {int(k): v for k, v in partition_info['partitions'].items()}
        
        return partition_info
