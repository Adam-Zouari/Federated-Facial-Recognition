"""
Visualize VGGFace2 data partitioning across federated clients.
Shows how data is divided using IID or Non-IID strategies.
"""

import argparse
import numpy as np
import os
from collections import defaultdict
from utils.federated_partitioner import VGGFace2Partitioner
from config import DATA_PATHS


def visualize_partition(num_clients=10, strategy='iid', alpha=0.5):
    """
    Partition data and display distribution statistics.
    
    Args:
        num_clients: Number of clients to partition data across
        strategy: 'iid' or 'non_iid'
        alpha: Dirichlet concentration parameter (only for non_iid)
    """
    print("="*80)
    print(f"VGGFace2 Data Partitioning Visualization")
    print(f"Strategy: {strategy.upper()}")
    print(f"Number of Clients: {num_clients}")
    if strategy == 'non_iid':
        print(f"Alpha (Dirichlet): {alpha}")
    print("="*80)
    
    # Get VGGFace2 root directory
    vggface2_root_dir = DATA_PATHS['vggface2']
    
    # Create partitioner (automatically creates partitions in __init__)
    partitioner = VGGFace2Partitioner(
        root_dir=vggface2_root_dir,
        num_clients=num_clients,
        partition_strategy=strategy,
        alpha=alpha,
        seed=42
    )
    
    # Get partitions
    client_partitions = partitioner.client_partitions
    
    print(f"\n" + "="*80)
    
    # Get identity to image count mapping
    identity_to_image_count = {}
    train_dir = os.path.join(vggface2_root_dir, 'train')
    for identity in partitioner.identities:
        identity_dir = os.path.join(train_dir, identity)
        if os.path.exists(identity_dir):
            image_count = len([f for f in os.listdir(identity_dir)
                             if f.endswith(('.jpg', '.png', '.jpeg'))])
            identity_to_image_count[identity] = image_count
    
    total_images = sum(identity_to_image_count.values())
    
    print(f"\nTotal Identities: {len(partitioner.identities)}")
    print(f"Total Images: {total_images:,}")
    print(f"Average Images per Identity: {total_images / len(partitioner.identities):.1f}")
    print("\n" + "="*80)
    
    # Analyze each client
    client_stats = []
    for client_id, identity_list in client_partitions.items():
        num_identities = len(identity_list)
        
        # Count images for this client
        num_images = sum(identity_to_image_count.get(identity, 0) 
                        for identity in identity_list)
        
        client_stats.append({
            'client_id': client_id,
            'num_identities': num_identities,
            'num_images': num_images
        })
        
        print(f"\nClient {client_id}:")
        print(f"  Identities: {num_identities}")
        print(f"  Images: {num_images:,}")
        print(f"  Avg Images per Identity: {num_images / num_identities:.1f}")
        print(f"  Sample Identities: {identity_list[:5]}...")
    
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    
    # Calculate statistics
    identities_per_client = [s['num_identities'] for s in client_stats]
    images_per_client = [s['num_images'] for s in client_stats]
    
    print(f"\nIdentities per Client:")
    print(f"  Min: {min(identities_per_client)}")
    print(f"  Max: {max(identities_per_client)}")
    print(f"  Mean: {np.mean(identities_per_client):.1f}")
    print(f"  Std: {np.std(identities_per_client):.2f}")
    
    print(f"\nImages per Client:")
    print(f"  Min: {min(images_per_client):,}")
    print(f"  Max: {max(images_per_client):,}")
    print(f"  Mean: {np.mean(images_per_client):,.1f}")
    print(f"  Std: {np.std(images_per_client):,.2f}")
    
    # Data imbalance ratio
    imbalance_ratio = max(images_per_client) / min(images_per_client)
    print(f"\nData Imbalance Ratio: {imbalance_ratio:.2f}x")
    print(f"  (Max client has {imbalance_ratio:.2f}x more data than min client)")
    
    # Distribution visualization (text-based bar chart)
    print("\n" + "="*80)
    print("Client Data Distribution (Images):")
    print("="*80)
    
    max_bar_width = 60
    max_images = max(images_per_client)
    
    for stats in client_stats:
        client_id = stats['client_id']
        num_images = stats['num_images']
        bar_length = int((num_images / max_images) * max_bar_width)
        bar = '█' * bar_length
        print(f"Client {client_id:2d}: {bar} {num_images:,} images")
    
    print("\n" + "="*80)
    
    # Identity overlap analysis (for non-iid)
    if strategy == 'non_iid':
        print("\nIdentity Distribution Analysis:")
        print("="*80)
        
        # Count how many clients have each identity
        identity_client_count = defaultdict(int)
        for client_id, identity_list in client_partitions.items():
            for identity in identity_list:
                identity_client_count[identity] += 1
        
        # Statistics on identity sharing
        sharing_counts = list(identity_client_count.values())
        print(f"\nIdentities shared across clients:")
        print(f"  Average clients per identity: {np.mean(sharing_counts):.1f}")
        print(f"  Min clients sharing an identity: {min(sharing_counts)}")
        print(f"  Max clients sharing an identity: {max(sharing_counts)}")
        
        # Count identities by sharing level
        sharing_distribution = defaultdict(int)
        for count in sharing_counts:
            sharing_distribution[count] += 1
        
        print(f"\nSharing distribution:")
        for num_clients_sharing in sorted(sharing_distribution.keys()):
            num_identities = sharing_distribution[num_clients_sharing]
            print(f"  {num_identities} identities shared by {num_clients_sharing} client(s)")
    
    print("\n" + "="*80)
    print("Partitioning Complete!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize VGGFace2 data partitioning for federated learning"
    )
    parser.add_argument(
        '--num_clients',
        type=int,
        default=10,
        help='Number of clients (default: 10)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['iid', 'non-iid'],
        default='iid',
        help='Partitioning strategy: iid or non-iid (default: iid)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Dirichlet concentration parameter for non-iid (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    visualize_partition(
        num_clients=args.num_clients,
        strategy=args.strategy,
        alpha=args.alpha
    )
