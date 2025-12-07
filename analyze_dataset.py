"""
Dataset Analysis Tool
Analyzes CelebA and VGGFace2 datasets to provide comprehensive statistics
and distribution information before training.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import config


def analyze_celeba(root_dir=None):
    """
    Analyze CelebA dataset distribution and statistics.
    
    Args:
        root_dir: Root directory of CelebA dataset
    
    Returns:
        Dictionary containing analysis results
    """
    root_dir = root_dir or config.DATA_PATHS['celeba']
    print(f"\n{'='*80}")
    print(f"CELEBA DATASET ANALYSIS")
    print(f"{'='*80}")
    print(f"Dataset path: {root_dir}\n")
    
    # Load partition file
    partition_file = os.path.join(root_dir, 'list_eval_partition.txt')
    if not os.path.exists(partition_file):
        print(f"Error: Partition file not found at {partition_file}")
        return None
    
    partitions = pd.read_csv(partition_file, sep=' ', header=None, 
                            names=['filename', 'partition'])
    
    # Load identity file
    identity_file = os.path.join(root_dir, 'Anno', 'Anno', 'identity_CelebA.txt')
    if not os.path.exists(identity_file):
        identity_file = os.path.join(root_dir, 'Anno', 'identity_CelebA.txt')
    
    if os.path.exists(identity_file):
        identities = pd.read_csv(identity_file, sep=' ', header=None, 
                                names=['filename', 'identity'])
    else:
        print(f"Warning: Identity file not found")
        return None
    
    # Merge data
    data = pd.merge(partitions, identities, on='filename')
    
    # Split analysis
    split_names = {0: 'Train', 1: 'Validation', 2: 'Test'}
    results = {}
    
    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total images: {len(data):,}")
    print(f"Total unique identities: {data['identity'].nunique():,}")
    print(f"Average images per identity: {len(data) / data['identity'].nunique():.2f}")
    
    # Per-split analysis
    print("\n\nSPLIT DISTRIBUTION")
    print("-" * 80)
    for split_id, split_name in split_names.items():
        split_data = data[data['partition'] == split_id]
        num_images = len(split_data)
        num_identities = split_data['identity'].nunique()
        
        print(f"\n{split_name}:")
        print(f"  Images: {num_images:,} ({num_images/len(data)*100:.1f}%)")
        print(f"  Unique identities: {num_identities:,}")
        print(f"  Avg images per identity: {num_images/num_identities:.2f}")
        
        # Identity distribution
        identity_counts = split_data['identity'].value_counts()
        print(f"  Min images per identity: {identity_counts.min()}")
        print(f"  Max images per identity: {identity_counts.max()}")
        print(f"  Median images per identity: {identity_counts.median():.0f}")
        print(f"  Std images per identity: {identity_counts.std():.2f}")
        
        results[split_name.lower()] = {
            'num_images': num_images,
            'num_identities': num_identities,
            'identity_counts': identity_counts
        }
    
    # Identity overlap analysis
    print("\n\nIDENTITY OVERLAP BETWEEN SPLITS")
    print("-" * 80)
    train_ids = set(data[data['partition'] == 0]['identity'])
    val_ids = set(data[data['partition'] == 1]['identity'])
    test_ids = set(data[data['partition'] == 2]['identity'])
    
    print(f"Train-only identities: {len(train_ids - val_ids - test_ids):,}")
    print(f"Val-only identities: {len(val_ids - train_ids - test_ids):,}")
    print(f"Test-only identities: {len(test_ids - train_ids - val_ids):,}")
    print(f"Identities in all splits: {len(train_ids & val_ids & test_ids):,}")
    print(f"Identities in train & val: {len(train_ids & val_ids):,}")
    print(f"Identities in train & test: {len(train_ids & test_ids):,}")
    print(f"Identities in val & test: {len(val_ids & test_ids):,}")
    
    # Recommendations
    print("\n\nRECOMMENDATIONS")
    print("-" * 80)
    train_counts = results['train']['identity_counts']
    min_samples = train_counts.min()
    
    if min_samples < 5:
        print(f"⚠ WARNING: Some identities have very few samples (min={min_samples})")
        print(f"  Recommendation: Filter identities with < 10 samples for balanced training")
        identities_with_10plus = (train_counts >= 10).sum()
        print(f"  Identities with ≥10 samples: {identities_with_10plus:,}")
    
    identities_with_5plus = (train_counts >= 5).sum()
    identities_with_20plus = (train_counts >= 20).sum()
    
    # Count how many identities have each specific number of samples
    print(f"\nIdentity distribution by exact sample count:")
    print("-" * 80)
    sample_count_distribution = train_counts.value_counts().sort_index()
    
    for num_samples, num_identities in sample_count_distribution.items():
        print(f"  {num_identities:5,} identities: {num_samples:3d} images each")
    
    print(f"\nCumulative distribution:")
    print(f"  ≥5 samples: {identities_with_5plus:,} identities")
    print(f"  ≥10 samples: {(train_counts >= 10).sum():,} identities")
    print(f"  ≥15 samples: {(train_counts >= 15).sum():,} identities")
    print(f"  ≥20 samples: {identities_with_20plus:,} identities")
    print(f"  ≥30 samples: {(train_counts >= 30).sum():,} identities")
    print(f"  ≥40 samples: {(train_counts >= 40).sum():,} identities")
    
    return results


def analyze_vggface2(root_dir=None):
    """
    Analyze VGGFace2 dataset distribution and statistics.
    
    Args:
        root_dir: Root directory of VGGFace2 dataset
    
    Returns:
        Dictionary containing analysis results
    """
    root_dir = root_dir or config.DATA_PATHS['vggface2']
    print(f"\n{'='*80}")
    print(f"VGGFACE2 DATASET ANALYSIS")
    print(f"{'='*80}")
    print(f"Dataset path: {root_dir}\n")
    
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    
    results = {}
    
    # Analyze train
    if os.path.exists(train_dir):
        print("TRAIN SET")
        print("-" * 80)
        train_identities = [d for d in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, d))]
        
        identity_counts = []
        for identity in train_identities:
            identity_path = os.path.join(train_dir, identity)
            num_images = len([f for f in os.listdir(identity_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            identity_counts.append(num_images)
        
        identity_counts = np.array(identity_counts)
        total_images = identity_counts.sum()
        
        print(f"Total identities: {len(train_identities):,}")
        print(f"Total images: {total_images:,}")
        print(f"Avg images per identity: {identity_counts.mean():.2f}")
        print(f"Min images per identity: {identity_counts.min()}")
        print(f"Max images per identity: {identity_counts.max()}")
        print(f"Median images per identity: {np.median(identity_counts):.0f}")
        print(f"Std images per identity: {identity_counts.std():.2f}")
        
        results['train'] = {
            'num_identities': len(train_identities),
            'num_images': total_images,
            'identity_counts': identity_counts
        }
    
    # Analyze test
    if os.path.exists(test_dir):
        print("\n\nTEST SET")
        print("-" * 80)
        test_identities = [d for d in os.listdir(test_dir) 
                         if os.path.isdir(os.path.join(test_dir, d))]
        
        identity_counts = []
        for identity in test_identities:
            identity_path = os.path.join(test_dir, identity)
            num_images = len([f for f in os.listdir(identity_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            identity_counts.append(num_images)
        
        identity_counts = np.array(identity_counts)
        total_images = identity_counts.sum()
        
        print(f"Total identities: {len(test_identities):,}")
        print(f"Total images: {total_images:,}")
        print(f"Avg images per identity: {identity_counts.mean():.2f}")
        print(f"Min images per identity: {identity_counts.min()}")
        print(f"Max images per identity: {identity_counts.max()}")
        print(f"Median images per identity: {np.median(identity_counts):.0f}")
        print(f"Std images per identity: {identity_counts.std():.2f}")
        
        results['test'] = {
            'num_identities': len(test_identities),
            'num_images': total_images,
            'identity_counts': identity_counts
        }
    
    return results


def plot_distribution(results, dataset_name, save_dir='./plots/analysis'):
    """
    Create visualization plots for dataset distribution.
    
    Args:
        results: Analysis results dictionary
        dataset_name: Name of dataset (celeba or vggface2)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot samples per identity distribution
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (split_name, split_data) in enumerate(results.items()):
        if 'identity_counts' in split_data:
            counts = split_data['identity_counts']
            
            axes[idx].hist(counts, bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel('Images per Identity')
            axes[idx].set_ylabel('Number of Identities')
            axes[idx].set_title(f'{split_name.capitalize()} - Distribution')
            axes[idx].grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f"Mean: {counts.mean():.1f}\nMedian: {np.median(counts):.0f}\nStd: {counts.std():.1f}"
            axes[idx].text(0.95, 0.95, stats_text,
                          transform=axes[idx].transAxes,
                          verticalalignment='top',
                          horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{dataset_name}_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution plot to {save_path}")
    plt.close()


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze dataset distribution')
    parser.add_argument('--dataset', type=str, choices=['celeba', 'vggface2', 'both'],
                       default='both', help='Which dataset to analyze')
    parser.add_argument('--plot', action='store_true', help='Generate distribution plots')
    
    args = parser.parse_args()
    
    if args.dataset in ['celeba', 'both']:
        celeba_results = analyze_celeba()
        if celeba_results and args.plot:
            plot_distribution(celeba_results, 'celeba')
    
    if args.dataset in ['vggface2', 'both']:
        vggface2_results = analyze_vggface2()
        if vggface2_results and args.plot:
            plot_distribution(vggface2_results, 'vggface2')
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
