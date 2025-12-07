"""
CelebA dataset client for federated learning.
Uses the official CelebA dataset structure with identity labels and evaluation partitions.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import config
from utils.preprocessing import get_train_transforms, get_test_transforms


class CelebADataset(Dataset):
    """CelebA dataset wrapper with identity labels and official train/val/test splits."""
    
    def __init__(self, root_dir, split='train', transform=None, max_identities=None):
        """
        Args:
            root_dir: Root directory of CelebA dataset (should contain img_align_celeba/, Anno/, etc.)
            split: 'train', 'val', or 'test' (uses official evaluation partitions)
            transform: Image transformations
            max_identities: Maximum number of identities to use (for limiting dataset size)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory not found: {self.img_dir}")
        
        # Load evaluation partitions (0=train, 1=val, 2=test)
        partition_file = os.path.join(root_dir, 'list_eval_partition.txt')
        if not os.path.exists(partition_file):
            raise ValueError(f"Partition file not found: {partition_file}")
        
        partitions = pd.read_csv(partition_file, sep=' ', header=None, 
                                names=['filename', 'partition'])
        
        # Filter by split
        split_map = {'train': 0, 'val': 1, 'test': 2}
        if split not in split_map:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        partition_id = split_map[split]
        partitions = partitions[partitions['partition'] == partition_id]
        
        # Load identity annotations
        # Note: Identity file must be provided separately (available upon request from dataset authors)
        # For this implementation, we'll create pseudo-identities from filenames if not available
        identity_file = os.path.join(root_dir, 'Anno', 'identity_CelebA.txt')
        if os.path.exists(identity_file):
            identities = pd.read_csv(identity_file, sep=' ', header=None, 
                                    names=['filename', 'identity'])
        else:
            print(f"Warning: Identity file not found at {identity_file}")
            print("Using pseudo-identities based on image grouping.")
            # Create pseudo-identities (group every 20 images as same person)
            all_files = sorted(partitions['filename'].tolist())
            pseudo_identities = [i // 20 for i in range(len(all_files))]
            identities = pd.DataFrame({
                'filename': all_files,
                'identity': pseudo_identities
            })
        
        # Merge partitions with identities
        self.data = pd.merge(partitions, identities, on='filename')
        
        # Limit number of identities if specified
        if max_identities is not None:
            top_identities = self.data['identity'].value_counts().head(max_identities).index
            self.data = self.data[self.data['identity'].isin(top_identities)]
        
        # Balance dataset ONLY for training split
        # Val/test splits keep all samples for better evaluation
        if split == 'train':
            identity_counts = self.data['identity'].value_counts()
            min_samples = identity_counts.min()
            
            balanced_data = []
            for identity in self.data['identity'].unique():
                identity_data = self.data[self.data['identity'] == identity]
                # Sample min_samples images from this identity
                sampled = identity_data.sample(n=min_samples, random_state=42)
                balanced_data.append(sampled)
            
            self.data = pd.concat(balanced_data, ignore_index=True)
            balance_info = f" ({min_samples} per identity)"
        else:
            balance_info = ""
        
        # Remap identity labels to contiguous range starting from 0
        unique_identities = sorted(self.data['identity'].unique())
        self.identity_map = {old_id: new_id for new_id, old_id in enumerate(unique_identities)}
        self.data['identity'] = self.data['identity'].map(self.identity_map)
        
        # Get file list and labels
        self.image_list = self.data['filename'].tolist()
        self.labels = self.data['identity'].tolist()
        
        print(f"CelebA {split}: {len(self.image_list)} images, {len(unique_identities)} identities{balance_info}")
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_list[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Return a blank image if loading fails
            image = Image.new('RGB', config.IMG_SIZE)
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_num_classes(self):
        """Return the number of unique identities."""
        return len(self.identity_map)


def load_celeba_data(root_dir=None, batch_size=None, num_workers=4, max_identities=1000):
    """
    Load CelebA dataset with official train/val/test splits.
    
    Args:
        root_dir: Root directory of CelebA dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        max_identities: Maximum number of identities to use
        
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    root_dir = root_dir or config.DATA_PATHS['celeba']
    batch_size = batch_size or config.LOCAL_BATCH_SIZE
    
    print(f"Loading CelebA dataset from {root_dir}...")
    
    # Create datasets using official splits
    train_dataset = CelebADataset(
        root_dir=root_dir,
        split='train',
        transform=get_train_transforms(use_augmentation=True),
        max_identities=max_identities
    )
    
    val_dataset = CelebADataset(
        root_dir=root_dir,
        split='val',
        transform=get_test_transforms(),
        max_identities=max_identities
    )
    
    test_dataset = CelebADataset(
        root_dir=root_dir,
        split='test',
        transform=get_test_transforms(),
        max_identities=max_identities
    )
    
    num_classes = train_dataset.get_num_classes()
    print(f"CelebA: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
    print(f"Number of identities: {num_classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes


def explore_celeba(root_dir=None):
    """
    Explore CelebA dataset characteristics.
    
    Args:
        root_dir: Root directory of CelebA dataset
    """
    from utils.preprocessing import analyze_dataset, print_dataset_stats
    
    root_dir = root_dir or config.DATA_PATHS['celeba']
    
    dataset = CelebADataset(root_dir=root_dir, max_identities=500)
    stats = analyze_dataset(dataset, "CelebA")
    print_dataset_stats(stats)
    
    return stats
