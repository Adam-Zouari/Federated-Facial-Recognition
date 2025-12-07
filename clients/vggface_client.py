"""
VGGFace2 dataset client for federated learning.
Uses the official VGGFace2 dataset structure with train/val splits.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import config
from utils.preprocessing import get_train_transforms, get_test_transforms


class VGGFace2Dataset(Dataset):
    """VGGFace2 dataset wrapper with official train/val structure."""
    
    def __init__(self, root_dir, split='train', transform=None, max_identities=500):
        """
        Args:
            root_dir: Root directory of VGGFace2 dataset (should contain train/ and val/ folders)
            split: 'train' or 'val' (VGGFace2 has official train/val splits)
            transform: Image transformations
            max_identities: Maximum number of identities to use
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # VGGFace2 structure: VGGFace2/train/n000001/0001_01.jpg
        if split == 'train':
            self.data_dir = os.path.join(root_dir, 'train')
        elif split in ['val', 'test']:
            # Use val for both val and test
            self.data_dir = os.path.join(root_dir, 'val')
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Scan directory structure
        self.data = []
        self.labels = []
        self.person_to_idx = {}
        
        # Get all person directories
        person_dirs = sorted([d for d in os.listdir(self.data_dir) 
                             if os.path.isdir(os.path.join(self.data_dir, d))])
        
        # Collect images per person
        person_images = []
        for person in person_dirs:
            person_dir = os.path.join(self.data_dir, person)
            images = sorted([f for f in os.listdir(person_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))])
            if len(images) > 0:
                person_images.append((person, images))
        
        # Sort by number of images and take top N identities
        person_images.sort(key=lambda x: len(x[1]), reverse=True)
        if max_identities is not None:
            person_images = person_images[:max_identities]
        
        # Create label mapping and collect all image paths
        for idx, (person, images) in enumerate(person_images):
            self.person_to_idx[person] = idx
            person_dir = os.path.join(self.data_dir, person)
            for img_name in images:
                img_path = os.path.join(person_dir, img_name)
                self.data.append(img_path)
                self.labels.append(idx)
        
        print(f"VGGFace2 {split}: Loaded {len(self.data)} images from {len(person_images)} identities")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        
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
        """Return the number of unique persons."""
        return len(self.person_to_idx)


def load_vggface2_data(root_dir=None, batch_size=None, num_workers=4, max_identities=500):
    """
    Load VGGFace2 dataset with official train/val splits.
    
    Args:
        root_dir: Root directory of VGGFace2 dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        max_identities: Maximum number of identities to use
        
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    root_dir = root_dir or config.DATA_PATHS['vggface2']
    batch_size = batch_size or config.LOCAL_BATCH_SIZE
    
    print(f"Loading VGGFace2 dataset from {root_dir}...")
    
    # Create datasets using official splits
    train_dataset = VGGFace2Dataset(
        root_dir=root_dir,
        split='train',
        transform=get_train_transforms(use_augmentation=True),
        max_identities=max_identities
    )
    
    val_dataset = VGGFace2Dataset(
        root_dir=root_dir,
        split='val',
        transform=get_test_transforms(),
        max_identities=max_identities
    )
    
    # Use val as test (VGGFace2 only has train/val)
    test_dataset = VGGFace2Dataset(
        root_dir=root_dir,
        split='test',  # Will use val folder
        transform=get_test_transforms(),
        max_identities=max_identities
    )
    
    num_classes = train_dataset.get_num_classes()
    print(f"VGGFace2: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
    print(f"Number of identities: {num_classes}")
    
    if len(train_dataset) == 0:
        print("Warning: VGGFace2 dataset is empty!")
        return None, None, None, 0
    
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


def explore_vggface2(root_dir=None):
    """
    Explore VGGFace2 dataset characteristics.
    
    Args:
        root_dir: Root directory of VGGFace2 dataset
    """
    from utils.preprocessing import analyze_dataset, print_dataset_stats
    
    root_dir = root_dir or config.DATA_PATHS['vggface2']
    
    dataset = VGGFace2Dataset(root_dir=root_dir, split='train', max_identities=500)
    stats = analyze_dataset(dataset, "VGGFace2")
    print_dataset_stats(stats)
    
    return stats
