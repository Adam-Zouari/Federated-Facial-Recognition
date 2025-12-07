"""
Preprocessing utilities for image datasets.
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from facenet_pytorch import MTCNN
import config


class FaceAligner:
    """Face detection and alignment using MTCNN."""
    
    def __init__(self, device=None):
        self.device = device or config.DEVICE
        self.mtcnn = MTCNN(
            image_size=config.IMG_SIZE[0],
            margin=0,
            keep_all=False,
            post_process=True,
            device=self.device
        )
    
    def align_face(self, img):
        """
        Detect and align face in image.
        
        Args:
            img: PIL Image or numpy array
            
        Returns:
            Aligned face as PIL Image, or original if detection fails
        """
        try:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            # Detect and align
            img_aligned = self.mtcnn(img)
            
            if img_aligned is not None:
                # Convert tensor to PIL Image
                img_aligned = img_aligned.permute(1, 2, 0).cpu().numpy()
                img_aligned = ((img_aligned + 1) * 127.5).astype(np.uint8)
                return Image.fromarray(img_aligned)
            else:
                # If detection fails, return resized original
                return img.resize(config.IMG_SIZE)
        except:
            return img.resize(config.IMG_SIZE)


def get_train_transforms(use_augmentation=True):
    """
    Get training data transformations.
    
    Args:
        use_augmentation: Whether to apply data augmentation
        
    Returns:
        torchvision.transforms.Compose object
    """
    transform_list = [
        transforms.Resize(config.IMG_SIZE),
    ]
    
    if use_augmentation:
        aug_config = config.AUGMENTATION_CONFIG
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip']),
            transforms.RandomRotation(degrees=aug_config['rotation']),
            transforms.ColorJitter(
                brightness=aug_config['color_jitter']['brightness'],
                contrast=aug_config['color_jitter']['contrast'],
                saturation=aug_config['color_jitter']['saturation'],
                hue=aug_config['color_jitter']['hue']
            ),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    return transforms.Compose(transform_list)


def get_test_transforms():
    """
    Get test/validation data transformations.
    
    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])


def split_dataset(dataset, train_ratio=None, val_ratio=None, test_ratio=None, seed=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: PyTorch dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    train_ratio = train_ratio or config.TRAIN_RATIO
    val_ratio = val_ratio or config.VAL_RATIO
    test_ratio = test_ratio or config.TEST_RATIO
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    return train_dataset, val_dataset, test_dataset


def analyze_dataset(dataset, dataset_name="Dataset"):
    """
    Analyze dataset characteristics.
    
    Args:
        dataset: PyTorch dataset
        dataset_name: Name for display
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'name': dataset_name,
        'total_samples': len(dataset),
        'num_classes': 0,
        'class_distribution': {},
        'image_sizes': [],
    }
    
    # Analyze a subset if dataset is too large
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    labels = []
    for idx in indices:
        try:
            img, label = dataset[idx]
            labels.append(label)
            
            # Get image size
            if isinstance(img, torch.Tensor):
                stats['image_sizes'].append(img.shape[-2:])
            elif isinstance(img, Image.Image):
                stats['image_sizes'].append(img.size)
        except:
            continue
    
    # Calculate class statistics
    labels = np.array(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    stats['num_classes'] = len(unique_labels)
    stats['class_distribution'] = {
        int(label): int(count) for label, count in zip(unique_labels, counts)
    }
    
    # Most common image size
    if stats['image_sizes']:
        from collections import Counter
        most_common_size = Counter(map(tuple, stats['image_sizes'])).most_common(1)[0][0]
        stats['common_image_size'] = most_common_size
    
    return stats


def print_dataset_stats(stats):
    """Print dataset statistics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Dataset: {stats['name']}")
    print(f"{'='*60}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Number of classes: {stats['num_classes']}")
    
    if 'common_image_size' in stats:
        print(f"Common image size: {stats['common_image_size']}")
    
    print(f"\nClass distribution (top 10):")
    sorted_classes = sorted(stats['class_distribution'].items(), 
                           key=lambda x: x[1], reverse=True)[:10]
    for class_id, count in sorted_classes:
        print(f"  Class {class_id}: {count} samples")
    print(f"{'='*60}\n")
