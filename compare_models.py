"""
Model Comparison Script
Compares federated vs locally trained models across multiple metrics and test scenarios.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
from tqdm import tqdm
import json
from datetime import datetime

import config
from models import create_model
from clients import get_client_data


class ModelComparison:
    """Compare two models across various metrics."""
    
    def __init__(self, model1_path, model2_path, model1_name="Model 1", model2_name="Model 2"):
        """
        Initialize model comparison.
        
        Args:
            model1_path: Path to first model checkpoint
            model2_path: Path to second model checkpoint
            model1_name: Display name for first model
            model2_name: Display name for second model
        """
        self.device = config.DEVICE
        self.model1_name = model1_name
        self.model2_name = model2_name
        
        # Load models
        print(f"Loading {model1_name}...")
        self.model1, self.num_classes1 = self.load_model(model1_path)
        
        print(f"Loading {model2_name}...")
        self.model2, self.num_classes2 = self.load_model(model2_path)
        
        print(f"✓ Models loaded successfully")
        print(f"  {model1_name}: {self.num_classes1} classes")
        print(f"  {model2_name}: {self.num_classes2} classes")
        
        # Results storage
        self.results = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'classification': {},
            'verification': {},
            'embedding_quality': {},
            'robustness': {}
        }
        
    def load_model(self, model_path):
        """Load a trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            model_state = checkpoint.get('model_state_dict', checkpoint.get('global_model_state', checkpoint))
        else:
            model_state = checkpoint
        
        # Determine number of classes
        classifier_weight_key = None
        for key in model_state.keys():
            if 'classifier.weight' in key or 'fc.weight' in key:
                classifier_weight_key = key
                break
        
        if classifier_weight_key:
            num_classes = model_state[classifier_weight_key].shape[0]
        else:
            num_classes = 480  # Default
        
        # Create and load model
        model = create_model('mobilenetv2', num_classes=num_classes)
        model.load_state_dict(model_state)
        model.to(self.device)
        model.eval()
        
        return model, num_classes
    
    def test_classification_accuracy(self, dataloader, split_name="Test"):
        """
        Test classification accuracy on a dataset.
        
        Args:
            dataloader: DataLoader for test data
            split_name: Name of the split (for display)
        """
        print(f"\n{'='*60}")
        print(f"Testing Classification Accuracy on {split_name} Set")
        print(f"{'='*60}")
        
        results = {}
        
        for model, model_name in [(self.model1, self.model1_name), 
                                   (self.model2, self.model2_name)]:
            print(f"\nEvaluating {model_name}...")
            
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for images, labels in tqdm(dataloader, desc=f"{model_name}"):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs, _ = model(images, return_embedding=True)
                    probs = F.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            
            # Calculate metrics
            accuracy = (all_preds == all_labels).mean()
            
            # Top-5 accuracy
            top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
            top5_accuracy = np.mean([label in top5_preds[i] 
                                     for i, label in enumerate(all_labels)])
            
            # Per-class accuracy
            unique_labels = np.unique(all_labels)
            per_class_acc = []
            for label in unique_labels:
                mask = all_labels == label
                if mask.sum() > 0:
                    class_acc = (all_preds[mask] == all_labels[mask]).mean()
                    per_class_acc.append(class_acc)
            
            avg_class_acc = np.mean(per_class_acc)
            
            results[model_name] = {
                'accuracy': accuracy,
                'top5_accuracy': top5_accuracy,
                'avg_class_accuracy': avg_class_acc,
                'predictions': all_preds,
                'labels': all_labels,
                'probabilities': all_probs
            }
            
            print(f"\n{model_name} Results:")
            print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
            print(f"  Avg Class Accuracy: {avg_class_acc:.4f} ({avg_class_acc*100:.2f}%)")
        
        self.results['classification'][split_name] = results
        return results
    
    def test_face_verification(self, dataloader, num_pairs=5000):
        """
        Test face verification performance (1:1 matching).
        
        Args:
            dataloader: DataLoader for verification data
            num_pairs: Number of pairs to test (genuine + impostor)
        """
        print(f"\n{'='*60}")
        print(f"Testing Face Verification (1:1 Matching)")
        print(f"{'='*60}")
        
        # Extract embeddings
        embeddings_dict = {}
        
        for model, model_name in [(self.model1, self.model1_name), 
                                   (self.model2, self.model2_name)]:
            print(f"\nExtracting embeddings with {model_name}...")
            
            embeddings_by_class = {}
            
            with torch.no_grad():
                for images, labels in tqdm(dataloader, desc=f"{model_name}"):
                    images = images.to(self.device)
                    labels = labels.cpu().numpy()
                    
                    _, embeddings = model(images, return_embedding=True)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    embeddings = embeddings.cpu().numpy()
                    
                    for emb, label in zip(embeddings, labels):
                        if label not in embeddings_by_class:
                            embeddings_by_class[label] = []
                        embeddings_by_class[label].append(emb)
            
            embeddings_dict[model_name] = embeddings_by_class
        
        # Generate pairs
        print(f"\nGenerating {num_pairs} test pairs...")
        genuine_pairs = []
        impostor_pairs = []
        
        classes = list(embeddings_dict[self.model1_name].keys())
        
        # Generate genuine pairs (same person)
        for _ in range(num_pairs // 2):
            label = np.random.choice(classes)
            embs = embeddings_dict[self.model1_name][label]
            if len(embs) >= 2:
                idx1, idx2 = np.random.choice(len(embs), 2, replace=False)
                genuine_pairs.append((label, idx1, idx2))
        
        # Generate impostor pairs (different persons)
        for _ in range(num_pairs // 2):
            label1, label2 = np.random.choice(classes, 2, replace=False)
            embs1 = embeddings_dict[self.model1_name][label1]
            embs2 = embeddings_dict[self.model1_name][label2]
            idx1 = np.random.choice(len(embs1))
            idx2 = np.random.choice(len(embs2))
            impostor_pairs.append((label1, label2, idx1, idx2))
        
        # Calculate similarities for both models
        results = {}
        
        for model_name in [self.model1_name, self.model2_name]:
            print(f"\nCalculating similarities for {model_name}...")
            
            genuine_scores = []
            impostor_scores = []
            
            embs_dict = embeddings_dict[model_name]
            
            # Genuine pairs
            for label, idx1, idx2 in genuine_pairs:
                emb1 = embs_dict[label][idx1]
                emb2 = embs_dict[label][idx2]
                score = np.dot(emb1, emb2)
                genuine_scores.append(score)
            
            # Impostor pairs
            for label1, label2, idx1, idx2 in impostor_pairs:
                emb1 = embs_dict[label1][idx1]
                emb2 = embs_dict[label2][idx2]
                score = np.dot(emb1, emb2)
                impostor_scores.append(score)
            
            # Calculate ROC curve
            y_true = [1] * len(genuine_scores) + [0] * len(impostor_scores)
            y_scores = genuine_scores + impostor_scores
            
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Calculate EER
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fnr - fpr))
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
            
            # Find threshold at different FPR
            fpr_001_idx = np.where(fpr <= 0.001)[0][-1] if np.any(fpr <= 0.001) else 0
            tpr_at_fpr_001 = tpr[fpr_001_idx]
            
            fpr_01_idx = np.where(fpr <= 0.01)[0][-1] if np.any(fpr <= 0.01) else 0
            tpr_at_fpr_01 = tpr[fpr_01_idx]
            
            results[model_name] = {
                'genuine_scores': genuine_scores,
                'impostor_scores': impostor_scores,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc,
                'eer': eer,
                'tpr_at_fpr_0.001': tpr_at_fpr_001,
                'tpr_at_fpr_0.01': tpr_at_fpr_01
            }
            
            print(f"\n{model_name} Verification Results:")
            print(f"  AUC: {roc_auc:.4f}")
            print(f"  EER: {eer:.4f} ({eer*100:.2f}%)")
            print(f"  TPR @ FPR=0.1%: {tpr_at_fpr_01:.4f} ({tpr_at_fpr_01*100:.2f}%)")
            print(f"  TPR @ FPR=0.01%: {tpr_at_fpr_001:.4f} ({tpr_at_fpr_001*100:.2f}%)")
        
        self.results['verification'] = results
        return results
    
    def test_embedding_quality(self, dataloader, num_classes_to_test=50):
        """
        Test embedding quality (intra-class vs inter-class distances).
        
        Args:
            dataloader: DataLoader for test data
            num_classes_to_test: Number of classes to analyze
        """
        print(f"\n{'='*60}")
        print(f"Testing Embedding Quality")
        print(f"{'='*60}")
        
        results = {}
        
        for model, model_name in [(self.model1, self.model1_name), 
                                   (self.model2, self.model2_name)]:
            print(f"\nAnalyzing {model_name} embeddings...")
            
            embeddings_by_class = {}
            
            with torch.no_grad():
                for images, labels in tqdm(dataloader, desc=f"{model_name}"):
                    images = images.to(self.device)
                    labels = labels.cpu().numpy()
                    
                    _, embeddings = model(images, return_embedding=True)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    embeddings = embeddings.cpu().numpy()
                    
                    for emb, label in zip(embeddings, labels):
                        if label not in embeddings_by_class:
                            embeddings_by_class[label] = []
                        embeddings_by_class[label].append(emb)
            
            # Select classes with enough samples
            valid_classes = [c for c, embs in embeddings_by_class.items() if len(embs) >= 3]
            selected_classes = np.random.choice(valid_classes, 
                                               min(num_classes_to_test, len(valid_classes)), 
                                               replace=False)
            
            # Calculate intra-class distances
            intra_class_distances = []
            for cls in selected_classes:
                embs = np.array(embeddings_by_class[cls])
                for i in range(len(embs)):
                    for j in range(i+1, len(embs)):
                        dist = np.linalg.norm(embs[i] - embs[j])
                        intra_class_distances.append(dist)
            
            # Calculate inter-class distances
            inter_class_distances = []
            for i, cls1 in enumerate(selected_classes):
                for cls2 in selected_classes[i+1:]:
                    embs1 = np.array(embeddings_by_class[cls1])
                    embs2 = np.array(embeddings_by_class[cls2])
                    
                    # Sample to avoid too many comparisons
                    for emb1 in embs1[:5]:
                        for emb2 in embs2[:5]:
                            dist = np.linalg.norm(emb1 - emb2)
                            inter_class_distances.append(dist)
            
            # Calculate metrics
            intra_mean = np.mean(intra_class_distances)
            intra_std = np.std(intra_class_distances)
            inter_mean = np.mean(inter_class_distances)
            inter_std = np.std(inter_class_distances)
            
            # Separation ratio (higher is better)
            separation_ratio = inter_mean / (intra_mean + 1e-8)
            
            results[model_name] = {
                'intra_class_mean': intra_mean,
                'intra_class_std': intra_std,
                'inter_class_mean': inter_mean,
                'inter_class_std': inter_std,
                'separation_ratio': separation_ratio,
                'intra_class_distances': intra_class_distances,
                'inter_class_distances': inter_class_distances
            }
            
            print(f"\n{model_name} Embedding Quality:")
            print(f"  Intra-class distance: {intra_mean:.4f} ± {intra_std:.4f}")
            print(f"  Inter-class distance: {inter_mean:.4f} ± {inter_std:.4f}")
            print(f"  Separation ratio: {separation_ratio:.4f}")
        
        self.results['embedding_quality'] = results
        return results
    
    def test_confidence_calibration(self, dataloader):
        """
        Test model confidence calibration (do predicted probabilities match actual accuracy?).
        
        Args:
            dataloader: DataLoader for test data
        """
        print(f"\n{'='*60}")
        print(f"Testing Confidence Calibration")
        print(f"{'='*60}")
        
        results = {}
        
        for model, model_name in [(self.model1, self.model1_name), 
                                   (self.model2, self.model2_name)]:
            print(f"\nEvaluating {model_name}...")
            
            all_confidences = []
            all_correct = []
            
            with torch.no_grad():
                for images, labels in tqdm(dataloader, desc=f"{model_name}"):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs, _ = model(images, return_embedding=True)
                    probs = F.softmax(outputs, dim=1)
                    confidences, preds = torch.max(probs, 1)
                    
                    correct = (preds == labels).cpu().numpy()
                    confidences = confidences.cpu().numpy()
                    
                    all_confidences.extend(confidences)
                    all_correct.extend(correct)
            
            all_confidences = np.array(all_confidences)
            all_correct = np.array(all_correct)
            
            # Bin by confidence
            bins = np.linspace(0, 1, 11)
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for i in range(len(bins)-1):
                mask = (all_confidences >= bins[i]) & (all_confidences < bins[i+1])
                if mask.sum() > 0:
                    bin_acc = all_correct[mask].mean()
                    bin_conf = all_confidences[mask].mean()
                    bin_accuracies.append(bin_acc)
                    bin_confidences.append(bin_conf)
                    bin_counts.append(mask.sum())
            
            # Expected Calibration Error (ECE)
            ece = 0
            total = len(all_confidences)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
                ece += (count / total) * abs(acc - conf)
            
            results[model_name] = {
                'confidences': all_confidences,
                'correct': all_correct,
                'bin_accuracies': bin_accuracies,
                'bin_confidences': bin_confidences,
                'bin_counts': bin_counts,
                'ece': ece
            }
            
            print(f"\n{model_name} Calibration:")
            print(f"  Expected Calibration Error (ECE): {ece:.4f}")
        
        self.results['calibration'] = results
        return results
    
    def plot_all_results(self, output_dir='plots/comparison'):
        """Generate all comparison plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Generating Comparison Plots")
        print(f"{'='*60}")
        
        # 1. Verification ROC Curves
        if self.results['verification']:
            self.plot_verification_comparison(output_dir)
        
        # 3. Embedding Quality
        if self.results['verification']:
            self.plot_verification_comparison(output_dir)
        
        # 2. Embedding Qualityation
        if 'calibration' in self.results and self.results['calibration']:
            self.plot_calibration_comparison(output_dir)
        
        # 4. Summary Comparisontion
        self.plot_summary_comparison(output_dir)
        
        print(f"\n✓ All plots saved to {output_dir}/")
    
    def plot_classification_comparison(self, output_dir):
        """Plot classification accuracy comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract data
        splits = list(self.results['classification'].keys())
        model1_accs = [self.results['classification'][s][self.model1_name]['accuracy'] 
                       for s in splits]
        model2_accs = [self.results['classification'][s][self.model2_name]['accuracy'] 
                       for s in splits]
        
        model1_top5 = [self.results['classification'][s][self.model1_name]['top5_accuracy'] 
                       for s in splits]
        model2_top5 = [self.results['classification'][s][self.model2_name]['top5_accuracy'] 
                       for s in splits]
        
        # Plot Top-1 Accuracy
        x = np.arange(len(splits))
        width = 0.35
        
        axes[0].bar(x - width/2, [a*100 for a in model1_accs], width, 
                   label=self.model1_name, alpha=0.8, color='#3498db')
        axes[0].bar(x + width/2, [a*100 for a in model2_accs], width, 
                   label=self.model2_name, alpha=0.8, color='#e74c3c')
        axes[0].set_xlabel('Dataset Split', fontsize=12)
        axes[0].set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        axes[0].set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(splits)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot Top-5 Accuracy
        axes[1].bar(x - width/2, [a*100 for a in model1_top5], width, 
                   label=self.model1_name, alpha=0.8, color='#3498db')
        axes[1].bar(x + width/2, [a*100 for a in model2_top5], width, 
                   label=self.model2_name, alpha=0.8, color='#e74c3c')
        axes[1].set_xlabel('Dataset Split', fontsize=12)
        axes[1].set_ylabel('Top-5 Accuracy (%)', fontsize=12)
        axes[1].set_title('Top-5 Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(splits)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/classification_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved classification_comparison.png")
    
    def plot_verification_comparison(self, output_dir):
        """Plot verification performance comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC Curve
        for model_name in [self.model1_name, self.model2_name]:
            data = self.results['verification'][model_name]
            color = '#3498db' if model_name == self.model1_name else '#e74c3c'
            axes[0].plot(data['fpr'], data['tpr'], 
                        label=f"{model_name} (AUC={data['auc']:.4f})", 
                        linewidth=2, color=color)
        
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Score Distribution
        for model_name in [self.model1_name, self.model2_name]:
            data = self.results['verification'][model_name]
            color = '#3498db' if model_name == self.model1_name else '#e74c3c'
            
            axes[1].hist(data['impostor_scores'], bins=50, alpha=0.3, 
                        label=f'{model_name} Impostor', color=color, density=True)
            axes[1].hist(data['genuine_scores'], bins=50, alpha=0.5, 
                        label=f'{model_name} Genuine', color=color, density=True, 
                        edgecolor=color, linewidth=1.5)
        
        axes[1].set_xlabel('Similarity Score', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].set_title('Score Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/verification_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved verification_comparison.png")
    
    def plot_embedding_quality(self, output_dir):
        """Plot embedding quality comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distance distributions
        for model_name in [self.model1_name, self.model2_name]:
            data = self.results['embedding_quality'][model_name]
            color = '#3498db' if model_name == self.model1_name else '#e74c3c'
            
            axes[0].hist(data['intra_class_distances'], bins=50, alpha=0.5, 
                        label=f'{model_name} Intra-class', color=color, density=True)
            axes[0].hist(data['inter_class_distances'], bins=50, alpha=0.3, 
                        label=f'{model_name} Inter-class', color=color, 
                        density=True, edgecolor=color, linewidth=1.5)
        
        axes[0].set_xlabel('L2 Distance', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Embedding Distance Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Separation ratio comparison
        model_names = [self.model1_name, self.model2_name]
        separation_ratios = [self.results['embedding_quality'][m]['separation_ratio'] 
                            for m in model_names]
        
        colors = ['#3498db', '#e74c3c']
        bars = axes[1].bar(model_names, separation_ratios, color=colors, alpha=0.8)
        axes[1].set_ylabel('Separation Ratio', fontsize=12)
        axes[1].set_title('Inter/Intra-class Separation', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, separation_ratios):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/embedding_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved embedding_quality.png")
    
    def plot_calibration_comparison(self, output_dir):
        """Plot confidence calibration comparison."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        
        for model_name in [self.model1_name, self.model2_name]:
            data = self.results['calibration'][model_name]
            color = '#3498db' if model_name == self.model1_name else '#e74c3c'
            
            ax.plot(data['bin_confidences'], data['bin_accuracies'], 
                   'o-', label=f"{model_name} (ECE={data['ece']:.4f})", 
                   linewidth=2, markersize=8, color=color)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')
        
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Confidence Calibration Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/calibration_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved calibration_comparison.png")
    
    def plot_summary_comparison(self, output_dir):
        """Plot overall summary comparison."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Collect metrics
        metrics = []
        model1_values = []
        model2_values = []
        
        # Verification AUC
        if self.results['verification']:
            metrics.append('Verification\nAUC')
            model1_values.append(self.results['verification'][self.model1_name]['auc'] * 100)
            model2_values.append(self.results['verification'][self.model2_name]['auc'] * 100)
        
        # EER (lower is better, so we plot 100-EER for visual consistency)
        if self.results['verification']:
            metrics.append('Verification\n(100-EER)')
            model1_values.append((1 - self.results['verification'][self.model1_name]['eer']) * 100)
            model2_values.append((1 - self.results['verification'][self.model2_name]['eer']) * 100)
        
        # Separation ratio (normalized to 0-100 scale)
        if self.results['embedding_quality']:
            max_ratio = max(self.results['embedding_quality'][self.model1_name]['separation_ratio'],
                           self.results['embedding_quality'][self.model2_name]['separation_ratio'])
            metrics.append('Embedding\nSeparation')
            model1_values.append((self.results['embedding_quality'][self.model1_name]['separation_ratio'] / max_ratio) * 100)
            model2_values.append((self.results['embedding_quality'][self.model2_name]['separation_ratio'] / max_ratio) * 100)
        
        # Calibration (100-ECE, lower ECE is better)
        if 'calibration' in self.results:
            metrics.append('Calibration\n(100-ECE*100)')
            model1_values.append(100 - self.results['calibration'][self.model1_name]['ece'] * 100)
            model2_values.append(100 - self.results['calibration'][self.model2_name]['ece'] * 100)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        model1_values += model1_values[:1]
        model2_values += model2_values[:1]
        angles += angles[:1]
        
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, model1_values, 'o-', linewidth=2, label=self.model1_name, color='#3498db')
        ax.fill(angles, model1_values, alpha=0.15, color='#3498db')
        ax.plot(angles, model2_values, 'o-', linewidth=2, label=self.model2_name, color='#e74c3c')
        ax.fill(angles, model2_values, alpha=0.15, color='#e74c3c')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=9)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        
        plt.title('Overall Model Comparison', size=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved summary_comparison.png")
    
    def save_results(self, output_file='comparison_results.json'):
        """Save comparison results to JSON."""
        # Prepare serializable results
        results_serializable = {
            'model1_name': self.model1_name,
            'model2_name': self.model2_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary': {}
        }
        
        # Classification
        if self.results['classification']:
            results_serializable['summary']['classification'] = {}
            for split in self.results['classification']:
                results_serializable = {
            'model1_name': self.model1_name,
            'model2_name': self.model2_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary': {}
        }
        
        # Verification
        
        # Embedding quality
        if self.results['embedding_quality']:
            results_serializable['summary']['embedding_quality'] = {
                self.model1_name: {
                    'separation_ratio': float(self.results['embedding_quality'][self.model1_name]['separation_ratio'])
                },
                self.model2_name: {
                    'separation_ratio': float(self.results['embedding_quality'][self.model2_name]['separation_ratio'])
                }
            }
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare two trained models')
    parser.add_argument('--model1', type=str, required=True,
                       help='Path to first model checkpoint (e.g., federated model)')
    parser.add_argument('--model2', type=str, required=True,
                       help='Path to second model checkpoint (e.g., local model)')
    parser.add_argument('--name1', type=str, default='Federated',
                       help='Display name for first model')
    parser.add_argument('--name2', type=str, default='Local',
                       help='Display name for second model')
    parser.add_argument('--dataset', type=str, default='vggface2',
                       choices=['vggface2', 'celeba'],
                       help='Dataset to test on')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num_pairs', type=int, default=5000,
                       help='Number of pairs for verification test')
    parser.add_argument('--output_dir', type=str, default='plots/comparison',
                       help='Directory to save plots')
    parser.add_argument('--skip_verification', action='store_true',
                       help='Skip verification test')
    parser.add_argument('--skip_embedding', action='store_true',
                       help='Skip embedding quality test')
    parser.add_argument('--skip_calibration', action='store_true',
                       help='Skip calibration test')
    
    args = parser.parse_args()
    
    # Check model paths
    if not os.path.exists(args.model1):
        print(f"Error: Model 1 not found at {args.model1}")
        return
    if not os.path.exists(args.model2):
        print(f"Error: Model 2 not found at {args.model2}")
        return
    
    print(f"\n{'='*60}")
    print(f"Model Comparison Tool")
    print(f"{'='*60}")
    print(f"Model 1: {args.name1} ({args.model1})")
    print(f"Model 2: {args.name2} ({args.model2})")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    # Initialize comparison
    comparison = ModelComparison(
        args.model1,
        args.model2,
        model1_name=args.name1,
        model2_name=args.name2
    )
    
    # Get dataloaders
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == 'vggface2':
        train_loader, val_loader, test_loader, num_classes = get_client_data(
            client_name='vggface2',
            batch_size=args.batch_size,
            num_workers=4,
            aug_config=None  # No augmentation for testing
        )
    else:
        train_loader, val_loader, test_loader, num_classes = get_client_data(
            client_name='celeba',
            batch_size=args.batch_size,
            num_workers=4,
            aug_config=None  # No augmentation for testing
        )
    
    # Run tests
    if not args.skip_verification:
        comparison.test_face_verification(val_loader, num_pairs=args.num_pairs)
    
    if not args.skip_embedding:
        comparison.test_embedding_quality(val_loader)
    
    if not args.skip_calibration:
        comparison.test_confidence_calibration(val_loader)
    
    # Generate plots
    comparison.plot_all_results(output_dir=args.output_dir)
    
    # Save results
    results_file = os.path.join(args.output_dir, 'comparison_results.json')
    comparison.save_results(results_file)
    
    print(f"\n{'='*60}")
    print(f"Comparison Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
