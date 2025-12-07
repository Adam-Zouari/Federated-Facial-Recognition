"""Utility package initialization."""

from .preprocessing import (
    FaceAligner, 
    get_train_transforms, 
    get_test_transforms,
    split_dataset,
    analyze_dataset,
    print_dataset_stats
)

from .metrics import (
    compute_metrics,
    evaluate_model,
    print_metrics,
    MetricsTracker,
    AverageMeter
)

from .plotting import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_multi_roc_curves,
    plot_embeddings_tsne,
    plot_embeddings_pca,
    plot_federated_convergence,
    plot_client_comparison
)

__all__ = [
    'FaceAligner',
    'get_train_transforms',
    'get_test_transforms',
    'split_dataset',
    'analyze_dataset',
    'print_dataset_stats',
    'compute_metrics',
    'evaluate_model',
    'print_metrics',
    'MetricsTracker',
    'AverageMeter',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_multi_roc_curves',
    'plot_embeddings_tsne',
    'plot_embeddings_pca',
    'plot_federated_convergence',
    'plot_client_comparison'
]
