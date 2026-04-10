"""
Feature engineering and selection utilities for STM32F4 side-channel analyses.

Implements:
  - PCA-based trace compression
  - Mutual Information (MI) based sample selection
  - Trace normalization
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import entropy


def normalize_traces(traces):
    """
    Standardize traces to mean=0, std=1.
    
    Args:
        traces: (N_traces, N_samples)
        
    Returns:
        Normalized traces
    """
    mean = np.mean(traces, axis=0)
    std = np.std(traces, axis=0)
    return (traces - mean) / (std + 1e-8)


def apply_pca_compression(train_traces, test_traces, n_components=100):
    """
    Apply PCA compression to traces.
    
    Args:
        train_traces: (N_train, N_samples)
        test_traces: (N_test, N_samples)
        n_components: Number of principal components to keep
        
    Returns:
        (train_compressed, test_compressed, variance_explained)
            - train_compressed: (N_train, n_components)
            - test_compressed: (N_test, n_components)
            - variance_explained: float — cumulative explained variance ratio
    """
    pca = PCA(n_components=n_components)
    train_compressed = pca.fit_transform(train_traces)
    test_compressed = pca.transform(test_traces)
    
    variance_explained = np.sum(pca.explained_variance_ratio_)
    
    return train_compressed, test_compressed, variance_explained


def compute_mutual_information(traces, labels):
    """
    Compute mutual information between each time sample and leakage labels.
    
    Uses equal-width binning for trace values and entropy-based MI estimation.
    
    Args:
        traces: (N_traces, N_samples)
        labels: (N_traces,) — target HW or other label (typically 0-8)
        
    Returns:
        mi_scores: (N_samples,) — MI score for each time sample
    """
    n_traces, n_samples = traces.shape
    mi_scores = np.zeros(n_samples)
    
    # Compute entropy of labels
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    p_labels = label_counts / n_traces
    h_labels = entropy(p_labels)
    
    # For each time sample, compute MI with labels
    for sample_idx in range(n_samples):
        trace_sample = traces[:, sample_idx]
        
        # Bin trace values (equal-width binning with 10 bins)
        n_bins = min(10, max(2, n_traces // 100))
        bin_edges = np.linspace(trace_sample.min(), trace_sample.max(), n_bins + 1)
        bin_edges[-1] += 1e-10  # Ensure max value is included
        trace_binned = np.digitize(trace_sample, bin_edges) - 1
        trace_binned = np.clip(trace_binned, 0, n_bins - 1)
        
        # Compute joint distribution
        unique_bins, bin_counts = np.unique(trace_binned, return_counts=True)
        
        # Compute conditional entropy H(labels | trace_sample)
        h_cond = 0.0
        for bin_val in unique_bins:
            bin_mask = (trace_binned == bin_val)
            p_bin = np.sum(bin_mask) / n_traces
            
            labels_in_bin = labels[bin_mask]
            unique_in_bin, counts_in_bin = np.unique(labels_in_bin, return_counts=True)
            p_labels_given_bin = counts_in_bin / np.sum(bin_mask)
            
            h_cond += p_bin * entropy(p_labels_given_bin)
        
        # MI = H(labels) - H(labels | trace_sample)
        mi_scores[sample_idx] = max(0, h_labels - h_cond)
    
    return mi_scores


def select_top_mi_samples(traces, labels, n_samples_select):
    """
    Select top-k time samples by mutual information with labels.
    
    Args:
        traces: (N_traces, N_samples)
        labels: (N_traces,) — HW or other labels
        n_samples_select: Number of samples to select
        
    Returns:
        (traces_selected, selected_indices)
            - traces_selected: (N_traces, n_samples_select)
            - selected_indices: (n_samples_select,) — original indices
    """
    print(f"  Computing MI scores for {traces.shape[1]} samples...")
    mi_scores = compute_mutual_information(traces, labels)
    
    # Select top n_samples_select
    selected_indices = np.argsort(-mi_scores)[:n_samples_select]
    selected_indices = np.sort(selected_indices)  # Keep in order for interpretability
    
    traces_selected = traces[:, selected_indices]
    
    print(f"  Selected {n_samples_select} samples with highest MI")
    print(f"    Top MI scores: {mi_scores[selected_indices[:5]]}")
    print(f"    Bottom MI scores: {mi_scores[selected_indices[-5:]]}")
    
    return traces_selected, selected_indices


def select_variance_threshold(traces, percentile=90):
    """
    Select time samples by variance threshold (simple baseline feature selection).
    
    Args:
        traces: (N_traces, N_samples)
        percentile: Percentile threshold (e.g., 90 = keep top 10%)
        
    Returns:
        (traces_selected, selected_indices)
    """
    variances = np.var(traces, axis=0)
    threshold = np.percentile(variances, percentile)
    
    selected_indices = np.where(variances >= threshold)[0]
    traces_selected = traces[:, selected_indices]
    
    print(f"  Variance-based selection (>{percentile}th percentile):")
    print(f"    Variance threshold: {threshold:.4f}")
    print(f"    Selected {len(selected_indices)} / {traces.shape[1]} samples")
    
    return traces_selected, selected_indices


if __name__ == "__main__":
    # Self-test
    print("Testing ml_features_cortexm4.py...")
    
    # Create dummy data
    np.random.seed(42)
    dummy_traces = np.random.randn(100, 200)
    dummy_labels = np.random.randint(0, 9, 100)
    
    # Test normalization
    norm_traces = normalize_traces(dummy_traces)
    print(f"✓ Normalization: mean={np.mean(norm_traces):.2e}, std={np.std(norm_traces):.4f}")
    
    # Test PCA
    train, test, var_exp = apply_pca_compression(dummy_traces[:80], dummy_traces[80:], n_components=50)
    print(f"✓ PCA: {dummy_traces.shape} -> {train.shape}, variance_explained={var_exp:.4f}")
    
    # Test MI
    mi_scores = compute_mutual_information(dummy_traces, dummy_labels)
    print(f"✓ MI computation: {len(mi_scores)} scores, max={mi_scores.max():.4f}")
    
    # Test MI selection
    selected, indices = select_top_mi_samples(dummy_traces, dummy_labels, 50)
    print(f"✓ MI selection: {dummy_traces.shape} -> {selected.shape}")
    
    # Test variance selection
    selected_var, indices_var = select_variance_threshold(dummy_traces, percentile=90)
    print(f"✓ Variance selection: {dummy_traces.shape} -> {selected_var.shape}")
