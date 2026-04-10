"""
Feature selection experiment: Random Forest with PCA and MI on STM32F4.

This script investigates how different feature selection strategies impact
Random Forest performance on side-channel key recovery.

Experiments:
  1. Full traces (baseline)
  2. PCA compression (vary n_components: 10, 30, 50, 100, 200, full)
  3. MI-based sample selection (vary n_samples: 50, 100, 200, 500, full)
  4. Variance-based sample selection (percentile: 75, 85, 95)

Output: Plots comparing key rank and computational efficiency across strategies
"""

import numpy as np
import argparse
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Import local utilities
from ml_data_utils_cortexm4 import (
    load_stm32f4_dataset,
    hamming_weight,
    AES,
    align_class_probabilities,
)
from ml_features_cortexm4 import (
    normalize_traces,
    apply_pca_compression,
    select_top_mi_samples,
    select_variance_threshold,
)


def compute_key_rank_from_hw_proba(hw_proba, plaintexts, key, byte_idx):
    """Compute key rank from HW probabilities."""
    n_test = hw_proba.shape[0]
    true_key_byte = key[byte_idx]
    
    scores = np.zeros(256)
    pt_byte = plaintexts[:, byte_idx]
    
    for k in range(256):
        pt_xor_k = pt_byte ^ k
        sbox_out = AES.SBOX[pt_xor_k].astype(np.uint8)
        expected_hw = hamming_weight(sbox_out)
        
        for i in range(n_test):
            hw = expected_hw[i]
            if 0 <= hw <= 8:
                scores[k] += np.log(hw_proba[i, hw] + 1e-40)
    
    rank = np.sum(scores >= scores[true_key_byte])
    return int(rank)


def train_and_evaluate_rf(train_traces, train_labels, test_traces, test_labels,
                           plaintexts, key, byte_idx, seed=42):
    """Train RF and compute key rank."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        random_state=seed + byte_idx,
        n_jobs=-1,
    )
    
    model.fit(train_traces, train_labels)
    hw_proba = align_class_probabilities(model, model.predict_proba(test_traces), n_classes=9)
    
    key_rank = compute_key_rank_from_hw_proba(
        hw_proba, plaintexts, key, byte_idx
    )
    
    return key_rank


def main(
    hdf5_path="traces.hdf5",
    n_bytes=4,  # Test on first N bytes (for speed)
    seed=42,
    output_dir="results/stm32f4_ml",
):
    """
    Main function: run feature selection experiments.
    
    Args:
        hdf5_path: Path to traces.hdf5
        n_bytes: Number of bytes to test on (for runtime)
        seed: Random seed
        output_dir: Output directory
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Feature Selection] Loading dataset from {hdf5_path}...")
    
    try:
        dataset = load_stm32f4_dataset(
            hdf5_path,
            train_split=0.8,
            seed=seed,
            target_byte=None,
        )
    except FileNotFoundError:
        print(f"✗ Error: {hdf5_path} not found")
        return
    
    train_traces = dataset["train_traces"]
    train_labels = dataset["train_labels"]
    test_traces = dataset["test_traces"]
    test_labels = dataset["test_labels"]
    key = dataset["key"]
    plaintexts = dataset["plaintexts"]
    
    n_train = train_traces.shape[0]
    n_test = test_traces.shape[0]
    n_samples = train_traces.shape[1]
    
    print(f"  Train: {n_train}, Test: {n_test}, Samples: {n_samples}")
    
    # Normalize traces
    print("\n[Feature Selection] Normalizing traces...")
    train_traces = normalize_traces(train_traces)
    test_traces = normalize_traces(test_traces)
    
    n_bytes_eval = min(n_bytes, 16)
    
    # =================
    # Experiment 1: Full traces (baseline)
    # =================
    print(f"\n[Feature Selection] Experiment 1: Full traces (baseline)...")
    rankings_full = []
    
    for byte_idx in tqdm(range(n_bytes_eval)):
        train_hw = train_labels[:, byte_idx]
        test_hw = test_labels[:, byte_idx]
        
        rank = train_and_evaluate_rf(
            train_traces, train_hw, test_traces, test_hw,
            plaintexts, key, byte_idx, seed
        )
        rankings_full.append(rank)
    
    avg_rank_full = np.mean(rankings_full)
    print(f"  Full traces: avg_rank={avg_rank_full:.2f}")
    
    # =================
    # Experiment 2: PCA compression
    # =================
    print(f"\n[Feature Selection] Experiment 2: PCA compression...")
    pca_components = [10, 30, 50, 100, 200]
    pca_results = {n: [] for n in pca_components}
    
    for n_comp in pca_components:
        if n_comp >= n_samples:
            print(f"  Skipping PCA(n_comp={n_comp}) — exceeds trace length")
            continue
        
        print(f"  Testing PCA(n_components={n_comp})...")
        train_pca, test_pca, var_exp = apply_pca_compression(
            train_traces, test_traces, n_components=n_comp
        )
        print(f"    Variance explained: {var_exp:.4f}")
        
        for byte_idx in tqdm(range(n_bytes_eval), leave=False):
            train_hw = train_labels[:, byte_idx]
            test_hw = test_labels[:, byte_idx]
            
            rank = train_and_evaluate_rf(
                train_pca, train_hw, test_pca, test_hw,
                plaintexts, key, byte_idx, seed
            )
            pca_results[n_comp].append(rank)
    
    for n_comp in pca_components:
        if len(pca_results[n_comp]) > 0:
            avg_rank = np.mean(pca_results[n_comp])
            print(f"  PCA({n_comp}): avg_rank={avg_rank:.2f}")
    
    # =================
    # Experiment 3: MI-based sample selection
    # =================
    print(f"\n[Feature Selection] Experiment 3: MI-based sample selection...")
    mi_sample_counts = [50, 100, 200, 500]
    mi_results = {n: [] for n in mi_sample_counts}
    
    for n_select in mi_sample_counts:
        if n_select >= n_samples:
            print(f"  Skipping MI selection (n_samples={n_select}) — exceeds trace length")
            continue
        
        print(f"  Testing MI selection (n_samples={n_select})...")
        
        # Use first byte's labels to select samples (representative)
        first_byte_labels = train_labels[:, 0]
        train_mi, mi_indices = select_top_mi_samples(
            train_traces, first_byte_labels, n_select
        )
        test_mi = test_traces[:, mi_indices]
        
        for byte_idx in tqdm(range(n_bytes_eval), leave=False):
            train_hw = train_labels[:, byte_idx]
            test_hw = test_labels[:, byte_idx]
            
            rank = train_and_evaluate_rf(
                train_mi, train_hw, test_mi, test_hw,
                plaintexts, key, byte_idx, seed
            )
            mi_results[n_select].append(rank)
    
    for n_select in mi_sample_counts:
        if len(mi_results[n_select]) > 0:
            avg_rank = np.mean(mi_results[n_select])
            print(f"  MI({n_select}): avg_rank={avg_rank:.2f}")
    
    # =================
    # Experiment 4: Variance-based selection
    # =================
    print(f"\n[Feature Selection] Experiment 4: Variance-based selection...")
    variance_percentiles = [75, 85, 95]
    var_results = {p: [] for p in variance_percentiles}
    
    for percentile in variance_percentiles:
        print(f"  Testing variance selection (>{percentile}th percentile)...")
        
        train_var, var_indices = select_variance_threshold(
            train_traces, percentile=percentile
        )
        test_var = test_traces[:, var_indices]
        
        for byte_idx in tqdm(range(n_bytes_eval), leave=False):
            train_hw = train_labels[:, byte_idx]
            test_hw = test_labels[:, byte_idx]
            
            rank = train_and_evaluate_rf(
                train_var, train_hw, test_var, test_hw,
                plaintexts, key, byte_idx, seed
            )
            var_results[percentile].append(rank)
    
    for percentile in variance_percentiles:
        if len(var_results[percentile]) > 0:
            avg_rank = np.mean(var_results[percentile])
            print(f"  Variance({percentile}): avg_rank={avg_rank:.2f}")
    
    # =================
    # Plot results
    # =================
    print(f"\n[Feature Selection] Generating comparison plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: PCA comparison
    ax = axes[0]
    pca_comp_ordered = sorted([n for n in pca_components if len(pca_results[n]) > 0])
    pca_ranks_ordered = [np.mean(pca_results[n]) for n in pca_comp_ordered]
    
    ax.plot(
        pca_comp_ordered, pca_ranks_ordered,
        marker='o', color='steelblue', linewidth=2, markersize=8,
        label='PCA', zorder=3
    )
    ax.axhline(y=avg_rank_full, color='red', linestyle='--', linewidth=2, label='Full traces (baseline)')
    
    ax.set_xlabel("Number of PCA Components", fontsize=11)
    ax.set_ylabel("Average Key Rank", fontsize=11)
    ax.set_title("PCA Compression Impact", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0.5)
    
    # Plot 2: MI and variance comparison
    ax = axes[1]
    
    mi_counts_ordered = sorted([n for n in mi_sample_counts if len(mi_results[n]) > 0])
    mi_ranks_ordered = [np.mean(mi_results[n]) for n in mi_counts_ordered]
    
    var_percentiles_ordered = sorted([p for p in variance_percentiles if len(var_results[p]) > 0])
    var_ranks_ordered = [np.mean(var_results[p]) for p in var_percentiles_ordered]
    
    ax.plot(
        mi_counts_ordered, mi_ranks_ordered,
        marker='s', color='darkorange', linewidth=2, markersize=8,
        label='MI selection', zorder=3
    )
    ax.plot(
        [100 - p for p in var_percentiles_ordered], var_ranks_ordered,
        marker='^', color='limegreen', linewidth=2, markersize=8,
        label='Variance selection', zorder=3
    )
    ax.axhline(y=avg_rank_full, color='red', linestyle='--', linewidth=2, label='Full traces (baseline)')
    
    ax.set_xlabel("Number of Selected Samples (%)", fontsize=11)
    ax.set_ylabel("Average Key Rank", fontsize=11)
    ax.set_title("Sample Selection Impact", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0.5)
    
    plt.tight_layout()
    
    plot_file = output_dir / "feature_selection_comparison_stm32f4.png"
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")
    
    # Save numerical results
    results = {
        "method": "Random Forest with Feature Selection",
        "dataset": "STM32F4",
        "n_bytes_evaluated": n_bytes_eval,
        "baseline_full_traces": {
            "ranks": rankings_full,
            "avg_rank": float(avg_rank_full),
        },
        "pca_results": {str(n): [float(r) for r in pca_results[n]] for n in pca_components},
        "mi_results": {str(n): [float(r) for r in mi_results[n]] for n in mi_sample_counts},
        "variance_results": {str(p): [float(r) for r in var_results[p]] for p in variance_percentiles},
        "seed": seed,
    }
    
    results_file = output_dir / "feature_selection_results_stm32f4.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Feature Selection] Saved results: {results_file}")
    
    print("\n[Feature Selection] Summary:")
    print(f"  Baseline (full traces): avg_rank={avg_rank_full:.2f}")
    print(f"  Best PCA: avg_rank={min([np.mean(pca_results[n]) for n in pca_comp_ordered]) if pca_comp_ordered else 999:.2f}")
    print(f"  Best MI: avg_rank={min([np.mean(mi_results[n]) for n in mi_counts_ordered]) if mi_counts_ordered else 999:.2f}")
    print(f"  Best Variance: avg_rank={min([np.mean(var_results[p]) for p in var_percentiles_ordered]) if var_percentiles_ordered else 999:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature selection experiments on STM32F4"
    )
    parser.add_argument(
        "--hdf5",
        type=str,
        default="traces.hdf5",
        help="Path to traces.hdf5",
    )
    parser.add_argument(
        "--n-bytes",
        type=int,
        default=4,
        help="Number of bytes to evaluate (for speed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/stm32f4_ml",
        help="Output directory",
    )
    
    args = parser.parse_args()
    main(
        hdf5_path=args.hdf5,
        n_bytes=args.n_bytes,
        seed=args.seed,
        output_dir=args.output_dir,
    )
