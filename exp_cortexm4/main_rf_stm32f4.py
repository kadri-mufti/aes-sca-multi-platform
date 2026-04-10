"""
Random Forest side-channel attack on STM32F4 (Cortex-M4) dataset.

This script trains Random Forest classifiers for key recovery using first-round
AES S-box Hamming Weight leakage from the STM32F4 traces.hdf5 dataset.

Evaluation metrics:
  - Key rank (position of correct key in ranking by probability)
  - Classification accuracy on test set
  - Trace efficiency (convergence vs. number of training traces)
  
Output:
  - Convergence plot: key rank vs. number of training traces
  - JSON results: model hyperparameters, evaluation metrics
"""

import os
import numpy as np
import argparse
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import local data utilities
from ml_data_utils_cortexm4 import load_stm32f4_dataset, align_class_probabilities


def compute_key_rank(predictions_proba, labels, true_key_byte):
    """
    Compute key rank from model predictions.
    
    Args:
        predictions_proba: (N_test, 256) — class probabilities from model
        labels: (N_test,) — true HW labels for metric computation
        true_key_byte: uint8 — ground truth key byte
        
    Returns:
        int — rank of true key (1 = ranked #1)
    """
    # Score each key hypothesis by sum of log-probabilities
    scores = np.sum(np.log(predictions_proba + 1e-40), axis=0)
    
    # Rank: how many keys scored better or equal?
    rank = np.sum(scores >= scores[true_key_byte])
    return int(rank)


def train_rf_byte(
    train_traces,
    train_labels,
    test_traces,
    test_labels,
    byte_idx,
    n_estimators=100,
    max_depth=30,
    min_samples_split=2,
    seed=42,
):
    """
    Train a Random Forest classifier for a single key byte.
    
    Args:
        train_traces: (N_train, N_samples) float array
        train_labels: (N_train,) int array — HW values [0, 8]
        test_traces: (N_test, N_samples) float array
        test_labels: (N_test,) int array — HW values [0, 8]
        byte_idx: int — byte index for logging
        n_estimators: int — number of trees
        max_depth: int — max tree depth
        min_samples_split: int — min samples to split
        seed: int — random seed
        
    Returns:
        dict with keys:
            - model: fitted RandomForestClassifier
            - accuracy: test set accuracy
            - conf_matrix: confusion matrix
            - pred_proba: (N_test, 256) — predicted probabilities
    """
    # Train RF on (traces, HW labels) to predict HW class
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    
    model.fit(train_traces, train_labels)
    
    # Predict on test set
    predictions = model.predict(test_traces)
    accuracy = np.mean(predictions == test_labels)
    
    # Confusion matrix (only for HW values, not all 256 classes)
    cm = confusion_matrix(test_labels, predictions, labels=np.arange(9))
    
    # For key ranking, we need class probabilities over all 256 possible bytes
    # Hack: since the model only predicts HW [0..8], we'll use a dummy probability
    # distribution that maps HW predictions to key candidates.
    # For a proper implementation, we'd need to train a classifier that directly
    # predicts key bytes (256 classes) using HW as labels.
    # For now, we'll create a simplified key-scoring using predicted HW probabilities.
    
    # Get RF predictions for each sample (soft predictions)
    hw_proba = align_class_probabilities(
        model, model.predict_proba(test_traces), n_classes=9
    )
    
    # Create 256-class probabilities by replicating HW probabilities
    # This is a simplification: each key candidate gets a score based on predicted HW
    # In reality, we'd need to train on (trace -> 256-class key) directly
    # For now, use the HW-based scoring as a proxy.
    
    # Workaround: distribute HW probabilities across key candidates that produce that HW
    hw_to_keys = {}  # Map HW -> list of all 256 keys that produce that HW
    from ml_data_utils_cortexm4 import hamming_weight, AES
    for k in range(256):
        # Compute what HW this key + test plaintexts would produce
        # (Use known plaintexts from the dataset)
        pass  # Will be filled below with actual plaintexts
    
    # Simpler approach: just return HW probabilities and do key ranking separately
    # We'll convert this to key ranking in the main function using ciphertexts
    
    return {
        "model": model,
        "accuracy": accuracy,
        "conf_matrix": cm,
        "hw_proba": hw_proba,
    }


def compute_key_rank_from_hw_proba(
    hw_proba,
    plaintexts,
    key,
    byte_idx,
):
    """
    Compute key rank from HW probabilities and known plaintexts.
    
    Args:
        hw_proba: (N_test, 9) — predicted HW probabilities [0..8]
        plaintexts: (N_test, 16) uint8 — plaintext bytes
        key: (16,) uint8 — ground truth key
        byte_idx: int — which key byte to rank (0-15)
        
    Returns:
        int — key rank
    """
    from ml_data_utils_cortexm4 import hamming_weight, AES
    
    n_test = hw_proba.shape[0]
    true_key_byte = key[byte_idx]
    
    # Score each key hypothesis
    scores = np.zeros(256)
    pt_byte = plaintexts[:, byte_idx]
    
    for k in range(256):
        # For this key hypothesis k, compute expected HW for all test plaintexts
        pt_xor_k = pt_byte ^ k
        sbox_out = AES.SBOX[pt_xor_k].astype(np.uint8)
        expected_hw = hamming_weight(sbox_out)  # (N_test,)
        
        # Score: sum of log P(predicted_hw) for each trace
        # hw_proba[:, i] = P(HW=i | trace)
        for i in range(n_test):
            hw = int(expected_hw[i])
            scores[k] += np.log(hw_proba[i, hw] + 1e-40)
    
    # Key rank
    rank = np.sum(scores >= scores[true_key_byte])
    return int(rank)


def main(
    hdf5_path="traces.hdf5",
    n_estimators=100,
    max_depth=30,
    seed=42,
    output_dir="results/stm32f4_ml",
):
    """
    Main function: train Random Forest classifiers and evaluate.
    
    Args:
        hdf5_path: Path to traces.hdf5
        n_estimators: Number of RF trees per byte
        max_depth: Max tree depth
        seed: Random seed
        output_dir: Output directory for results
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[RF-STM32F4] Loading dataset from {hdf5_path}...")
    
    try:
        # Load full dataset with all 16 bytes (for main evaluation)
        dataset = load_stm32f4_dataset(
            hdf5_path,
            train_split=0.8,
            seed=seed,
            target_byte=None,  # Load all 16 bytes
        )
    except FileNotFoundError:
        print(f"✗ Error: {hdf5_path} not found")
        return
    
    train_traces = dataset["train_traces"]
    train_labels = dataset["train_labels"]  # Shape: (N_train, 16)
    test_traces = dataset["test_traces"]
    test_labels = dataset["test_labels"]  # Shape: (N_test, 16)
    key = dataset["key"]
    plaintexts = dataset["plaintexts"]
    
    n_train = train_traces.shape[0]
    n_test = test_traces.shape[0]
    n_samples = train_traces.shape[1]
    
    print(f"  Train: {n_train} traces, Test: {n_test} traces")
    print(f"  Trace window: {n_samples} samples")
    print(f"  Training RF with {n_estimators} trees, max_depth={max_depth}")

    fast_mode = os.environ.get("ML_FAST", "0") == "1"
    byte_indices = list(range(16)) if not fast_mode else list(range(4))
    
    # Train RF for each byte
    print("\n[RF-STM32F4] Training classifiers for each key byte...")
    models_by_byte = {}
    key_ranks_all = {}  # key_ranks_all[byte_idx] = key rank for that byte
    
    for byte_idx in tqdm(byte_indices):
        train_hw = train_labels[:, byte_idx]
        test_hw = test_labels[:, byte_idx]
        
        result = train_rf_byte(
            train_traces,
            train_hw,
            test_traces,
            test_hw,
            byte_idx,
            n_estimators=n_estimators,
            max_depth=max_depth,
            seed=seed + byte_idx,
        )
        
        models_by_byte[byte_idx] = result
        
        # Compute key rank
        hw_proba = result["hw_proba"]
        key_rank = compute_key_rank_from_hw_proba(
            hw_proba, plaintexts, key, byte_idx
        )
        key_ranks_all[byte_idx] = key_rank
        
        accuracy = result["accuracy"]
        print(f"  Byte {byte_idx:2d}: accuracy={accuracy:.3f}, key_rank={key_rank}")
    
    convergence_data = {byte_idx: [] for byte_idx in byte_indices}
    if not fast_mode:
        # Convergence analysis: vary number of training traces
        print("\n[RF-STM32F4] Running convergence analysis...")
        trace_counts = [100, 250, 500, 1000, 2500, 5000, n_train]
        
        for n_traces in trace_counts:
            if n_traces > n_train:
                continue
            
            for byte_idx in tqdm(byte_indices, desc=f"n_traces={n_traces}"):
                train_hw_subset = train_labels[:n_traces, byte_idx]
                test_hw = test_labels[:, byte_idx]
                
                result = train_rf_byte(
                    train_traces[:n_traces],
                    train_hw_subset,
                    test_traces,
                    test_hw,
                    byte_idx,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    seed=seed + byte_idx,
                )
                
                hw_proba = result["hw_proba"]
                key_rank = compute_key_rank_from_hw_proba(
                    hw_proba, plaintexts, key, byte_idx
                )
                convergence_data[byte_idx].append(key_rank)
        
        # Plot convergence
        print("\n[RF-STM32F4] Generating convergence plot...")
        plt.figure(figsize=(10, 6))
        
        for byte_idx in byte_indices:
            ranks = convergence_data[byte_idx]
            plt.plot(
                trace_counts[: len(ranks)],
                ranks,
                marker='o',
                alpha=0.7,
                label=f"Byte {byte_idx}",
                linewidth=1.5,
            )
        
        plt.xlabel("Number of Training Traces", fontsize=12)
        plt.ylabel("Key Rank", fontsize=12)
        plt.title("Random Forest Convergence: STM32F4 Dataset", fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        
        conv_plot = output_dir / "rf_convergence_stm32f4.png"
        plt.savefig(conv_plot, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {conv_plot}")
    else:
        print("\n[RF-STM32F4] FAST mode enabled; skipping convergence analysis.")
    
    # Save results as JSON
    results = {
        "method": "Random Forest",
        "dataset": "STM32F4",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "n_training": n_train,
        "n_test": n_test,
        "trace_window_samples": n_samples,
        "key_ranks": {str(k): v for k, v in key_ranks_all.items()},
        "correctly_recovered": sum(1 for k, v in key_ranks_all.items() if v == 1),
        "evaluated_bytes": byte_indices,
        "seed": seed,
    }
    
    results_file = output_dir / "rf_results_stm32f4.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[RF-STM32F4] Saved results: {results_file}")
    
    # Print summary
    correct_count = results["correctly_recovered"]
    print(f"\n[RF-STM32F4] Summary:")
    print(f"  Correctly recovered: {correct_count}/16 bytes")
    print(f"  Average key rank: {np.mean(list(key_ranks_all.values())):.2f}")
    print(f"  Best key rank: {min(key_ranks_all.values())}")
    print(f"  Worst key rank: {max(key_ranks_all.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random Forest side-channel attack on STM32F4"
    )
    parser.add_argument(
        "--hdf5",
        type=str,
        default="traces.hdf5",
        help="Path to traces.hdf5",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of RF trees per byte",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=30,
        help="Max tree depth",
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
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
        output_dir=args.output_dir,
    )
