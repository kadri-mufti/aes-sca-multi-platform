"""
Support Vector Machine side-channel attack on STM32F4 (Cortex-M4) dataset.

This script trains SVM classifiers for key recovery using first-round
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
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import local data utilities
from ml_data_utils_cortexm4 import load_stm32f4_dataset, hamming_weight, AES, align_class_probabilities


def compute_key_rank_from_hw_scores(
    hw_scores,
    plaintexts,
    key,
    byte_idx,
):
    """
    Compute key rank from HW decision function scores and known plaintexts.
    
    Args:
        hw_scores: (N_test, 9) — aligned class probabilities for each HW class
        plaintexts: (N_test, 16) uint8 — plaintext bytes
        key: (16,) uint8 — ground truth key
        byte_idx: int — which key byte to rank (0-15)
        
    Returns:
        int — key rank
    """
    n_test = hw_scores.shape[0]
    true_key_byte = key[byte_idx]
    
    # Score each key hypothesis
    scores = np.zeros(256)
    pt_byte = plaintexts[:, byte_idx]
    
    for k in range(256):
        # For this key hypothesis k, compute expected HW for all test plaintexts
        pt_xor_k = pt_byte ^ k
        sbox_out = AES.SBOX[pt_xor_k].astype(np.uint8)
        expected_hw = hamming_weight(sbox_out)  # (N_test,)
        
        # Score: sum of decision function values for expected HW
        for i in range(n_test):
            hw = expected_hw[i]
            if 0 <= hw <= 8:
                scores[k] += np.log(hw_scores[i, hw] + 1e-40)
    
    # Key rank
    rank = np.sum(scores >= scores[true_key_byte])
    return int(rank)


def train_svm_byte(
    train_traces,
    train_labels,
    test_traces,
    test_labels,
    byte_idx,
    kernel='rbf',
    C=1.0,
    gamma='scale',
    seed=42,
):
    """
    Train an SVM classifier for a single key byte.
    
    Args:
        train_traces: (N_train, N_samples) float array
        train_labels: (N_train,) int array — HW values [0, 8]
        test_traces: (N_test, N_samples) float array
        test_labels: (N_test,) int array — HW values [0, 8]
        byte_idx: int — byte index for logging
        kernel: str — SVM kernel type
        C: float — regularization parameter
        gamma: str or float — kernel coefficient
        seed: int — random seed
        
    Returns:
        dict with keys:
            - model: fitted SVC
            - scaler: fitted StandardScaler
            - accuracy: test set accuracy
            - conf_matrix: confusion matrix
            - hw_scores: (N_test, 9) — decision function scores
    """
    # Standardize traces (important for SVM)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_traces)
    test_scaled = scaler.transform(test_traces)
    
    # Train SVM
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,
        decision_function_shape='ovr',
        random_state=seed,
        cache_size=200,
    )
    
    model.fit(train_scaled, train_labels)
    
    # Predict on test set
    predictions = model.predict(test_scaled)
    accuracy = np.mean(predictions == test_labels)
    
    # Confusion matrix (only for HW values 0-8)
    cm = confusion_matrix(test_labels, predictions, labels=np.arange(9))
    
    # Align probabilities to HW classes 0..8 so missing classes do not break ranking.
    hw_scores = align_class_probabilities(
        model, model.predict_proba(test_scaled), n_classes=9
    )
    
    return {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy,
        "conf_matrix": cm,
        "hw_scores": hw_scores,
    }


def main(
    hdf5_path="traces.hdf5",
    kernel='rbf',
    C=1.0,
    gamma='scale',
    seed=42,
    output_dir="results/stm32f4_ml",
):
    """
    Main function: train SVM classifiers and evaluate.
    
    Args:
        hdf5_path: Path to traces.hdf5
        kernel: SVM kernel type ('rbf' or 'linear')
        C: Regularization parameter
        gamma: Kernel coefficient
        seed: Random seed
        output_dir: Output directory for results
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[SVM-STM32F4] Loading dataset from {hdf5_path}...")
    
    try:
        # Load full dataset with all 16 bytes
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
    print(f"  SVM kernel: {kernel}, C={C}, gamma={gamma}")

    fast_mode = os.environ.get("ML_FAST", "0") == "1"
    max_fast_traces = 500
    byte_indices = list(range(16)) if not fast_mode else list(range(4))
    if fast_mode and n_train > max_fast_traces:
        train_traces = train_traces[:max_fast_traces]
        train_labels = train_labels[:max_fast_traces]
        n_train = max_fast_traces
        print(f"  FAST mode enabled; using first {max_fast_traces} training traces")
    
    # Train SVM for each byte
    print("\n[SVM-STM32F4] Training classifiers for each key byte...")
    models_by_byte = {}
    key_ranks_all = {}
    
    for byte_idx in tqdm(byte_indices):
        train_hw = train_labels[:, byte_idx]
        test_hw = test_labels[:, byte_idx]
        
        result = train_svm_byte(
            train_traces,
            train_hw,
            test_traces,
            test_hw,
            byte_idx,
            kernel=kernel,
            C=C,
            gamma=gamma,
            seed=seed + byte_idx,
        )
        
        models_by_byte[byte_idx] = result
        
        # Compute key rank
        hw_scores = result["hw_scores"]
        key_rank = compute_key_rank_from_hw_scores(
            hw_scores, plaintexts, key, byte_idx
        )
        key_ranks_all[byte_idx] = key_rank
        
        accuracy = result["accuracy"]
        print(f"  Byte {byte_idx:2d}: accuracy={accuracy:.3f}, key_rank={key_rank}")
    
    convergence_data = {byte_idx: [] for byte_idx in byte_indices}
    if not fast_mode:
        # Convergence analysis
        print("\n[SVM-STM32F4] Running convergence analysis...")
        trace_counts = [100, 250, 500, 1000, 2500, 5000, n_train]
        
        for n_traces in trace_counts:
            if n_traces > n_train:
                continue
            
            for byte_idx in tqdm(byte_indices, desc=f"n_traces={n_traces}"):
                train_hw_subset = train_labels[:n_traces, byte_idx]
                test_hw = test_labels[:, byte_idx]
                
                result = train_svm_byte(
                    train_traces[:n_traces],
                    train_hw_subset,
                    test_traces,
                    test_hw,
                    byte_idx,
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    seed=seed + byte_idx,
                )
                
                hw_scores = result["hw_scores"]
                key_rank = compute_key_rank_from_hw_scores(
                    hw_scores, plaintexts, key, byte_idx
                )
                convergence_data[byte_idx].append(key_rank)
        
        # Plot convergence
        print("\n[SVM-STM32F4] Generating convergence plot...")
        plt.figure(figsize=(10, 6))
        
        for byte_idx in byte_indices:
            ranks = convergence_data[byte_idx]
            plt.plot(
                trace_counts[: len(ranks)],
                ranks,
                marker='s',
                alpha=0.7,
                label=f"Byte {byte_idx}",
                linewidth=1.5,
            )
        
        plt.xlabel("Number of Training Traces", fontsize=12)
        plt.ylabel("Key Rank", fontsize=12)
        plt.title(f"SVM Convergence (kernel={kernel}): STM32F4 Dataset", fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        
        conv_plot = output_dir / "svm_convergence_stm32f4.png"
        plt.savefig(conv_plot, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {conv_plot}")
    else:
        print("\n[SVM-STM32F4] FAST mode enabled; skipping convergence analysis.")
    
    # Save results as JSON
    results = {
        "method": "Support Vector Machine",
        "kernel": kernel,
        "dataset": "STM32F4",
        "C": C,
        "gamma": gamma,
        "n_training": n_train,
        "n_test": n_test,
        "trace_window_samples": n_samples,
        "key_ranks": {str(k): v for k, v in key_ranks_all.items()},
        "correctly_recovered": sum(1 for k, v in key_ranks_all.items() if v == 1),
        "evaluated_bytes": byte_indices,
        "seed": seed,
    }
    
    results_file = output_dir / "svm_results_stm32f4.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SVM-STM32F4] Saved results: {results_file}")
    
    # Print summary
    correct_count = results["correctly_recovered"]
    print(f"\n[SVM-STM32F4] Summary:")
    print(f"  Correctly recovered: {correct_count}/16 bytes")
    print(f"  Average key rank: {np.mean(list(key_ranks_all.values())):.2f}")
    print(f"  Best key rank: {min(key_ranks_all.values())}")
    print(f"  Worst key rank: {max(key_ranks_all.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SVM side-channel attack on STM32F4"
    )
    parser.add_argument(
        "--hdf5",
        type=str,
        default="traces.hdf5",
        help="Path to traces.hdf5",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear"],
        help="SVM kernel type",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="SVM regularization parameter",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="SVM gamma parameter",
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
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        seed=args.seed,
        output_dir=args.output_dir,
    )
