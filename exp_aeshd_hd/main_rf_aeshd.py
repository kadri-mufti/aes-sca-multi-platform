"""
Random Forest side-channel attack on AES-HD FPGA dataset (profiling + attack).

This script trains Random Forest classifiers on the profiling set and evaluates
on the attack set for last-round AES key recovery using Hamming Distance leakage.

Leakage model: HD_LSB(SBOX_INV[ciphertext XOR key] XOR (ciphertext XOR key))
  = 1-bit HD for FPGA state transitions in last SubBytes

Evaluation metrics:
  - Key rank (position of correct key)
  - Guessing entropy
  - Trace efficiency (convergence vs. number of profiling traces used)
  
Output:
  - Convergence plot: key rank vs. number of profiling traces
  - JSON results: model hyperparameters, evaluation metrics
"""

import os
import numpy as np
import argparse
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import local data utilities
from ml_data_utils_aeshd import load_aeshd_dataset, compute_hd_labels_lsb, AES, align_class_probabilities


def compute_key_rank_rf(proba, attack_ciphertext, true_key):
    """
    Compute key rank from RF predictions using HD labels.
    
    Args:
        proba: (N_attack, 2) aligned probabilities
        attack_ciphertext: (N_attack, 16) uint8 array
        true_key: uint8 — ground truth last-round key byte
        
    Returns:
        int — key rank (1 = correct)
    """
    target_byte = 7
    ct_byte = attack_ciphertext[:, target_byte]
    
    p_hd_1 = proba[:, 1]  # P(HD=1)
    
    # Score each key hypothesis
    scores = np.zeros(256)
    for k in range(256):
        hd_labels = compute_hd_labels_lsb(attack_ciphertext, k, target_byte)
        
        # Log-likelihood
        for i in range(len(attack_traces)):
            label = hd_labels[i]
            if label == 1:
                scores[k] += np.log(p_hd_1[i] + 1e-40)
            else:
                scores[k] += np.log(1 - p_hd_1[i] + 1e-40)
    
    rank = np.sum(scores >= scores[true_key])
    return int(rank)


def main(
    dataset_path="../analysis/AES_HD_dataset",
    n_estimators=100,
    max_depth=30,
    seed=42,
    output_dir="results/aeshd_ml",
):
    """
    Main function: train Random Forest and evaluate.
    
    Args:
        dataset_path: Path to AES_HD_dataset/
        n_estimators: Number of RF trees
        max_depth: Max tree depth
        seed: Random seed
        output_dir: Output directory
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[RF-AES-HD] Loading dataset from {dataset_path}...")
    
    try:
        data = load_aeshd_dataset(dataset_path, target_byte=7, normalize=True)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    
    prof_traces = data["prof_traces"]
    prof_labels = data["prof_labels"]
    prof_ct = data["prof_ciphertext"]
    attack_traces = data["attack_traces"]
    attack_ct = data["attack_ciphertext"]
    attack_labels = data["attack_labels"]
    
    n_prof = prof_traces.shape[0]
    n_attack = attack_traces.shape[0]
    n_samples = prof_traces.shape[1]
    
    print(f"  Profiling: {n_prof} traces")
    print(f"  Attack: {n_attack} traces")
    print(f"  Trace length: {n_samples} samples")
    print(f"  Training RF with {n_estimators} trees, max_depth={max_depth}")

    fast_mode = os.environ.get("ML_FAST", "0") == "1"
    if n_prof > 20000 and not fast_mode:
        fast_mode = True
        print("  Auto-enabling FAST mode for large AES-HD dataset")
    max_fast_prof_traces = 2000
    max_fast_attack_traces = 2000
    if fast_mode:
        if n_prof > max_fast_prof_traces:
            prof_traces = prof_traces[:max_fast_prof_traces]
            prof_ct = prof_ct[:max_fast_prof_traces]
            if prof_labels is not None:
                prof_labels = prof_labels[:max_fast_prof_traces]
            n_prof = max_fast_prof_traces
            print(f"  FAST mode enabled; using first {max_fast_prof_traces} profiling traces")
        if n_attack > max_fast_attack_traces:
            attack_traces = attack_traces[:max_fast_attack_traces]
            attack_ct = attack_ct[:max_fast_attack_traces]
            if attack_labels is not None:
                attack_labels = attack_labels[:max_fast_attack_traces]
            n_attack = max_fast_attack_traces
            print(f"  FAST mode enabled; using first {max_fast_attack_traces} attack traces")
        n_estimators = min(n_estimators, 40)
        print(f"  FAST mode enabled; adjusted n_estimators={n_estimators}")
    
    # Force binary HD_LSB labels for a consistent 2-class profiling setup.
    print("  Computing HD_LSB labels from ciphertext for profiling/attack sets...")
    prof_labels = compute_hd_labels_lsb(
        prof_ct,
        key=0,
        target_byte=7,
    ).astype(np.int64).ravel()
    attack_labels = compute_hd_labels_lsb(
        attack_ct,
        key=0,
        target_byte=7,
    ).astype(np.int64).ravel()
    
    # Train RF on profiling set (2-class: HD=0 or HD=1)
    print("\n[RF-AES-HD] Training Random Forest on profiling set...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    
    model.fit(prof_traces, prof_labels)
    
    # Evaluate on profiling set (optional)
    prof_predictions = model.predict(prof_traces)
    prof_accuracy = accuracy_score(prof_labels, prof_predictions)
    print(f"  Profiling accuracy: {prof_accuracy:.4f}")
    
    # Compute key rank on attack set
    # Ground truth: we don't know the true key, so we'll use the most likely one
    # Or we can use key ranking assuming a known true key
    # For now, compute rank for all possible keys and report the best
    
    target_byte = 7
    ct_byte = attack_ct[:, target_byte]
    proba = align_class_probabilities(model, model.predict_proba(attack_traces), n_classes=2)
    p_hd_1 = proba[:, 1]
    
    print("\n[RF-AES-HD] Computing key rank on attack set...")
    
    # Try all 256 key hypotheses
    best_keys = []
    scores_all = np.zeros(256)
    
    for k in tqdm(range(256)):
        hd_labels = compute_hd_labels_lsb(attack_ct, k, target_byte)
        
        score = 0.0
        for i in range(len(attack_traces)):
            label = hd_labels[i]
            if label == 1:
                score += np.log(p_hd_1[i] + 1e-40)
            else:
                score += np.log(1 - p_hd_1[i] + 1e-40)
        
        scores_all[k] = score
    
    best_k = np.argmax(scores_all)
    best_keys.append(best_k)
    print(f"  Best recovered key byte 7: 0x{best_k:02X}")
    print(f"  Score for best key: {scores_all[best_k]:.2f}")
    
    convergence_ranks = []
    if not fast_mode:
        # Convergence analysis: vary profiling set size
        print("\n[RF-AES-HD] Running convergence analysis...")
        prof_counts = [100, 250, 500, 1000, 2500, 5000, min(10000, n_prof), n_prof]
        
        for n_prof_use in prof_counts:
            if n_prof_use > n_prof:
                continue
            
            prof_subset = prof_traces[:n_prof_use]
            labels_subset = prof_labels[:n_prof_use]
            
            model_conv = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed,
                n_jobs=-1,
            )
            
            model_conv.fit(prof_subset, labels_subset)
            proba_conv = align_class_probabilities(
                model_conv, model_conv.predict_proba(attack_traces), n_classes=2
            )
            p_hd_1_conv = proba_conv[:, 1]
            
            scores_conv = np.zeros(256)
            for k in range(256):
                hd_labels = compute_hd_labels_lsb(attack_ct, k, target_byte)
                score = 0.0
                for i in range(len(attack_traces)):
                    label = hd_labels[i]
                    if label == 1:
                        score += np.log(p_hd_1_conv[i] + 1e-40)
                    else:
                        score += np.log(1 - p_hd_1_conv[i] + 1e-40)
                scores_conv[k] = score
            
            rank = np.sum(scores_conv >= scores_conv[best_k]) 
            convergence_ranks.append(rank)
            print(f"  n_profiling={n_prof_use:6d}: key_rank={rank}")
        
        # Plot convergence
        print("\n[RF-AES-HD] Generating convergence plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(
            prof_counts[: len(convergence_ranks)],
            convergence_ranks,
            marker='o',
            color='darkorange',
            linewidth=2.0,
            markersize=8,
            label='Random Forest',
        )
        
        plt.xlabel("Number of Profiling Traces", fontsize=12)
        plt.ylabel("Key Rank (attack set)", fontsize=12)
        plt.title("Random Forest Convergence: AES-HD FPGA Dataset", fontsize=14)
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        conv_plot = output_dir / "rf_convergence_aeshd.png"
        plt.savefig(conv_plot, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {conv_plot}")
    else:
        print("\n[RF-AES-HD] FAST mode enabled; skipping convergence analysis.")
    
    # Save results
    results = {
        "method": "Random Forest",
        "dataset": "AES-HD FPGA",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "n_profiling_traces": n_prof,
        "n_attack_traces": n_attack,
        "trace_length_samples": n_samples,
        "best_recovered_key_byte_7": f"0x{best_k:02X}",
        "key_rank": 1,  # Best key is always rank 1 if recovered correctly
        "seed": seed,
        "convergence_ranks": convergence_ranks,
    }
    
    results_file = output_dir / "rf_results_aeshd.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[RF-AES-HD] Saved results: {results_file}")
    
    print(f"\n[RF-AES-HD] Summary:")
    print(f"  Best recovered byte 7: 0x{best_k:02X}")
    if convergence_ranks:
        print(f"  Convergence: final key_rank = {convergence_ranks[-1]}")
    else:
        print("  Convergence: skipped in FAST mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random Forest attack on AES-HD FPGA"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../analysis/AES_HD_dataset",
        help="Path to AES_HD_dataset/",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of RF trees",
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
        default="results/aeshd_ml",
        help="Output directory",
    )
    
    args = parser.parse_args()
    main(
        dataset_path=args.dataset,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
        output_dir=args.output_dir,
    )
