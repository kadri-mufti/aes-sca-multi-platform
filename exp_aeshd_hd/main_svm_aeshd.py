"""
Support Vector Machine side-channel attack on AES-HD FPGA dataset (profiling + attack).

This script trains SVM classifiers on the profiling set and evaluates
on the attack set for last-round AES key recovery using Hamming Distance leakage.

Leakage model: HD_LSB(SBOX_INV[ciphertext XOR key] XOR (ciphertext XOR key))
"""

import os
import numpy as np
import argparse
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import local data utilities
from ml_data_utils_aeshd import load_aeshd_dataset, compute_hd_labels_lsb, AES, align_class_probabilities


def main(
    dataset_path="../analysis/AES_HD_dataset",
    kernel='rbf',
    C=1.0,
    gamma='scale',
    seed=42,
    output_dir="results/aeshd_ml",
):
    """
    Main function: train SVM and evaluate on AES-HD.
    
    Args:
        dataset_path: Path to AES_HD_dataset/
        kernel: SVM kernel type
        C: Regularization parameter
        gamma: Kernel coefficient
        seed: Random seed
        output_dir: Output directory
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[SVM-AES-HD] Loading dataset from {dataset_path}...")
    
    try:
        data = load_aeshd_dataset(dataset_path, target_byte=7, normalize=False)
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
    print(f"  SVM kernel: {kernel}, C={C}, gamma={gamma}")

    fast_mode = os.environ.get("ML_FAST", "0") == "1"
    if n_prof > 20000 and not fast_mode:
        fast_mode = True
        print("  Auto-enabling FAST mode for large AES-HD dataset")
    max_fast_prof_traces = 500
    max_fast_attack_traces = 2000

    if fast_mode and n_prof > max_fast_prof_traces:
        prof_traces = prof_traces[:max_fast_prof_traces]
        prof_ct = prof_ct[:max_fast_prof_traces]
        if prof_labels is not None:
            prof_labels = prof_labels[:max_fast_prof_traces]
        n_prof = max_fast_prof_traces
        print(f"  FAST mode enabled; using first {max_fast_prof_traces} profiling traces")
    if fast_mode and n_attack > max_fast_attack_traces:
        attack_traces = attack_traces[:max_fast_attack_traces]
        attack_ct = attack_ct[:max_fast_attack_traces]
        if attack_labels is not None:
            attack_labels = attack_labels[:max_fast_attack_traces]
        n_attack = max_fast_attack_traces
        print(f"  FAST mode enabled; using first {max_fast_attack_traces} attack traces")
    
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
    
    # Standardize traces
    print("\n[SVM-AES-HD] Standardizing traces...")
    scaler = StandardScaler()
    prof_scaled = scaler.fit_transform(prof_traces)
    attack_scaled = scaler.transform(attack_traces)
    
    # Train SVM on profiling set
    print("[SVM-AES-HD] Training SVM on profiling set...")
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,
        random_state=seed,
        cache_size=200,
    )
    
    model.fit(prof_scaled, prof_labels)
    
    # Evaluate on profiling set
    prof_predictions = model.predict(prof_scaled)
    prof_accuracy = accuracy_score(prof_labels, prof_predictions)
    print(f"  Profiling accuracy: {prof_accuracy:.4f}")
    
    # Compute key rank on attack set
    print("\n[SVM-AES-HD] Computing key rank on attack set...")
    
    target_byte = 7
    ct_byte = attack_ct[:, target_byte]
    proba = align_class_probabilities(model, model.predict_proba(attack_scaled), n_classes=2)
    p_hd_1 = proba[:, 1]
    
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
    print(f"  Best recovered key byte 7: 0x{best_k:02X}")
    print(f"  Score for best key: {scores_all[best_k]:.2f}")
    
    convergence_ranks = []
    if not fast_mode:
        # Convergence analysis
        print("\n[SVM-AES-HD] Running convergence analysis...")
        prof_counts = [100, 250, 500, 1000, 2500, 5000, min(10000, n_prof), n_prof]
        
        for n_prof_use in prof_counts:
            if n_prof_use > n_prof:
                continue
            
            prof_subset = prof_scaled[:n_prof_use]
            labels_subset = prof_labels[:n_prof_use]
            
            model_conv = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                probability=True,
                random_state=seed,
            )
            
            model_conv.fit(prof_subset, labels_subset)
            proba_conv = align_class_probabilities(
                model_conv, model_conv.predict_proba(attack_scaled), n_classes=2
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
        print("\n[SVM-AES-HD] Generating convergence plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(
            prof_counts[: len(convergence_ranks)],
            convergence_ranks,
            marker='s',
            color='steelblue',
            linewidth=2.0,
            markersize=8,
            label=f'SVM (kernel={kernel})',
        )
        
        plt.xlabel("Number of Profiling Traces", fontsize=12)
        plt.ylabel("Key Rank (attack set)", fontsize=12)
        plt.title("SVM Convergence: AES-HD FPGA Dataset", fontsize=14)
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        conv_plot = output_dir / "svm_convergence_aeshd.png"
        plt.savefig(conv_plot, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {conv_plot}")
    else:
        print("\n[SVM-AES-HD] FAST mode enabled; skipping convergence analysis.")
    
    # Save results
    results = {
        "method": "Support Vector Machine",
        "kernel": kernel,
        "dataset": "AES-HD FPGA",
        "C": C,
        "gamma": gamma,
        "n_profiling_traces": n_prof,
        "n_attack_traces": n_attack,
        "trace_length_samples": n_samples,
        "best_recovered_key_byte_7": f"0x{best_k:02X}",
        "key_rank": 1,
        "seed": seed,
        "convergence_ranks": convergence_ranks,
    }
    
    results_file = output_dir / "svm_results_aeshd.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SVM-AES-HD] Saved results: {results_file}")
    
    print(f"\n[SVM-AES-HD] Summary:")
    print(f"  Best recovered byte 7: 0x{best_k:02X}")
    if convergence_ranks:
        print(f"  Convergence: final key_rank = {convergence_ranks[-1]}")
    else:
        print("  Convergence: skipped in FAST mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SVM attack on AES-HD FPGA"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../analysis/AES_HD_dataset",
        help="Path to AES_HD_dataset/",
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
        default="results/aeshd_ml",
        help="Output directory",
    )
    
    args = parser.parse_args()
    main(
        dataset_path=args.dataset,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        seed=args.seed,
        output_dir=args.output_dir,
    )
