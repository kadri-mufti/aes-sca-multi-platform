"""
1D Convolutional Neural Network side-channel attack on AES-HD FPGA (profiling + attack).

This script trains a 1D CNN on the profiling set and evaluates on the attack set
for last-round AES key recovery using Hamming Distance leakage.

CNN Architecture:
  - Input: (batch, trace, 1)
  - Conv1D layers with ReLU, BatchNorm, MaxPooling
  - Global average pooling
  - Dense layers with Dropout
  - Output: 2-class (HD=0 or HD=1)
  
Evaluation: Key rank convergence vs. profiling traces used
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import argparse
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import local data utilities
from ml_data_utils_aeshd import load_aeshd_dataset, compute_hd_labels_lsb, AES


def build_1dcnn_model(input_shape, num_classes=2):
    """
    Build a 1D CNN model for profiling attacks on weak signals.
    
    Args:
        input_shape: (n_samples,) — trace length
        num_classes: int — number of classes (2 for HD 0/1)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Reshape input
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        # Conv block 1: deeper for weak signals
        Conv1D(64, kernel_size=11, strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        Conv1D(64, kernel_size=11, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Conv block 2
        Conv1D(128, kernel_size=11, strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        Conv1D(128, kernel_size=11, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Conv block 3
        Conv1D(256, kernel_size=11, strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        
        # Global pooling
        GlobalAveragePooling1D(),
        
        # Fully connected
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        
        # Output layer
        Dense(num_classes, activation='softmax'),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    return model


def main(
    dataset_path="../analysis/AES_HD_dataset",
    epochs=100,
    batch_size=32,
    seed=42,
    output_dir="results/aeshd_ml",
):
    """
    Main function: train 1D CNN on AES-HD and evaluate.
    
    Args:
        dataset_path: Path to AES_HD_dataset/
        epochs: Max training epochs
        batch_size: Training batch size
        seed: Random seed
        output_dir: Output directory
    """
    
    # Set seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)
    fast_mode = os.environ.get("ML_FAST", "0") == "1"
    if epochs <= 5:
        fast_mode = True
    max_fast_prof_traces = 500
    max_fast_attack_traces = 2000
    if fast_mode:
        epochs = 1
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[CNN1D-AES-HD] Loading dataset from {dataset_path}...")
    
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

    if n_prof > 20000 and not fast_mode:
        fast_mode = True
        print("  Auto-enabling FAST mode for large AES-HD dataset")
    
    print(f"  Profiling: {n_prof} traces")
    print(f"  Attack: {n_attack} traces")
    print(f"  Trace length: {n_samples} samples")
    print(f"  Training with batch_size={batch_size}, epochs={epochs}")

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
    
    # Normalize traces
    print("\n[CNN1D-AES-HD] Normalizing traces...")
    prof_traces = (prof_traces - np.mean(prof_traces)) / (np.std(prof_traces) + 1e-8)
    attack_traces = (attack_traces - np.mean(attack_traces)) / (np.std(attack_traces) + 1e-8)
    
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
    
    # Train 1D CNN on profiling set
    print("\n[CNN1D-AES-HD] Training 1D CNN on profiling set...")
    model = build_1dcnn_model(input_shape=n_samples, num_classes=2)
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
    )
    
    history = model.fit(
        prof_traces,
        prof_labels,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    
    # Evaluate
    prof_loss, prof_accuracy = model.evaluate(prof_traces, prof_labels, verbose=0)
    print(f"  Profiling accuracy: {prof_accuracy:.4f}")
    
    # Compute key rank on attack set
    print("\n[CNN1D-AES-HD] Computing key rank on attack set...")
    
    target_byte = 7
    ct_byte = attack_ct[:, target_byte]
    proba = model.predict(attack_traces, verbose=0)  # (N_attack, 2)
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
    if fast_mode:
        print("\n[CNN1D-AES-HD] FAST mode enabled; skipping convergence analysis.")
    else:
        # Convergence analysis
        print("\n[CNN1D-AES-HD] Running convergence analysis...")
        prof_counts = [100, 250, 500, 1000, 2500, 5000, min(10000, n_prof), n_prof]

        for n_prof_use in prof_counts:
            if n_prof_use > n_prof:
                continue

            prof_subset = prof_traces[:n_prof_use]
            labels_subset = prof_labels[:n_prof_use]

            model_conv = build_1dcnn_model(input_shape=n_samples, num_classes=2)

            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
            )

            history = model_conv.fit(
                prof_subset,
                labels_subset,
                validation_split=0.1 if n_prof_use > 100 else 0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0,
            )

            proba_conv = model_conv.predict(attack_traces, verbose=0)
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
        print("\n[CNN1D-AES-HD] Generating convergence plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(
            prof_counts[: len(convergence_ranks)],
            convergence_ranks,
            marker='d',
            color='limegreen',
            linewidth=2.0,
            markersize=8,
            label='1D CNN',
        )

        plt.xlabel("Number of Profiling Traces", fontsize=12)
        plt.ylabel("Key Rank (attack set)", fontsize=12)
        plt.title("1D CNN Convergence: AES-HD FPGA Dataset", fontsize=14)
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()

        conv_plot = output_dir / "cnn1d_convergence_aeshd.png"
        plt.savefig(conv_plot, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {conv_plot}")
    
    # Save results
    results = {
        "method": "1D Convolutional Neural Network",
        "dataset": "AES-HD FPGA",
        "epochs": epochs,
        "batch_size": batch_size,
        "n_profiling_traces": n_prof,
        "n_attack_traces": n_attack,
        "trace_length_samples": n_samples,
        "best_recovered_key_byte_7": f"0x{best_k:02X}",
        "key_rank": 1,
        "seed": seed,
        "convergence_ranks": convergence_ranks,
    }
    
    results_file = output_dir / "cnn1d_results_aeshd.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[CNN1D-AES-HD] Saved results: {results_file}")
    
    print(f"\n[CNN1D-AES-HD] Summary:")
    print(f"  Best recovered byte 7: 0x{best_k:02X}")
    if convergence_ranks:
        print(f"  Convergence: final key_rank = {convergence_ranks[-1]}")
    else:
        print("  Convergence: skipped in FAST mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1D CNN attack on AES-HD FPGA"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../analysis/AES_HD_dataset",
        help="Path to AES_HD_dataset/",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
