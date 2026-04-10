"""
1D Convolutional Neural Network side-channel attack on STM32F4 (Cortex-M4) dataset.

This script trains a 1D CNN for key recovery using first-round AES S-box
Hamming Weight leakage from the STM32F4 traces.hdf5 dataset.

CNN Architecture:
  - Input: (batch, trace_window, 1)
  - Conv1D layers with ReLU activation
  - GlobalAveragePooling or Flatten
  - Dense layers culminating in 9-class output (HW = 0..8)
  
Evaluation metrics:
  - Key rank (position of correct key in ranking by probability)
  - Classification accuracy on test set
  - Trace efficiency (convergence vs. number of training traces)
  
Output:
  - Convergence plot: key rank vs. number of training traces
  - Training history (loss, accuracy)
  - JSON results: model hyperparameters, evaluation metrics
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

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
from ml_data_utils_cortexm4 import load_stm32f4_dataset, hamming_weight, AES


def build_1dcnn_model(input_shape, num_classes=9):
    """
    Build a 1D CNN model for trace classification.
    
    Args:
        input_shape: (n_samples,) — length of trace window
        num_classes: int — number of output classes (9 for HW 0-8)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Input reshape (assumes input is 1D)
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        # Conv block 1
        Conv1D(32, kernel_size=11, strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        Conv1D(32, kernel_size=11, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Conv block 2
        Conv1D(64, kernel_size=11, strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        Conv1D(64, kernel_size=11, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Conv block 3
        Conv1D(128, kernel_size=11, strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        
        # Global pooling and dense
        GlobalAveragePooling1D(),
        
        # Fully connected layers
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Output layer (9 classes for HW)
        Dense(num_classes, activation='softmax'),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    return model


def compute_key_rank_from_cnn_proba(
    proba,
    plaintexts,
    key,
    byte_idx,
):
    """
    Compute key rank from CNN probabilities and known plaintexts.
    
    Args:
        proba: (N_test, 9) — predicted HW class probabilities [0..8]
        plaintexts: (N_test, 16) uint8 — plaintext bytes
        key: (16,) uint8 — ground truth key
        byte_idx: int — which key byte to rank (0-15)
        
    Returns:
        int — key rank
    """
    n_test = proba.shape[0]
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
        for i in range(n_test):
            hw = expected_hw[i]
            if 0 <= hw <= 8:
                scores[k] += np.log(proba[i, hw] + 1e-40)
    
    # Key rank
    rank = np.sum(scores >= scores[true_key_byte])
    return int(rank)


def main(
    hdf5_path="traces.hdf5",
    epochs=100,
    batch_size=32,
    seed=42,
    output_dir="results/stm32f4_ml",
):
    """
    Main function: train 1D CNN and evaluate.
    
    Args:
        hdf5_path: Path to traces.hdf5
        epochs: Max training epochs
        batch_size: Training batch size
        seed: Random seed
        output_dir: Output directory for results
    """
    
    # Set seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    fast_mode = os.environ.get("ML_FAST", "0") == "1"
    if epochs <= 5:
        fast_mode = True
    max_fast_traces = 2000
    byte_indices = list(range(16)) if not fast_mode else [0]
    if fast_mode:
        epochs = 1
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[CNN1D-STM32F4] Loading dataset from {hdf5_path}...")
    
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
    print(f"  Training with batch_size={batch_size}, epochs={epochs}")

    if fast_mode and n_train > max_fast_traces:
        train_traces = train_traces[:max_fast_traces]
        train_labels = train_labels[:max_fast_traces]
        n_train = max_fast_traces
        print(f"  FAST mode enabled; using first {max_fast_traces} training traces")
    
    # Normalize traces
    print("\n[CNN1D-STM32F4] Normalizing traces...")
    train_traces = (train_traces - np.mean(train_traces)) / (np.std(train_traces) + 1e-8)
    test_traces = (test_traces - np.mean(test_traces)) / (np.std(test_traces) + 1e-8)
    
    # Train CNN for each byte
    print("\n[CNN1D-STM32F4] Training CNN classifiers for each key byte...")
    models_by_byte = {}
    key_ranks_all = {}
    
    for byte_idx in tqdm(byte_indices):
        train_hw = train_labels[:, byte_idx]
        test_hw = test_labels[:, byte_idx]
        
        # Build and train model
        model = build_1dcnn_model(input_shape=n_samples, num_classes=9)
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        )
        
        history = model.fit(
            train_traces,
            train_hw,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(test_traces, test_hw, verbose=0)
        
        # Get predictions
        proba = model.predict(test_traces, verbose=0)  # (N_test, 9)
        
        # Compute key rank
        key_rank = compute_key_rank_from_cnn_proba(
            proba, plaintexts, key, byte_idx
        )
        key_ranks_all[byte_idx] = key_rank
        
        models_by_byte[byte_idx] = {
            'model': model,
            'accuracy': test_accuracy,
            'history': history,
        }
        
        print(f"  Byte {byte_idx:2d}: accuracy={test_accuracy:.3f}, key_rank={key_rank}")
    
    convergence_data = {byte_idx: [] for byte_idx in byte_indices}
    if fast_mode:
        print("\n[CNN1D-STM32F4] FAST mode enabled; skipping convergence analysis.")
    else:
        # Convergence analysis
        print("\n[CNN1D-STM32F4] Running convergence analysis...")
        trace_counts = [100, 250, 500, 1000, 2500, 5000, n_train]

        for n_traces in trace_counts:
            if n_traces > n_train:
                continue

            for byte_idx in tqdm(byte_indices, desc=f"n_traces={n_traces}"):
                train_hw_subset = train_labels[:n_traces, byte_idx]
                test_hw = test_labels[:, byte_idx]

                model = build_1dcnn_model(input_shape=n_samples, num_classes=9)

                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                )

                history = model.fit(
                    train_traces[:n_traces],
                    train_hw_subset,
                    validation_split=0.1,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    verbose=0,
                )

                proba = model.predict(test_traces, verbose=0)
                key_rank = compute_key_rank_from_cnn_proba(
                    proba, plaintexts, key, byte_idx
                )
                convergence_data[byte_idx].append(key_rank)

        # Plot convergence
        print("\n[CNN1D-STM32F4] Generating convergence plot...")
        plt.figure(figsize=(10, 6))

        for byte_idx in byte_indices:
            ranks = convergence_data[byte_idx]
            plt.plot(
                trace_counts[: len(ranks)],
                ranks,
                marker='^',
                alpha=0.7,
                label=f"Byte {byte_idx}",
                linewidth=1.5,
            )

        plt.xlabel("Number of Training Traces", fontsize=12)
        plt.ylabel("Key Rank", fontsize=12)
        plt.title("1D CNN Convergence: STM32F4 Dataset", fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()

        conv_plot = output_dir / "cnn1d_convergence_stm32f4.png"
        plt.savefig(conv_plot, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {conv_plot}")
    
    # Save results
    results = {
        "method": "1D Convolutional Neural Network",
        "dataset": "STM32F4",
        "epochs": epochs,
        "batch_size": batch_size,
        "n_training": n_train,
        "n_test": n_test,
        "trace_window_samples": n_samples,
        "key_ranks": {str(k): v for k, v in key_ranks_all.items()},
        "correctly_recovered": sum(1 for k, v in key_ranks_all.items() if v == 1),
        "evaluated_bytes": byte_indices,
        "seed": seed,
    }
    
    results_file = output_dir / "cnn1d_results_stm32f4.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[CNN1D-STM32F4] Saved results: {results_file}")
    
    # Print summary
    correct_count = results["correctly_recovered"]
    print(f"\n[CNN1D-STM32F4] Summary:")
    print(f"  Correctly recovered: {correct_count}/16 bytes")
    print(f"  Average key rank: {np.mean(list(key_ranks_all.values())):.2f}")
    print(f"  Best key rank: {min(key_ranks_all.values())}")
    print(f"  Worst key rank: {max(key_ranks_all.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1D CNN side-channel attack on STM32F4"
    )
    parser.add_argument(
        "--hdf5",
        type=str,
        default="traces.hdf5",
        help="Path to traces.hdf5",
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
        default="results/stm32f4_ml",
        help="Output directory",
    )
    
    args = parser.parse_args()
    main(
        hdf5_path=args.hdf5,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
