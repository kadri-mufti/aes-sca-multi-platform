"""
Data loading and preprocessing utilities for STM32F4 (Cortex-M4) side-channel dataset.

This module provides utilities to load traces from traces.hdf5 and compute
Hamming Weight (HW) labels for first-round AES S-box leakage modeling.
Leakage model: HW(SBOX[plaintext XOR key])
"""

import numpy as np
import h5py
from aes import AES


def hamming_weight(x):
    """
    Compute the Hamming weight (number of 1-bits) for each element in x.
    
    Args:
        x: Integer array of any shape
        
    Returns:
        Integer array of same shape, containing Hamming weights
    """
    weight = np.zeros(x.shape, dtype=int)
    for i in range(8):
        weight += (x >> i) & 1
    return weight


def load_traces_hdf5(filepath, first_round_start=2000, first_round_stop=4000):
    """
    Load traces and metadata from HDF5 file.
    
    Args:
        filepath: Path to traces.hdf5
        first_round_start: Start index of first-round trace window
        first_round_stop: End index of first-round trace window
        
    Returns:
        tuple: (traces, plaintexts, key)
            - traces: (N_traces, N_samples) float array — power trace window
            - plaintexts: (N_traces, 16) uint8 array — plaintext bytes
            - key: (16,) uint8 array — ground truth key bytes
    """
    with h5py.File(filepath, 'r') as f:
        # Defensive key access
        if "keys" not in f or "plaintexts" not in f or "power" not in f:
            raise KeyError(
                f"HDF5 file missing required keys. Found: {list(f.keys())}"
            )
        
        key = f["keys"][0, :]
        plaintexts = f["plaintexts"][0, :, :]
        traces = f["power"][0, :, first_round_start:first_round_stop].astype(float)
    
    return traces, plaintexts, key


def compute_hw_labels(plaintexts, key, target_byte=None):
    """
    Compute Hamming Weight labels for first-round S-box leakage.
    
    Leakage model: HW(SBOX[plaintext_byte XOR key_byte])
    
    Args:
        plaintexts: (N_traces, 16) uint8 array
        key: (16,) uint8 array or None (will use first row of plaintexts)
        target_byte: int in [0, 15] or None (if None, compute for all bytes)
        
    Returns:
        If target_byte is None:
            labels: (N_traces, 16) int array — HW for each byte
        If target_byte is specified:
            labels: (N_traces,) int array — HW for that byte only
    """
    if key is None:
        raise ValueError("key cannot be None for HW label computation")
    
    if target_byte is not None:
        if not (0 <= target_byte <= 15):
            raise ValueError(f"target_byte must be in [0, 15], got {target_byte}")
        
        pt_xor_k = plaintexts[:, target_byte] ^ key[target_byte]
        sbox_out = AES.SBOX[pt_xor_k].astype(np.uint8)
        labels = hamming_weight(sbox_out)
        return labels
    else:
        # Compute for all 16 bytes
        labels = np.zeros((plaintexts.shape[0], 16), dtype=int)
        for byte_idx in range(16):
            pt_xor_k = plaintexts[:, byte_idx] ^ key[byte_idx]
            sbox_out = AES.SBOX[pt_xor_k].astype(np.uint8)
            labels[:, byte_idx] = hamming_weight(sbox_out)
        return labels


def align_class_probabilities(model, probabilities, n_classes=9):
    """
    Pad class probabilities to a fixed class range when some classes are missing.

    RandomForestClassifier and SVC only return probabilities for classes seen
    during training. For trace subsets, some HW classes can be absent, which
    would otherwise make indexing by HW label fail during key ranking.

    Args:
        model: fitted sklearn estimator with a ``classes_`` attribute
        probabilities: (N, C_present) probability matrix
        n_classes: total number of expected classes

    Returns:
        (N, n_classes) array aligned to class labels 0..n_classes-1
    """
    aligned = np.full((probabilities.shape[0], n_classes), 1e-40, dtype=float)
    for column_idx, class_label in enumerate(model.classes_):
        class_label = int(class_label)
        if 0 <= class_label < n_classes:
            aligned[:, class_label] = probabilities[:, column_idx]
    return aligned


def load_stm32f4_dataset(
    filepath,
    first_round_start=2000,
    first_round_stop=4000,
    train_split=0.8,
    seed=42,
    target_byte=None,
):
    """
    Load and split the STM32F4 dataset for ML experiments.
    
    Args:
        filepath: Path to traces.hdf5
        first_round_start: Start index of first-round window
        first_round_stop: End index of first-round window
        train_split: Fraction [0, 1] for train/test split
        seed: Random seed for reproducibility
        target_byte: Specific byte to extract (0-15) or None (all bytes)
        
    Returns:
        dict with keys:
            - trains_traces, train_labels: Training set
            - test_traces, test_labels: Test set
            - key: Ground truth key (for reference)
            - plaintexts: All plaintexts (for reference)
            - train_indices, test_indices: Split indices
    """
    # Load raw data
    traces, plaintexts, key = load_traces_hdf5(
        filepath, first_round_start, first_round_stop
    )
    
    n_traces = traces.shape[0]
    
    # Set random seed for reproducibility
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_traces)
    n_train = int(train_split * n_traces)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split traces
    train_traces = traces[train_indices]
    test_traces = traces[test_indices]
    
    # Compute labels
    if target_byte is not None:
        # Single byte classification (256-class problem)
        labels = compute_hw_labels(plaintexts, key, target_byte)
    else:
        # Multi-byte (16 separate 256-class problems, but structure compatible)
        labels = compute_hw_labels(plaintexts, key, target_byte=None)
    
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    
    return {
        "train_traces": train_traces,
        "train_labels": train_labels,
        "test_traces": test_traces,
        "test_labels": test_labels,
        "key": key,
        "plaintexts": plaintexts,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }


if __name__ == "__main__":
    # Simple self-test
    print("Testing ml_data_utils_cortexm4.py...")
    try:
        dataset = load_stm32f4_dataset("traces.hdf5", target_byte=0, train_split=0.8)
        print(f"✓ Loaded STM32F4 dataset")
        print(f"  - Train traces: {dataset['train_traces'].shape}")
        print(f"  - Train labels: {dataset['train_labels'].shape}")
        print(f"  - Test traces: {dataset['test_traces'].shape}")
        print(f"  - Test labels: {dataset['test_labels'].shape}")
        print(f"  - Key (first 4 bytes): {dataset['key'][:4]}")
    except FileNotFoundError:
        print("⚠ traces.hdf5 not found; skipping data load test")
    except Exception as e:
        print(f"✗ Error: {e}")
