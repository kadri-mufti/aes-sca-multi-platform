"""
Data loading and preprocessing utilities for AES-HD FPGA side-channel dataset.

This module provides utilities to load traces from the AES_HD_dataset/ folder
and compute Hamming Distance (HD) labels for last-round AES key recovery.

Leakage model: HD(SBOX_INV[ciphertext XOR key] XOR (ciphertext XOR key))
  = number of bits that flip during SubBytes in last round (FPGA state transitions)
  = LSB of (SBOX_INV[ct^k] XOR (ct^k)) in our simplified 1-bit variant

Dataset structure:
  - profiling_traces_AES_HD.npy: (N_prof, 700) float32
  - profiling_ciphertext_AES_HD.npy: (N_prof, 16) uint8
  - profiling_labels_AES_HD.npy: (N_prof,) uint8
  - attack_traces_AES_HD.npy: (N_attack, 700) float32
  - attack_ciphertext_AES_HD.npy: (N_attack, 16) uint8
  - attack_labels_AES_HD.npy: (N_attack,) uint8 (optional)
"""

import numpy as np
from pathlib import Path
from aes import AES


def hamming_distance_lsb(a, b):
    """
    Compute LSB of Hamming distance: (a XOR b) & 0x1.
    
    For AES-HD, we model the leakage as 1-bit HD:
    HD_LSB(SBOX_INV[ct^k] XOR (ct^k))
    
    Args:
        a, b: Integer arrays or scalars
        
    Returns:
        Array of LSBs of (a XOR b)
    """
    return (a ^ b) & 0x1


def compute_hd_labels_lsb(ciphertext, key, target_byte=7):
    """
    Compute Hamming Distance (LSB variant) labels for last-round key byte.
    
    Leakage model: LSB(SBOX_INV[ct^k] XOR (ct^k))
    
    Args:
        ciphertext: (N_traces, 16) uint8 array
        key: scalar uint8 — the key byte to test (not the actual key, but a guess)
        target_byte: int in [0, 15] — which byte to target (typically 7 for AES-HD)
        
    Returns:
        labels: (N_traces,) uint8 array — 1-bit HD for each trace
    """
    if not (0 <= target_byte <= 15):
        raise ValueError(f"target_byte must be in [0, 15], got {target_byte}")
    
    ct_byte = ciphertext[:, target_byte]
    intermediate = ct_byte ^ key
    sbox_inv_val = AES.SBOX_INV[intermediate].astype(np.uint8)
    
    labels = hamming_distance_lsb(sbox_inv_val, intermediate)
    return labels.astype(np.uint8)


def load_aeshd_dataset(
    dataset_path="../analysis/AES_HD_dataset",
    target_byte=7,
    normalize=True,
):
    """
    Load the AES-HD profiling and attack datasets from numpy files.
    
    Args:
        dataset_path: Path to AES_HD_dataset/ folder
        target_byte: Byte to target (typically 7, the only varying byte)
        normalize: If True, standardize traces to mean=0, std=1
        
    Returns:
        dict with keys:
            - prof_traces: (N_prof, 700) float32 — profiling traces
            - prof_ciphertext: (N_prof, 16) uint8 — profiling ciphertexts
            - prof_labels: (N_prof,) uint8 — profiling labels (if file exists)
            - attack_traces: (N_attack, 700) float32 — attack traces
            - attack_ciphertext: (N_attack, 16) uint8 — attack ciphertexts
            - attack_labels: (N_attack,) uint8 — attack labels (if file exists)
            - n_profiling: int — number of profiling traces
            - n_attack: int — number of attack traces
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Load profiling set
    prof_traces_path = dataset_path / "profiling_traces_AES_HD.npy"
    prof_ct_path = dataset_path / "profiling_ciphertext_AES_HD.npy"
    prof_labels_path = dataset_path / "profiling_labels_AES_HD.npy"
    
    if not prof_traces_path.exists():
        raise FileNotFoundError(f"Missing {prof_traces_path}")
    
    prof_traces = np.load(prof_traces_path).astype(np.float32)
    prof_ciphertext = np.load(prof_ct_path).astype(np.uint8)
    prof_labels = np.load(prof_labels_path).astype(np.uint8) if prof_labels_path.exists() else None
    
    # Load attack set
    attack_traces_path = dataset_path / "attack_traces_AES_HD.npy"
    attack_ct_path = dataset_path / "attack_ciphertext_AES_HD.npy"
    attack_labels_path = dataset_path / "attack_labels_AES_HD.npy"
    
    if not attack_traces_path.exists():
        raise FileNotFoundError(f"Missing {attack_traces_path}")
    
    attack_traces = np.load(attack_traces_path).astype(np.float32)
    attack_ciphertext = np.load(attack_ct_path).astype(np.uint8)
    attack_labels = np.load(attack_labels_path).astype(np.uint8) if attack_labels_path.exists() else None
    
    # Normalize if requested
    if normalize:
        prof_traces = (prof_traces - np.mean(prof_traces)) / (np.std(prof_traces) + 1e-8)
        attack_traces = (attack_traces - np.mean(attack_traces)) / (np.std(attack_traces) + 1e-8)
    
    return {
        "prof_traces": prof_traces,
        "prof_ciphertext": prof_ciphertext,
        "prof_labels": prof_labels,  # May be None if file doesn't exist
        "attack_traces": attack_traces,
        "attack_ciphertext": attack_ciphertext,
        "attack_labels": attack_labels,  # May be None if file doesn't exist
        "n_profiling": prof_traces.shape[0],
        "n_attack": attack_traces.shape[0],
        "target_byte": target_byte,
    }


def compute_key_rank(
    attack_traces,
    attack_ciphertext,
    model_predict_proba,
    true_key_byte,
    target_byte=7,
):
    """
    Compute key rank (position of true key in ranking by log-likelihood).
    
    Args:
        attack_traces: (N_attack, N_samples) array
        attack_ciphertext: (N_attack, 16) uint8 array
        model_predict_proba: function that returns (N_attack, 256) probabilities
            for each of 256 possible key bytes
        true_key_byte: uint8 — the ground truth key byte
        target_byte: int — which ciphertext byte to use
        
    Returns:
        int — rank of true key (1 = correct key ranked #1)
    """
    # Get model predictions (shape: N_attack x 256)
    probs = model_predict_proba(attack_traces)  # (N_attack, 256)
    
    if probs.shape[1] != 256:
        raise ValueError(f"Expected 256 classes, got {probs.shape[1]}")
    
    # Compute log-likelihood for each key hypothesis
    scores = np.zeros(256)
    ct_byte = attack_ciphertext[:, target_byte]
    
    for k in range(256):
        # Compute HD-LSB labels for this key hypothesis
        intermediate = ct_byte ^ k
        sbox_inv_val = AES.SBOX_INV[intermediate].astype(np.uint8)
        hd_lsb = (sbox_inv_val ^ intermediate) & 0x1
        
        # Log-likelihood: sum log(P(hd_lsb | trace))
        # If hd_lsb==1, use P(class=1); if hd_lsb==0, use P(class=0)
        for i in range(len(attack_traces)):
            label = hd_lsb[i]
            # Assume probs[:, k] corresponds to class probabilities for key k
            # We need a mapping: class 0, 1 from k
            # For simplicity: class label = hd_lsb, and we use probs directly
            # This assumes model outputs: probs[:, k] = P(class=1 | trace, key=k)
            # Score for class=hd_lsb:
            p = probs[i, k]
            if label == 1:
                scores[k] += np.log(p + 1e-40)
            else:
                scores[k] += np.log(1 - p + 1e-40)
    
    # Rank: higher score = better
    rank = np.sum(scores >= scores[true_key_byte]) 
    return int(rank)


def align_class_probabilities(model, probabilities, n_classes=2):
    """
    Pad class probabilities to a fixed class range when a subset contains only
    one of the two binary HD classes.

    Args:
        model: fitted sklearn estimator with a ``classes_`` attribute
        probabilities: (N, C_present) probability matrix
        n_classes: expected class count, default 2 for HD labels

    Returns:
        (N, n_classes) array aligned to class labels 0..n_classes-1
    """
    aligned = np.full((probabilities.shape[0], n_classes), 1e-40, dtype=float)
    for column_idx, class_label in enumerate(model.classes_):
        class_label = int(class_label)
        if 0 <= class_label < n_classes:
            aligned[:, class_label] = probabilities[:, column_idx]
    return aligned


if __name__ == "__main__":
    # Simple self-test
    print("Testing ml_data_utils_aeshd.py...")
    try:
        data = load_aeshd_dataset()
        print(f"✓ Loaded AES-HD dataset")
        print(f"  - Profiling traces: {data['prof_traces'].shape}")
        print(f"  - Profiling ciphertext: {data['prof_ciphertext'].shape}")
        print(f"  - Attack traces: {data['attack_traces'].shape}")
        print(f"  - Attack ciphertext: {data['attack_ciphertext'].shape}")
        if data['prof_labels'] is not None:
            print(f"  - Profiling labels: {data['prof_labels'].shape}")
        if data['attack_labels'] is not None:
            print(f"  - Attack labels: {data['attack_labels'].shape}")
    except FileNotFoundError as e:
        print(f"⚠ Dataset not found: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
