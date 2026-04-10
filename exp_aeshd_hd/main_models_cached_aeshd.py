"""
Cached ML training/evaluation for AES-HD side-channel experiments.

This additive companion script trains RF/SVM/1D-CNN models and saves model
artifacts so subsequent runs can skip retraining.

Artifacts:
  - RF model: pickle
  - SVM model + scaler: pickle
  - CNN model: .keras
  - Metadata: JSON with model paths and metrics
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    MaxPooling1D,
    Reshape,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from ml_data_utils_aeshd import compute_hd_labels_lsb, load_aeshd_dataset, align_class_probabilities


def save_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_1dcnn_model(input_shape, num_classes=2):
    model = Sequential([
        Reshape((input_shape, 1), input_shape=(input_shape,)),
        Conv1D(64, 11, padding="same", activation="relu"),
        BatchNormalization(),
        Conv1D(64, 11, padding="same", activation="relu"),
        MaxPooling1D(2),
        Conv1D(128, 11, padding="same", activation="relu"),
        BatchNormalization(),
        Conv1D(128, 11, padding="same", activation="relu"),
        MaxPooling1D(2),
        Conv1D(256, 11, padding="same", activation="relu"),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def rank_key_from_binary_proba(p_hd_1, attack_ct, target_byte=7):
    scores = np.zeros(256, dtype=float)
    for k in range(256):
        labels = compute_hd_labels_lsb(attack_ct, k, target_byte)
        ll = 0.0
        for i in range(len(labels)):
            if int(labels[i]) == 1:
                ll += np.log(p_hd_1[i] + 1e-40)
            else:
                ll += np.log(1.0 - p_hd_1[i] + 1e-40)
        scores[k] = ll
    best_k = int(np.argmax(scores))
    rank = int(np.sum(scores >= scores[best_k]))
    return best_k, rank


def main():
    parser = argparse.ArgumentParser(description="Cached AES-HD ML models")
    parser.add_argument("--dataset", type=str, default="../analysis/AES_HD_dataset")
    parser.add_argument("--output-dir", type=str, default="../results/aeshd_ml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnn-epochs", type=int, default=25)
    parser.add_argument("--cnn-batch-size", type=int, default=32)
    parser.add_argument("--max-prof-traces", type=int, default=10000)
    parser.add_argument("--max-attack-traces", type=int, default=4000)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data = load_aeshd_dataset(args.dataset, target_byte=7, normalize=False)
    prof_x = data["prof_traces"]
    attack_x = data["attack_traces"]
    prof_ct = data["prof_ciphertext"]
    attack_ct = data["attack_ciphertext"]

    if args.max_prof_traces > 0 and prof_x.shape[0] > args.max_prof_traces:
        prof_x = prof_x[: args.max_prof_traces]
        prof_ct = prof_ct[: args.max_prof_traces]
    if args.max_attack_traces > 0 and attack_x.shape[0] > args.max_attack_traces:
        attack_x = attack_x[: args.max_attack_traces]
        attack_ct = attack_ct[: args.max_attack_traces]

    prof_y = compute_hd_labels_lsb(prof_ct, key=0, target_byte=7).astype(np.int64).ravel()

    # Keep full-size behavior for publication-style reruns, but allow users to trim externally.

    # RF
    rf_path = models_dir / "rf_aeshd.pkl"
    if rf_path.exists() and not args.force_retrain:
        rf = load_pickle(rf_path)
    else:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=35,
            random_state=args.seed,
            n_jobs=-1,
        )
        rf.fit(prof_x, prof_y)
        save_pickle(rf_path, rf)

    rf_proba = align_class_probabilities(rf, rf.predict_proba(attack_x), n_classes=2)
    rf_best, rf_rank = rank_key_from_binary_proba(rf_proba[:, 1], attack_ct, target_byte=7)

    # SVM
    svm_path = models_dir / "svm_aeshd.pkl"
    if svm_path.exists() and not args.force_retrain:
        payload = load_pickle(svm_path)
        scaler = payload["scaler"]
        svm = payload["model"]
    else:
        scaler = StandardScaler()
        prof_scaled = scaler.fit_transform(prof_x)
        svm = SVC(
            kernel="linear",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=args.seed,
            cache_size=512,
        )
        svm.fit(prof_scaled, prof_y)
        save_pickle(svm_path, {"scaler": scaler, "model": svm})

    attack_scaled = scaler.transform(attack_x)
    svm_proba = align_class_probabilities(svm, svm.predict_proba(attack_scaled), n_classes=2)
    svm_best, svm_rank = rank_key_from_binary_proba(svm_proba[:, 1], attack_ct, target_byte=7)

    # CNN
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    cnn_path = models_dir / "cnn_aeshd.keras"
    prof_x_cnn = (prof_x - np.mean(prof_x)) / (np.std(prof_x) + 1e-8)
    attack_x_cnn = (attack_x - np.mean(attack_x)) / (np.std(attack_x) + 1e-8)

    if cnn_path.exists() and not args.force_retrain:
        cnn = load_model(cnn_path)
    else:
        cnn = build_1dcnn_model(input_shape=prof_x.shape[1], num_classes=2)
        early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
        cnn.fit(
            prof_x_cnn,
            prof_y,
            validation_split=0.1,
            epochs=args.cnn_epochs,
            batch_size=args.cnn_batch_size,
            callbacks=[early_stop],
            verbose=0,
        )
        cnn.save(cnn_path)

    cnn_proba = cnn.predict(attack_x_cnn, verbose=0)
    cnn_best, cnn_rank = rank_key_from_binary_proba(cnn_proba[:, 1], attack_ct, target_byte=7)

    results = {
        "dataset": "AES-HD",
        "script": "main_models_cached_aeshd.py",
        "rf": {
            "best_recovered_key_byte_7": f"0x{rf_best:02X}",
            "key_rank": int(rf_rank),
            "artifact": str(rf_path),
        },
        "svm": {
            "best_recovered_key_byte_7": f"0x{svm_best:02X}",
            "key_rank": int(svm_rank),
            "artifact": str(svm_path),
        },
        "cnn": {
            "best_recovered_key_byte_7": f"0x{cnn_best:02X}",
            "key_rank": int(cnn_rank),
            "artifact": str(cnn_path),
            "epochs": int(args.cnn_epochs),
            "batch_size": int(args.cnn_batch_size),
        },
        "seed": int(args.seed),
    }

    out_json = out_dir / "cached_models_results_aeshd.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[AESHD-CACHED] Completed.")
    print(f"[AESHD-CACHED] Results: {out_json}")


if __name__ == "__main__":
    main()
