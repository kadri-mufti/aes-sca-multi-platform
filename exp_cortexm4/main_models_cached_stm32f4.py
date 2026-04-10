"""
Cached ML training/evaluation for STM32F4 side-channel experiments.

This additive companion script trains RF/SVM/1D-CNN models and saves model
artifacts so subsequent runs can skip retraining.

Artifacts:
  - RF models: pickle files
  - SVM models + scaler: pickle files
  - CNN models: .keras files
  - Metadata: JSON file with model paths and metrics
"""

import argparse
import json
import os
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

from ml_data_utils_cortexm4 import AES, hamming_weight, load_stm32f4_dataset, align_class_probabilities


def build_1dcnn_model(input_shape, num_classes=9):
    model = Sequential([
        Reshape((input_shape, 1), input_shape=(input_shape,)),
        Conv1D(32, 11, padding="same", activation="relu"),
        BatchNormalization(),
        Conv1D(32, 11, padding="same", activation="relu"),
        MaxPooling1D(2),
        Conv1D(64, 11, padding="same", activation="relu"),
        BatchNormalization(),
        Conv1D(64, 11, padding="same", activation="relu"),
        MaxPooling1D(2),
        Conv1D(128, 11, padding="same", activation="relu"),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def key_rank_from_hw_proba(hw_proba, plaintexts, key, byte_idx):
    scores = np.zeros(256, dtype=float)
    pt_byte = plaintexts[:, byte_idx]
    true_key_byte = int(key[byte_idx])

    for k in range(256):
        sbox_out = AES.SBOX[(pt_byte ^ k).astype(np.uint8)].astype(np.uint8)
        expected_hw = hamming_weight(sbox_out)
        ll = 0.0
        for i in range(len(expected_hw)):
            ll += np.log(hw_proba[i, int(expected_hw[i])] + 1e-40)
        scores[k] = ll

    return int(np.sum(scores >= scores[true_key_byte]))


def save_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_rf(train_x, train_y, test_x, byte_idx, model_path, force_retrain, seed):
    if model_path.exists() and not force_retrain:
        model = load_pickle(model_path)
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=35,
            random_state=seed + byte_idx,
            n_jobs=-1,
        )
        model.fit(train_x, train_y)
        save_pickle(model_path, model)

    proba = align_class_probabilities(model, model.predict_proba(test_x), n_classes=9)
    return model, proba


def run_svm(train_x, train_y, test_x, byte_idx, model_path, force_retrain, seed):
    if model_path.exists() and not force_retrain:
        payload = load_pickle(model_path)
        scaler = payload["scaler"]
        model = payload["model"]
    else:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_x)
        model = SVC(
            kernel="linear",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=seed + byte_idx,
            cache_size=512,
        )
        model.fit(train_scaled, train_y)
        save_pickle(model_path, {"scaler": scaler, "model": model})

    test_scaled = scaler.transform(test_x)
    proba = align_class_probabilities(model, model.predict_proba(test_scaled), n_classes=9)
    return model, scaler, proba


def run_cnn(train_x, train_y, test_x, model_path, force_retrain, epochs, batch_size, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    if model_path.exists() and not force_retrain:
        model = load_model(model_path)
    else:
        model = build_1dcnn_model(input_shape=train_x.shape[1], num_classes=9)
        early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
        model.fit(
            train_x,
            train_y,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )
        model.save(model_path)

    proba = model.predict(test_x, verbose=0)
    return model, proba


def main():
    parser = argparse.ArgumentParser(description="Cached STM32F4 ML models")
    parser.add_argument("--hdf5", type=str, default="../analysis/traces.hdf5")
    parser.add_argument("--output-dir", type=str, default="../results/stm32f4_ml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bytes", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--cnn-epochs", type=int, default=25)
    parser.add_argument("--cnn-batch-size", type=int, default=32)
    parser.add_argument("--max-train-traces", type=int, default=3000)
    parser.add_argument("--max-test-traces", type=int, default=1500)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("[STM32F4-CACHED] Loading dataset...", flush=True)

    data = load_stm32f4_dataset(
        args.hdf5,
        train_split=0.8,
        seed=args.seed,
        target_byte=None,
    )
    train_x = data["train_traces"]
    test_x = data["test_traces"]
    train_labels = data["train_labels"]
    key = data["key"]
    plaintexts = data["plaintexts"][data["test_indices"]]

    if args.max_train_traces > 0 and train_x.shape[0] > args.max_train_traces:
        train_x = train_x[: args.max_train_traces]
        train_labels = train_labels[: args.max_train_traces]
    if args.max_test_traces > 0 and test_x.shape[0] > args.max_test_traces:
        test_x = test_x[: args.max_test_traces]
        plaintexts = plaintexts[: args.max_test_traces]

    print(f"[STM32F4-CACHED] Train traces: {train_x.shape[0]}, Test traces: {test_x.shape[0]}", flush=True)

    # Normalize traces for CNN only.
    train_x_cnn = (train_x - np.mean(train_x)) / (np.std(train_x) + 1e-8)
    test_x_cnn = (test_x - np.mean(test_x)) / (np.std(test_x) + 1e-8)

    rf_ranks = {}
    svm_ranks = {}

    for b in args.bytes:
        print(f"[STM32F4-CACHED] RF byte {b}...", flush=True)
        rf_path = models_dir / f"rf_byte_{b}.pkl"
        _, rf_proba = run_rf(
            train_x,
            train_labels[:, b],
            test_x,
            b,
            rf_path,
            args.force_retrain,
            args.seed,
        )
        rf_ranks[str(b)] = key_rank_from_hw_proba(rf_proba, plaintexts, key, b)

        print(f"[STM32F4-CACHED] SVM byte {b}...", flush=True)
        svm_path = models_dir / f"svm_byte_{b}.pkl"
        _, _, svm_proba = run_svm(
            train_x,
            train_labels[:, b],
            test_x,
            b,
            svm_path,
            args.force_retrain,
            args.seed,
        )
        svm_ranks[str(b)] = key_rank_from_hw_proba(svm_proba, plaintexts, key, b)

    cnn_byte = int(args.bytes[0]) if args.bytes else 0
    print(f"[STM32F4-CACHED] CNN byte {cnn_byte}...", flush=True)
    cnn_path = models_dir / f"cnn_byte_{cnn_byte}.keras"
    _, cnn_proba = run_cnn(
        train_x_cnn,
        train_labels[:, cnn_byte],
        test_x_cnn,
        cnn_path,
        args.force_retrain,
        args.cnn_epochs,
        args.cnn_batch_size,
        args.seed,
    )
    cnn_rank = key_rank_from_hw_proba(cnn_proba, plaintexts, key, cnn_byte)

    results = {
        "dataset": "STM32F4",
        "script": "main_models_cached_stm32f4.py",
        "bytes_evaluated": [int(x) for x in args.bytes],
        "rf": {
            "key_ranks": rf_ranks,
            "avg_rank": float(np.mean(list(rf_ranks.values()))) if rf_ranks else None,
            "artifacts": [str(models_dir / f"rf_byte_{b}.pkl") for b in args.bytes],
        },
        "svm": {
            "key_ranks": svm_ranks,
            "avg_rank": float(np.mean(list(svm_ranks.values()))) if svm_ranks else None,
            "artifacts": [str(models_dir / f"svm_byte_{b}.pkl") for b in args.bytes],
        },
        "cnn": {
            "byte": cnn_byte,
            "key_rank": int(cnn_rank),
            "artifact": str(cnn_path),
            "epochs": int(args.cnn_epochs),
            "batch_size": int(args.cnn_batch_size),
        },
        "seed": int(args.seed),
    }

    out_json = out_dir / "cached_models_results_stm32f4.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[STM32F4-CACHED] Completed.")
    print(f"[STM32F4-CACHED] Results: {out_json}")


if __name__ == "__main__":
    main()
