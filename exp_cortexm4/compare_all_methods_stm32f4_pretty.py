"""
Pretty markdown comparison for STM32F4 ML outputs.

Additive companion formatter that summarizes available JSON outputs into
concise markdown + CSV for easy report insertion.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(results_dir, output_md, output_csv):
    rdir = Path(results_dir)
    files = {
        "rf": rdir / "rf_results_stm32f4.json",
        "svm": rdir / "svm_results_stm32f4.json",
        "cnn": rdir / "cnn1d_results_stm32f4.json",
        "fs": rdir / "feature_selection_results_stm32f4.json",
        "cached": rdir / "cached_models_results_stm32f4.json",
    }

    rows = []

    if files["rf"].exists():
        d = load_json(files["rf"])
        ranks = [int(v) for v in d.get("key_ranks", {}).values()]
        rows.append(["RF", "avg_key_rank(bytes listed)", f"{np.mean(ranks):.2f}" if ranks else "NA", str(files["rf"].name)])

    if files["svm"].exists():
        d = load_json(files["svm"])
        ranks = [int(v) for v in d.get("key_ranks", {}).values()]
        rows.append(["SVM", "avg_key_rank(bytes listed)", f"{np.mean(ranks):.2f}" if ranks else "NA", str(files["svm"].name)])

    if files["cnn"].exists():
        d = load_json(files["cnn"])
        ranks = d.get("key_ranks", {})
        if isinstance(ranks, dict) and ranks:
            b0 = list(ranks.items())[0]
            rows.append(["CNN1D", f"key_rank(byte {b0[0]})", str(b0[1]), str(files["cnn"].name)])

    if files["fs"].exists():
        d = load_json(files["fs"])
        best = None
        best_tag = ""
        for section in ["pca_results", "mi_results", "variance_results"]:
            for k, vals in d.get(section, {}).items():
                avg = float(np.mean(vals))
                if best is None or avg < best:
                    best = avg
                    best_tag = f"{section}:{k}"
        if best is not None:
            rows.append(["RF+FS", f"best_avg_rank({best_tag})", f"{best:.2f}", str(files["fs"].name)])

    if files["cached"].exists():
        d = load_json(files["cached"])
        rf_avg = d.get("rf", {}).get("avg_rank")
        svm_avg = d.get("svm", {}).get("avg_rank")
        cnn_rank = d.get("cnn", {}).get("key_rank")
        if rf_avg is not None:
            rows.append(["RF-CACHED", "avg_key_rank", f"{rf_avg:.2f}", str(files["cached"].name)])
        if svm_avg is not None:
            rows.append(["SVM-CACHED", "avg_key_rank", f"{svm_avg:.2f}", str(files["cached"].name)])
        if cnn_rank is not None:
            rows.append(["CNN-CACHED", "key_rank", str(cnn_rank), str(files["cached"].name)])

    out_md_path = rdir / output_md
    out_csv_path = rdir / output_csv

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# STM32F4 ML Comparison (Pretty Summary)\n\n")
        if not rows:
            f.write("No result JSON files found.\n")
        else:
            f.write("| Method | Metric | Value | Source |\n")
            f.write("|---|---|---:|---|\n")
            for r in rows:
                f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n")

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "value", "source"])
        w.writerows(rows)

    print(f"Saved {out_md_path}")
    print(f"Saved {out_csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="../results/stm32f4_ml")
    p.add_argument("--output-md", default="comparison_stm32f4_pretty.md")
    p.add_argument("--output-csv", default="comparison_stm32f4_pretty.csv")
    a = p.parse_args()
    main(a.results_dir, a.output_md, a.output_csv)
