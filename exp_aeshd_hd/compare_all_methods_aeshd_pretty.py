"""
Pretty markdown comparison for AES-HD ML outputs.

Additive companion formatter that summarizes available JSON outputs into
concise markdown + CSV for easy report insertion.
"""

import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(results_dir, output_md, output_csv):
    rdir = Path(results_dir)
    files = {
        "rf": rdir / "rf_results_aeshd.json",
        "svm": rdir / "svm_results_aeshd.json",
        "cnn": rdir / "cnn1d_results_aeshd.json",
        "cached": rdir / "cached_models_results_aeshd.json",
    }

    rows = []

    for key, label in [("rf", "RF"), ("svm", "SVM"), ("cnn", "CNN1D")]:
        fp = files[key]
        if fp.exists():
            d = load_json(fp)
            recovered = d.get("best_recovered_key_byte_7", "NA")
            rank = d.get("key_rank", "NA")
            rows.append([label, "key_rank(byte7)", str(rank), str(recovered), fp.name])

    if files["cached"].exists():
        d = load_json(files["cached"])
        for label, section in [("RF-CACHED", "rf"), ("SVM-CACHED", "svm"), ("CNN-CACHED", "cnn")]:
            s = d.get(section, {})
            rank = s.get("key_rank")
            rec = s.get("best_recovered_key_byte_7")
            if rank is not None:
                rows.append([label, "key_rank(byte7)", str(rank), str(rec), files["cached"].name])

    out_md_path = rdir / output_md
    out_csv_path = rdir / output_csv

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# AES-HD ML Comparison (Pretty Summary)\n\n")
        if not rows:
            f.write("No result JSON files found.\n")
        else:
            f.write("| Method | Metric | Value | Recovered Byte7 | Source |\n")
            f.write("|---|---|---:|---|---|\n")
            for r in rows:
                f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n")

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "value", "recovered_byte7", "source"])
        w.writerows(rows)

    print(f"Saved {out_md_path}")
    print(f"Saved {out_csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="../results/aeshd_ml")
    p.add_argument("--output-md", default="comparison_aeshd_pretty.md")
    p.add_argument("--output-csv", default="comparison_aeshd_pretty.csv")
    a = p.parse_args()
    main(a.results_dir, a.output_md, a.output_csv)
