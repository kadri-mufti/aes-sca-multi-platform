"""
Unified comparison and evaluation of all ML methods on AES-HD FPGA.

Aggregates results from:
  - Random Forest (main_rf_aeshd.py)
  - SVM (main_svm_aeshd.py)
  - 1D CNN (main_cnn1d_aeshd.py)

Generates:
  - Comparative plots (key rank vs. profiling traces)
  - Summary table: method, key_recovery, final_rank
  - Comparison with classical attacks (DPA, CPA) when available
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def extract_recovery_fields(data):
    """Extract comparable recovery fields from flat or cached nested schemas."""
    recovered_key = data.get("best_recovered_key_byte_7")
    key_rank = data.get("key_rank")
    conv_ranks = data.get("convergence_ranks", [])

    if recovered_key is not None and key_rank is not None:
        return recovered_key, key_rank, conv_ranks

    nested_candidates = []
    for section in ["rf", "svm", "cnn"]:
        item = data.get(section)
        if isinstance(item, dict) and item.get("key_rank") is not None:
            nested_candidates.append(item)

    if nested_candidates:
        best_item = min(nested_candidates, key=lambda x: x.get("key_rank", 256))
        return (
            best_item.get("best_recovered_key_byte_7", "Unknown"),
            best_item.get("key_rank", 256),
            [],
        )

    return "Unknown", 256, []


def load_results_json(results_dir, pattern):
    """Load JSON results files matching a pattern."""
    results_dir = Path(results_dir)
    results = {}
    
    for f in results_dir.glob(pattern):
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                method = data.get("method", f.stem)
                results[method] = data
        except Exception as e:
            print(f"  ⚠ Failed to load {f.name}: {e}")
    
    return results


def main(results_dir="results/aeshd_ml", output_file="comparison_aeshd.txt"):
    """
    Main function: compare all methods and generate report.
    
    Args:
        results_dir: Path to results directory
        output_file: Output file for text report
    """
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Comparison-AES_HD] Loading results from {results_dir}...")
    
    # Load ML results
    ml_results = load_results_json(results_dir, "*_results_aeshd.json")
    
    if not ml_results:
        print("✗ No results found. Run ML scripts first:")
        print("  cd exp_aeshd_hd")
        print("  python main_rf_aeshd.py")
        print("  python main_svm_aeshd.py")
        print("  python main_cnn1d_aeshd.py")
        return
    
    print(f"  Loaded {len(ml_results)} ML method result(s)")
    
    # Prepare summary
    summary = []
    
    for method, data in sorted(ml_results.items()):
        recovered_key, key_rank, conv_ranks = extract_recovery_fields(data)
        final_rank = conv_ranks[-1] if conv_ranks else key_rank
        
        summary.append({
            "method": method.replace(" ", "-"),
            "key": recovered_key,
            "rank": key_rank,
            "final_rank": final_rank,
            "convergence": conv_ranks,
            "data": data,
        })
    
    # Generate text report
    with open(results_dir / output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AES-HD FPGA ML Methods Comparison Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("Summary Table (Last-Round Key Byte 7):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<25} {'Recovered Key':<18} {'Initial Rank':<15} {'Final Rank':<15}\n")
        f.write("-" * 80 + "\n")
        
        for item in sorted(summary, key=lambda x: x["final_rank"]):
            f.write(f"{item['method']:<25} {str(item['key']):<18} {item['rank']:<15} {item['final_rank']:<15}\n")
        
        f.write("-" * 80 + "\n\n")
        
        # Detailed results
        f.write("Detailed Results:\n")
        f.write("=" * 80 + "\n\n")
        
        for item in sorted(summary, key=lambda x: x["final_rank"]):
            data = item["data"]
            f.write(f"Method: {item['method']}\n")
            f.write(f"  Dataset: {data.get('dataset', 'N/A')}\n")
            f.write(f"  Configuration:\n")
            
            for key in sorted(data.keys()):
                if key not in ["method", "dataset", "best_recovered_key_byte_7", "key_rank", "convergence_ranks", "seed"]:
                    f.write(f"    {key}: {data[key]}\n")
            
            f.write(f"  Recovery:\n")
            f.write(f"    Best key (byte 7): {item['key']}\n")
            f.write(f"    Initial rank: {item['rank']}\n")
            f.write(f"    Final rank: {item['final_rank']}\n")
            
            if item['convergence']:
                f.write(f"    Convergence (ranks by trace count):\n")
                for i, rank in enumerate(item['convergence']):
                    f.write(f"      Step {i}: rank={rank}\n")
            
            f.write("\n")
        
        # Recommendation
        f.write("=" * 80 + "\n")
        f.write("Recommendation:\n")
        f.write("=" * 80 + "\n")
        best = min(summary, key=lambda x: x["final_rank"])
        f.write(f"Best performing method: {best['method']}\n")
        f.write(f"  Final Key Rank: {best['final_rank']}\n")
        f.write(f"  Recovered Key Byte 7: {best['key']}\n\n")
        
        # Note about AES-HD difficulty
        f.write("Note: AES-HD has weak Hamming Distance leakage.\n")
        f.write("Higher key ranks are expected compared to strong HW leakage datasets.\n")
    
    print(f"\n[Comparison-AES_HD] Saved report: {results_dir / output_file}")
    
    # Generate comparative plots
    print(f"\n[Comparison-AES_HD] Generating comparison plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Key rank comparison
    ax = axes[0]
    methods = [item["method"] for item in sorted(summary, key=lambda x: x["final_rank"])]
    final_ranks = [item["final_rank"] for item in sorted(summary, key=lambda x: x["final_rank"])]
    
    bars = ax.bar(range(len(methods)), final_ranks, color='steelblue', alpha=0.7)
    ax.set_xlabel("Method", fontsize=11)
    ax.set_ylabel("Final Key Rank (Lower is Better)", fontsize=11)
    ax.set_title("Key Rank Comparison on Attack Set", fontsize=12)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rank in zip(bars, final_ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rank}", ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Convergence lines (all methods)
    ax = axes[1]
    
    for item in sorted(summary, key=lambda x: x["method"]):
        conv = item["convergence"]
        if conv:
            x_vals = range(1, len(conv) + 1)
            ax.plot(x_vals, conv, marker='o', label=item["method"], linewidth=2, markersize=6)
    
    ax.set_xlabel("Profiling Step (varying trace counts)", fontsize=11)
    ax.set_ylabel("Key Rank", fontsize=11)
    ax.set_title("Convergence During Profiling", fontsize=12)
    ax.grid(True, alpha=0.3)
    if len(summary) <= 5:
        ax.legend()
    
    plt.tight_layout()
    
    comp_plot = results_dir / "comparison_aeshd_all_methods.png"
    plt.savefig(comp_plot, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comp_plot}")
    
    # Print to console
    print(f"\n[Comparison-AES_HD] Summary:")
    with open(results_dir / output_file, 'r') as f:
        for line in f:
            print(line.rstrip())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare all ML methods on AES-HD FPGA"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/aeshd_ml",
        help="Results directory path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_aeshd.txt",
        help="Output file name",
    )
    
    args = parser.parse_args()
    main(results_dir=args.results_dir, output_file=args.output)
