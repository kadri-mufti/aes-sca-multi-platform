"""
Unified comparison and evaluation of all ML methods on STM32F4.

Aggregates results from:
  - Random Forest (main_rf_stm32f4.py)
  - SVM (main_svm_stm32f4.py)
  - 1D CNN (main_cnn1d_stm32f4.py)
  - Feature selection experiments (main_rf_pca_stm32f4.py)

Generates:
  - Comparative plots (key rank vs. trace count, all methods overlaid)
  - Summary table: method, bytes_recovered, avg_key_rank, best/worst rank
  - Recommendation: which method performs best
"""

import json
import io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_results_json(results_dir, pattern):
    """
    Load JSON results files matching a pattern.
    
    Args:
        results_dir: Path to results directory
        pattern: File name pattern (e.g., "*_results_stm32f4.json")
        
    Returns:
        dict: {method_name -> results_dict}
    """
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


def main(results_dir="results/stm32f4_ml", output_file="comparison_stm32f4.txt"):
    """
    Main function: compare all methods and generate report.
    
    Args:
        results_dir: Path to results directory
        output_file: Output file for text report
    """
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Comparison-STM32F4] Loading results from {results_dir}...")
    
    # Load all results
    ml_results = load_results_json(results_dir, "*_results_stm32f4.json")
    
    if not ml_results:
        print("✗ No results found. Run ML scripts first:")
        print("  python main_rf_stm32f4.py")
        print("  python main_svm_stm32f4.py")
        print("  python main_cnn1d_stm32f4.py")
        return
    
    print(f"  Loaded {len(ml_results)} method result(s)")
    
    # Prepare summary table
    summary = []
    
    for method, data in sorted(ml_results.items()):
        key_ranks = data.get("key_ranks", {})
        correct_count = data.get("correctly_recovered", 0)
        
        if isinstance(key_ranks, dict):
            ranks_list = [int(v) for v in key_ranks.values()]
        else:
            ranks_list = key_ranks
        
        avg_rank = np.mean(ranks_list) if ranks_list else 0
        best_rank = min(ranks_list) if ranks_list else 0
        worst_rank = max(ranks_list) if ranks_list else 0
        
        summary.append({
            "method": method.replace(" ", "-"),
            "correct": correct_count,
            "avg_rank": avg_rank,
            "best_rank": best_rank,
            "worst_rank": worst_rank,
            "data": data,
        })
    
    # Generate text report
    with open(results_dir / output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STM32F4 ML Methods Comparison Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("Summary Table:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<25} {'Bytes OK':<12} {'Avg Rank':<12} {'Best':<8} {'Worst':<8}\n")
        f.write("-" * 80 + "\n")
        
        for item in sorted(summary, key=lambda x: x["avg_rank"]):
            f.write(f"{item['method']:<25} {item['correct']/16*100:>10.1f}% {item['avg_rank']:>10.2f}  {item['best_rank']:>6} {item['worst_rank']:>6}\n")
        
        f.write("-" * 80 + "\n\n")
        
        # Detailed results for each method
        f.write("Detailed Results:\n")
        f.write("=" * 80 + "\n\n")
        
        for item in sorted(summary, key=lambda x: x["avg_rank"]):
            data = item["data"]
            f.write(f"Method: {item['method']}\n")
            f.write(f"  Dataset: {data.get('dataset', 'N/A')}\n")
            f.write(f"  Parameters:\n")
            
            for key in data.keys():
                if key not in ["method", "dataset", "key_ranks", "correctly_recovered", "seed"]:
                    f.write(f"    {key}: {data[key]}\n")
            
            f.write(f"  Key Ranks: {item['correct']}/16 recovered correctly\n")
            f.write(f"    Average: {item['avg_rank']:.2f}\n")
            f.write(f"    Best:    {item['best_rank']}\n")
            f.write(f"    Worst:   {item['worst_rank']}\n")
            f.write("\n")
        
        # Recommendation
        f.write("=" * 80 + "\n")
        f.write("Recommendation:\n")
        f.write("=" * 80 + "\n")
        best_method = min(summary, key=lambda x: x["avg_rank"])
        f.write(f"Best performing method: {best_method['method']}\n")
        f.write(f"  Average Key Rank: {best_method['avg_rank']:.2f}\n")
        f.write(f"  Bytes Fully Recovered: {best_method['correct']}/16\n\n")
    
    print(f"\n[Comparison-STM32F4] Saved report: {results_dir / output_file}")
    
    # Generate comparative plots
    print(f"\n[Comparison-STM32F4] Generating comparison plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Key rank comparison (bar chart)
    ax = axes[0]
    methods = [item["method"] for item in sorted(summary, key=lambda x: x["avg_rank"])]
    avg_ranks = [item["avg_rank"] for item in sorted(summary, key=lambda x: x["avg_rank"])]
    
    bars = ax.bar(range(len(methods)), avg_ranks, color='steelblue', alpha=0.7)
    ax.set_xlabel("Method", fontsize=11)
    ax.set_ylabel("Average Key Rank", fontsize=11)
    ax.set_title("Key Rank Comparison (Lower is Better)", fontsize=12)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, rank) in enumerate(zip(bars, avg_ranks)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{rank:.2f}", ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Bytes recovered (bar chart)
    ax = axes[1]
    correct_counts = [item["correct"] for item in sorted(summary, key=lambda x: x["avg_rank"])]
    
    bars = ax.bar(range(len(methods)), correct_counts, color='limegreen', alpha=0.7)
    ax.set_xlabel("Method", fontsize=11)
    ax.set_ylabel("Number of Bytes Recovered (16 max)", fontsize=11)
    ax.set_title("Correct Key Byte Recovery", fontsize=12)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim([0, 16])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, correct_counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{count}", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    comp_plot = results_dir / "comparison_stm32f4_all_methods.png"
    plt.savefig(comp_plot, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comp_plot}")
    
    # Print to console
    print(f"\n[Comparison-STM32F4] Summary:")
    with open(results_dir / output_file, 'r') as f:
        for line in f:
            print(line.rstrip())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare all ML methods on STM32F4"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/stm32f4_ml",
        help="Results directory path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_stm32f4.txt",
        help="Output file name",
    )
    
    args = parser.parse_args()
    main(results_dir=args.results_dir, output_file=args.output)
