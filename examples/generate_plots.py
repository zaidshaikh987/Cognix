"""
COGNIX Plot Generation for RQ_SYNTHETIC_001
Reads aggregated metrics from results/RQ_SYNTHETIC_001/metrics.json
and generates validation plots or a markdown report.
"""
import os
import json

def generate_markdown_report(metrics: list, output_dir: str):
    """Fallback if matplotlib is not available."""
    md_path = os.path.join(output_dir, "RESULTS_SUMMARY.md")
    
    # Sort logically
    cond_order = ["NORMAL", "HIGH_NOISE", "OOD_SHIFT", "MISSING_AGENT"]
    bline_order = ["uniform", "confidence", "bayesian", "epistemic_weighted"]
    
    metrics.sort(key=lambda x: (
        cond_order.index(x["condition"]) if x["condition"] in cond_order else 99,
        bline_order.index(x["baseline"]) if x["baseline"] in bline_order else 99
    ))
    
    lines = [
        "# RQ_SYNTHETIC_001 Aggregated Results\n",
        "| Condition | Baseline | ECE (Mean ± Std) | Acc (Mean ± Std) | Brier (Mean) |",
        "|-----------|----------|------------------|------------------|--------------|"
    ]
    
    for m in metrics:
        lines.append(
            f"| {m['condition']} | {m['baseline']} | "
            f"{m['ece_mean']:.4f} ± {m['ece_std']:.4f} | "
            f"{m['acc_mean']:.3f} ± {m['acc_std']:.3f} | "
            f"{m['brier_mean']:.4f} |"
        )
        
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown summary saved to {md_path}")

def generate_plots():
    results_dir = os.path.join("results", "RQ_SYNTHETIC_001")
    metrics_file = os.path.join(results_dir, "metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Could not find {metrics_file}. Run rq_synthetic_001.py first.")
        return
        
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # We can do basic seaborn bar plots for ECE across conditions
        import pandas as pd
        df = pd.DataFrame(metrics)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="condition", y="ece_mean", hue="baseline")
        plt.title("Expected Calibration Error (Lower is Better)")
        plt.ylabel("ECE")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "ece_comparison.png"))
        plt.close()
        
        print(f"Plots saved to {plots_dir}")
        
    except ImportError:
        print("Matplotlib/Seaborn not installed. Falling back to Markdown report.")
        generate_markdown_report(metrics, plots_dir)

if __name__ == "__main__":
    generate_plots()
