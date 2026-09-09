import json
import argparse
import numpy as np
from scipy import stats
from typing import Dict, List, Any

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d for paired samples."""
    diff = x - y
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))

def paired_analysis(baseline_metrics: List[float], experimental_metrics: List[float]) -> Dict[str, Any]:
    """Perform Wilcoxon signed-rank test and compute effect size."""
    x = np.array(baseline_metrics)
    y = np.array(experimental_metrics)
    
    # Paired difference
    diff = y - x
    mean_diff = float(np.mean(diff))
    
    # 95% Confidence Interval for the mean difference
    ci_95 = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=stats.sem(diff))
    
    # Wilcoxon signed-rank test
    try:
        w_stat, p_val = stats.wilcoxon(x, y)
    except ValueError:
        w_stat, p_val = float('nan'), float('nan')
        
    return {
        "mean_baseline": float(np.mean(x)),
        "mean_experimental": float(np.mean(y)),
        "mean_difference": mean_diff,
        "ci_95_lower": float(ci_95[0]) if not np.isnan(ci_95[0]) else 0.0,
        "ci_95_upper": float(ci_95[1]) if not np.isnan(ci_95[1]) else 0.0,
        "p_value_wilcoxon": float(p_val),
        "effect_size_cohens_d": cohens_d(y, x),
        "significant_05": bool(p_val < 0.05)
    }

def analyze_benchmark_results(json_path: str, baseline_graph: str, experimental_graph: str, target_metric: str = "ece") -> None:
    with open(json_path, "r") as f:
        results = json.load(f)
        
    # Group by seed and graph type
    data = {}
    for res in results:
        g_type = res["graph_type"]
        seed = res["seed"]
        val = res[target_metric]
        
        if g_type not in data:
            data[g_type] = {}
        data[g_type][seed] = val
        
    if baseline_graph not in data or experimental_graph not in data:
        print(f"Graph types {baseline_graph} or {experimental_graph} not found in results.")
        return
        
    # Find overlapping seeds
    seeds_b = set(data[baseline_graph].keys())
    seeds_e = set(data[experimental_graph].keys())
    common_seeds = sorted(list(seeds_b.intersection(seeds_e)))
    
    if len(common_seeds) < 5:
        print("Warning: Less than 5 paired seeds found. Statistical testing may be unreliable.")
        
    baseline_vals = [data[baseline_graph][s] for s in common_seeds]
    experimental_vals = [data[experimental_graph][s] for s in common_seeds]
    
    stats_res = paired_analysis(baseline_vals, experimental_vals)
    
    print(f"\nStatistical Analysis: {baseline_graph} vs {experimental_graph} ({target_metric})")
    print(f"{'-'*60}")
    print(f"Paired Seeds (N={len(common_seeds)})")
    print(f"Baseline Mean: {stats_res['mean_baseline']:.4f}")
    print(f"Experimental Mean: {stats_res['mean_experimental']:.4f}")
    print(f"Mean Difference: {stats_res['mean_difference']:.4f}")
    print(f"95% CI of Difference: [{stats_res['ci_95_lower']:.4f}, {stats_res['ci_95_upper']:.4f}]")
    print(f"Effect Size (Cohen's d): {stats_res['effect_size_cohens_d']:.4f}")
    print(f"Wilcoxon p-value: {stats_res['p_value_wilcoxon']:.4e} (Significant at a=0.05: {stats_res['significant_05']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to JSON summary results")
    parser.add_argument("--baseline", default="StandardGAT", help="Baseline graph type")
    parser.add_argument("--experimental", default="EpistemicGAT", help="Experimental graph type")
    parser.add_argument("--metric", default="ece", help="Metric to test (e.g. ece, accuracy, nll)")
    args = parser.parse_args()
    
    analyze_benchmark_results(args.results, args.baseline, args.experimental, args.metric)
