import json
import argparse
import numpy as np
from scipy import stats
from typing import Dict, List, Any

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-8: return 0.0
    return float(np.mean(diff) / std_diff)

def paired_analysis(baseline_metrics: List[float], experimental_metrics: List[float]) -> Dict[str, Any]:
    x = np.array(baseline_metrics)
    y = np.array(experimental_metrics)
    diff = y - x
    mean_diff = float(np.mean(diff))
    
    ci_95 = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=stats.sem(diff))
    
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

def analyze_all_results(json_path: str):
    with open(json_path, "r") as f:
        results = json.load(f)
        
    metrics = ["accuracy", "ece", "nll", "brier_score", "mean_epistemic", "mean_aleatoric", "empirical_coverage", "mean_set_size", "latency_ms"]
    graphs = ["NoGraph", "StandardGAT", "EpistemicGAT"]
    
    data = {g: {m: {} for m in metrics} for g in graphs}
    
    for res in results:
        g_type = res.get("graph_type")
        if g_type not in graphs: continue
        seed = res.get("seed")
        
        for m in metrics:
            val = res.get(m)
            if val is not None:
                data[g_type][m][seed] = val
                
    # 1. Aggregate Means and Std Devs
    print("="*80)
    print("AGGREGATE RESULTS (Mean ± Std Dev over 20 seeds)")
    print("="*80)
    
    for m in metrics:
        print(f"\n--- Metric: {m.upper()} ---")
        for g in graphs:
            vals = list(data[g][m].values())
            if not vals: continue
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            print(f"{g:<15}: {mean_val:.4f} ± {std_val:.4f}")
            
    # 2. Paired Comparisons
    print("\n" + "="*80)
    print("PAIRED STATISTICAL COMPARISONS")
    print("="*80)
    
    comparisons = [
        ("NoGraph", "EpistemicGAT"),
        ("StandardGAT", "EpistemicGAT"),
        ("NoGraph", "StandardGAT")
    ]
    
    for m in ["ece", "nll", "empirical_coverage"]:
        print(f"\n=== METRIC: {m.upper()} ===")
        for base, exp in comparisons:
            seeds_b = set(data[base][m].keys())
            seeds_e = set(data[exp][m].keys())
            common_seeds = sorted(list(seeds_b.intersection(seeds_e)))
            
            if len(common_seeds) < 2: continue
            
            vals_b = [data[base][m][s] for s in common_seeds]
            vals_e = [data[exp][m][s] for s in common_seeds]
            
            res = paired_analysis(vals_b, vals_e)
            
            sig_mark = "*" if res["significant_05"] else " "
            print(f"{exp} vs {base}:")
            print(f"  Diff: {res['mean_difference']:>8.4f}  |  95% CI: [{res['ci_95_lower']:>7.4f}, {res['ci_95_upper']:>7.4f}]  |  Cohen's d: {res['effect_size_cohens_d']:>7.4f}  |  p-val: {res['p_value_wilcoxon']:.4e} {sig_mark}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to JSON summary results")
    args = parser.parse_args()
    analyze_all_results(args.results)
