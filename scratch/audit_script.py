import os
import ast
import json
import time
import numpy as np
import torch
from cognix import DecisionEngine
from cognix.config.schema import CognixConfig
from examples.rq_synthetic_001 import DataGenerator, FeatureModel, MultiFeatureModel, HeterogeneousAgent

def perform_codebase_scan(search_dirs=["cognix", "examples", "dashboard"]):
    issues = []
    keywords = ["TODO", "FIXME", "NotImplementedError", "pass", "random", "mock"]
    
    for d in search_dirs:
        for root, _, files in os.walk(d):
            for file in files:
                if not file.endswith(".py") and not file.endswith(".js"): continue
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines):
                        for kw in keywords:
                            if kw.lower() in line.lower() and "import" not in line and "audit" not in line:
                                issues.append({
                                    "file": path,
                                    "line": i+1,
                                    "content": line.strip(),
                                    "keyword": kw
                                })
                except Exception as e:
                    pass
    return issues

def trace_single_sample():
    seed = 42
    torch.manual_seed(seed)
    gen = DataGenerator(n_samples=100, seed=seed)
    
    X_train, y_train = gen.generate_train()
    
    agents = [
        HeterogeneousAgent("Agent_A_Feat0", FeatureModel(0)),
        HeterogeneousAgent("Agent_B_Feat1", FeatureModel(1)),
        HeterogeneousAgent("Agent_C_Feat2", FeatureModel(2)),
        HeterogeneousAgent("Agent_D_AllFeat", MultiFeatureModel()),
    ]
    
    print("\n--- 1. TRAINING ---")
    for agent in agents:
        agent.fit(X_train, y_train, epochs=10)
        
    print("\n--- 2. GENERATING ONE OOD SAMPLE ---")
    X_test, y_test, ood_labels = gen.generate_test(condition="OOD_SHIFT")
    sample_x = X_test[0]
    sample_y = y_test[0]
    
    print(f"Sample Features: {sample_x}")
    print(f"Ground Truth: {sample_y}")
    
    config = CognixConfig()
    engine = DecisionEngine(config=config, belief="epistemic_weighted", attribution="epistemic_shapley")
    
    print("\n--- 3. TRACING ENGINE.DECIDE() ---")
    t0 = time.perf_counter()
    res = engine.decide(agents, sample_x)
    latency = (time.perf_counter() - t0) * 1000
    
    print(f"Agent Predictions: {res.agent_predictions}")
    print(f"Agent Epistemic UQ: {res.epistemic_uncertainty}")
    print(f"Trust Weights: {res.agent_trust_weights}")
    print(f"Fused Belief: {res.fused_belief}")
    print(f"Calibrated Confidence: {res.calibrated_confidence}")
    print(f"Decision: {res.decision.value}")
    print(f"Risk Level: {res.risk_level.value}")
    print(f"Attribution: {res.agent_contributions}")
    print(f"Latency: {latency:.2f}ms")

if __name__ == "__main__":
    print("===== CODEBASE SCAN =====")
    issues = perform_codebase_scan()
    for iss in issues:
        print(f"{iss['file']}:{iss['line']} | {iss['keyword']} | {iss['content']}")
        
    print("\n===== SAMPLE TRACE =====")
    trace_single_sample()
