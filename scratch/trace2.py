import time
import numpy as np
import torch
from cognix import DecisionEngine
from cognix.config.schema import CognixConfig
from examples.rq_synthetic_001 import DataGenerator, FeatureModel, MultiFeatureModel, HeterogeneousAgent

def trace():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = DataGenerator(n_samples=100, seed=seed)
    
    X_train, y_train = gen.generate_train()
    
    agents = [
        HeterogeneousAgent("Agent_A_Feat0", FeatureModel(0)),
        HeterogeneousAgent("Agent_B_Feat1", FeatureModel(1)),
        HeterogeneousAgent("Agent_C_Feat2", FeatureModel(2)),
        HeterogeneousAgent("Agent_D_AllFeat", MultiFeatureModel()),
    ]
    
    for agent in agents:
        agent.fit(X_train, y_train, epochs=2)
        
    X_test, y_test, ood_labels = gen.generate_test(condition="OOD_SHIFT")
    sample_x = X_test[0]
    sample_y = y_test[0]
    
    print(f"SAMPLE_FEATURES|{sample_x.tolist()}")
    print(f"GROUND_TRUTH|{sample_y}")
    
    config = CognixConfig()
    engine = DecisionEngine(config=config, belief="epistemic_weighted", attribution="epistemic_shapley")
    
    t0 = time.perf_counter()
    res = engine.decide(agents, sample_x)
    latency = (time.perf_counter() - t0) * 1000
    
    print(f"AGENT_PREDS|{res.agent_predictions}")
    print(f"AGENT_EPI|{res.epistemic_uncertainty}")
    print(f"TRUST_WEIGHTS|{res.agent_trust_weights}")
    print(f"FUSED|{res.fused_belief}")
    print(f"CALIBRATED|{res.calibrated_confidence}")
    print(f"DECISION|{res.decision.value}")
    print(f"RISK|{res.risk_level.value}")
    print(f"SHAPLEY|{res.agent_contributions}")
    print(f"LATENCY|{latency:.2f}")

trace()
