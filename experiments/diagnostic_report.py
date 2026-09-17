import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognix import DecisionEngine, CognixConfig, UncertaintyDecomposition, EpistemicWeightedFusion, ConformalPredictor, EscalationEngine
from cognix.adapters.carla.dataset import CarlAnomalyDataset
from cognix.adapters.carla.agents import CameraAgent, DepthAgent, LiDARAgent, GNSSAgent, IMUAgent, SegAgent
from experiments.run_20seed_benchmark_clean import make_agents, frame_to_inputs, split_frames, calibrate_conformal, MODELS

seed = 42
N_FRAMES = 200
SCENARIOS = ["NORMAL", "HIGH_NOISE", "OOD", "MISSING_AGENT", "PARTIAL_FAILURE", "MULTIPLE_FAILURE", "CONFLICT"]

def run_diagnostic():
    print("============================================================")
    print("COGNIX DIAGNOSTIC REPORT (Seed 42)")
    print("============================================================\n")

    dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=N_FRAMES, seed=seed)
    all_frames = dataset.get_all_scenarios()
    split_rng = np.random.default_rng(seed)

    splits = {}
    gat_train_frames = []
    gat_train_labels = []

    for scenario in SCENARIOS:
        train, cal, test = split_frames(all_frames[scenario], split_rng)
        splits[scenario] = (train, cal, test)
        for f in train:
            gat_train_frames.append(frame_to_inputs(f))
            gat_train_labels.append(f.label)

    engines = {}
    for model_name, cfg in MODELS.items():
        config = CognixConfig()
        config.graph.method = cfg["method"]
        calibrator = ConformalPredictor()
        
        graph_module = None
        if cfg["method"] == "gat":
            from cognix.graph.epistemic_gat import EpistemicGAT
            graph_module = EpistemicGAT(use_epistemic_prior=False)
        elif cfg["method"] == "epistemic_gat":
            from cognix.graph.epistemic_gat import EpistemicGAT
            graph_module = EpistemicGAT(use_epistemic_prior=True)

        engine = DecisionEngine(
            config=config,
            uncertainty=UncertaintyDecomposition(),
            belief=EpistemicWeightedFusion(),
            calibrator=calibrator,
            escalation=EscalationEngine(),
            graph=graph_module,
            mode="production"
        )
        engines[model_name] = engine

        if graph_module is not None:
            agents_for_train = make_agents()
            graph_module.fit(
                agents_for_train,
                gat_train_frames,
                gat_train_labels,
                epochs=30,
                lr=1e-3,
                verbose=False,
                seed=seed
            )

    for scenario in SCENARIOS:
        print(f"\n[{scenario}]")
        train_frames, cal_frames, test_frames = splits[scenario]
        
        true_labels = [f.label for f in test_frames]
        n_class0 = sum(1 for y in true_labels if y == 0)
        n_class1 = sum(1 for y in true_labels if y == 1)
        print(f"  Test-set class distribution: Class 0: {n_class0} ({n_class0/len(true_labels):.1%}), Class 1: {n_class1} ({n_class1/len(true_labels):.1%})")

        for model_name, engine in engines.items():
            print(f"\n  -- {model_name} --")
            calibrate_conformal(engine, cal_frames)
            
            agents = make_agents()
            probs_class1 = []
            pred_classes = []
            sets = []
            
            samples = []

            for i, frame in enumerate(test_frames):
                inputs = frame_to_inputs(frame)
                
                # capture agent predictions directly for samples
                if i < 10 and scenario not in ["NORMAL", "HIGH_NOISE"]:
                    agent_preds = {}
                    for a in agents:
                        p = a.predict(inputs)
                        u = a.estimate_uncertainty(inputs)
                        agent_preds[a.agent_id] = {"p": p.value, "u_e": u.epistemic}

                result = engine.decide(agents, inputs, {a.agent_id: 1.0 for a in agents})
                
                prob = result.calibrated_confidence if result.calibrated_confidence else result.confidence
                pred_label = 1 if result.risk_level.name in ["HIGH", "CRITICAL"] else 0
                prob_1 = prob if pred_label == 1 else 1.0 - prob
                
                probs_class1.append(prob_1)
                pred_classes.append(pred_label)
                
                c_set = result.calibration_metrics.get("prediction_set", []) if result.calibration_metrics else []
                sets.append(tuple(sorted(c_set)))
                
                if i < 10 and scenario not in ["NORMAL", "HIGH_NOISE"]:
                    samples.append({
                        "id": i,
                        "true_label": frame.label,
                        "agents": agent_preds,
                        "final_prob_1": prob_1,
                        "pred_label": pred_label,
                        "conformal_set": c_set
                    })

            probs_class1 = np.array(probs_class1)
            pred_classes = np.array(pred_classes)
            
            n_pred0 = sum(1 for y in pred_classes if y == 0)
            n_pred1 = sum(1 for y in pred_classes if y == 1)
            frac_gt_05 = np.mean(probs_class1 > 0.5)
            
            print(f"    mean P(class 1) = {np.mean(probs_class1):.4f}")
            print(f"    min  P(class 1) = {np.min(probs_class1):.4f}")
            print(f"    max  P(class 1) = {np.max(probs_class1):.4f}")
            print(f"    median P(class 1) = {np.median(probs_class1):.4f}")
            print(f"    std  P(class 1) = {np.std(probs_class1):.4f}")
            print(f"    predicted class 0 = {n_pred0} ({n_pred0/len(pred_classes):.1%})")
            print(f"    predicted class 1 = {n_pred1} ({n_pred1/len(pred_classes):.1%})")
            print(f"    fraction predictions > 0.5 = {frac_gt_05:.1%}")
            
            # Confusion matrix
            tp = np.sum((pred_classes == 1) & (true_labels == 1))
            tn = np.sum((pred_classes == 0) & (true_labels == 0))
            fp = np.sum((pred_classes == 1) & (true_labels == 0))
            fn = np.sum((pred_classes == 0) & (true_labels == 1))
            print(f"    Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
            
            # Conformal sets
            set_counts = { (): 0, (0,): 0, (1,): 0, (0,1): 0 }
            for s in sets:
                if s not in set_counts: set_counts[s] = 0
                set_counts[s] += 1
            print(f"    Conformal Sets: {{}}: {set_counts.get((),0)/len(sets):.1%}, {{0}}: {set_counts.get((0,),0)/len(sets):.1%}, {{1}}: {set_counts.get((1,),0)/len(sets):.1%}, {{0,1}}: {set_counts.get((0,1),0)/len(sets):.1%}")

            if samples and model_name == "EpistemicGAT":
                print(f"\n    [10 Representative Samples for EpistemicGAT]")
                for s in samples:
                    agents_str = " ".join([f"{k}: p={v['p']:.2f}/ue={v['u_e']:.2f}" for k,v in s["agents"].items()])
                    print(f"      Sample {s['id']}: True={s['true_label']} | {agents_str} | Final P(1)={s['final_prob_1']:.4f} Pred={s['pred_label']} Set={s['conformal_set']}")

if __name__ == '__main__':
    run_diagnostic()
