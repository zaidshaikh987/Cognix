import argparse
import yaml
import numpy as np
import sys
import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

from cognix.evaluation.benchmark import BenchmarkConfig, ExperimentResult
from cognix.evaluation.runner import BenchmarkRunner
from cognix.evaluation.reporting import JSONReporter, CSVReporter
from cognix.evaluation.scenarios import DataGenerator

from cognix.engine.pipeline import CognixPipeline
from cognix.config.schema import CognixConfig
from cognix.engine.decision_engine import DecisionEngine

# Concrete Implementations for the evaluation
from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.graph.standard_gat import StandardGAT
from cognix.graph.no_graph import NoGraph
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.calibration.conformal import ConformalPredictor, evaluate_coverage
from cognix.core.interfaces import AgentInterface
from cognix.core.types import PredictionResult, UncertaintyResult
from cognix.metrics.evaluation import calculate_ece, accuracy, brier_score

# Reusing the FeatureModel and HeterogeneousAgent from the examples for synthetic domains
class FeatureModel(nn.Module):
    def __init__(self, feature_idx: int):
        super().__init__()
        self.feature_idx = feature_idx
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = x[:, self.feature_idx:self.feature_idx + 1]
        return self.net(feat)

class HeterogeneousAgent(AgentInterface):
    def __init__(self, agent_id: str, feature_idx: int):
        self.agent_id = agent_id
        self.model = FeatureModel(feature_idx)
        self.healthy = True
        self.T = 30
        self.last_pred = 0.5
        self.last_epi = 0.0
        self.last_ale = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        for _ in range(50):
            optimizer.zero_grad()
            t_X = torch.FloatTensor(X)
            t_y = torch.FloatTensor(y).view(-1, 1)
            preds = self.model(t_X)
            loss = criterion(preds, t_y)
            loss.backward()
            optimizer.step()

    def predict(self, x: np.ndarray) -> PredictionResult:
        if not self.healthy:
            return PredictionResult(value=0.5, confidence=0.0)
        self.model.eval()
        with torch.no_grad():
            t_x = torch.FloatTensor(x).view(1, -1)
            self.last_pred = float(self.model(t_x).item())
        return PredictionResult(value=self.last_pred, confidence=1.0)

    def estimate_uncertainty(self, x: Any) -> UncertaintyResult:
        if not self.healthy:
            return UncertaintyResult(prediction=0.5, epistemic=1.0, aleatoric=1.0, total=2.0)
        if x is None:
            raise RuntimeError("Input required")
        self.model.train()
        with torch.no_grad():
            t_x = torch.FloatTensor(x).view(1, -1)
            preds = np.array([self.model(t_x).item() for _ in range(self.T)])
        p_bar = np.mean(preds)
        epi = float(np.mean((preds - p_bar) ** 2))
        ale = float(np.mean(preds * (1.0 - preds)))
        total = epi + ale
        self.last_epi = epi
        self.last_ale = ale
        return UncertaintyResult(prediction=p_bar, epistemic=epi, aleatoric=ale, total=total)
        
    def metadata(self) -> dict:
        return {"id": self.agent_id}


def run_pre_calibration(agents, x, gat, fuser, config):
    agent_order = [a.agent_id for a in agents]
    preds = {}
    uncs = {}
    for a in agents:
        preds[a.agent_id] = float(a.predict(x).value)
        u = a.estimate_uncertainty(x)
        uncs[a.agent_id] = u.epistemic

    nf = []
    for a in agents:
        nf.append([preds[a.agent_id], uncs[a.agent_id], 0.0])
    node_features = torch.tensor(nf, dtype=torch.float32)
    adjacency = torch.ones((len(agents), len(agents)))
    
    g_res = gat.forward(node_features, adjacency, uncs, agent_order)
    
    refined_preds = {}
    for i, a_id in enumerate(agent_order):
        import math
        # Extract the refined scalar probability for each agent
        val = float(g_res.node_outputs[i, 0]) if len(g_res.node_outputs.shape) > 1 else float(g_res.node_outputs[i])
        p_ref = 1.0 / (1.0 + math.exp(-val))
        refined_preds[a_id] = p_ref
        
    f_res = fuser.fuse(predictions=refined_preds, uncertainties=uncs, reliabilities={})
    return float(f_res.probability)

def evaluation_fn(seed: int, config: BenchmarkConfig) -> Dict[str, Any]:
    # 1. Data Generation
    gen = DataGenerator(n_samples=config.num_samples, seed=seed)
    
    # Train / Cal / Test split
    X_train, y_train = gen.generate_base(num_features=config.num_agents)
    X_cal, y_cal = gen.generate_base(num_features=config.num_agents)
    X_test_base, y_test_base = gen.generate_base(num_features=config.num_agents)
    
    # Generate condition OOD shift for test set
    X_test, y_test, _ = gen.apply_condition(
        X_test_base, y_test_base, 
        condition=config.scenario_name,
        noise_level=config.noise_level,
        ood_severity=config.ood_severity,
        missing_agent_idx=config.missing_agents
    )
    
    # 2. Agents Setup
    agents = [HeterogeneousAgent(f"Agent_{i}", i) for i in range(config.num_agents)]
    for agent in agents:
        agent.fit(X_train, y_train)
        
    if config.scenario_name == "MISSING_AGENT":
        for idx in config.missing_agents:
            if idx < len(agents):
                agents[idx].healthy = False

    # Reliability calculation for weighting
    agent_reliabilities = {}
    for agent in agents:
        agent.healthy = True  # Reliability measured under healthy assumption
        preds_cal = [round(agent.predict(X_cal[i]).value) for i in range(len(X_cal))]
        agent_reliabilities[agent.agent_id] = float(np.mean(np.array(preds_cal) == y_cal))
        
    # Set agent health back based on condition
    if config.scenario_name == "MISSING_AGENT":
        for idx in config.missing_agents:
            if idx < len(agents):
                agents[idx].healthy = False
    
    # 3. Instantiate Architecture components
    cgx_config = CognixConfig()
    
    if config.graph_type == "EpistemicGAT":
        gat = EpistemicGAT(input_dim=3, hidden_dim=8, output_dim=1)
        gat.fit(agents, X_train, y_train)
    elif config.graph_type == "StandardGAT":
        gat = StandardGAT(input_dim=3, hidden_dim=8, output_dim=1)
        gat.fit(agents, X_train, y_train)
    elif config.graph_type == "NoGraph":
        gat = NoGraph()
    else:
        raise ValueError(f"Unknown graph type: {config.graph_type}")
        
    fuser = EpistemicWeightedFusion(eps=1e-8)
    
    # 4. Calibration
    cal_probs = []
    for i in range(len(X_cal)):
        p_col = run_pre_calibration(agents, X_cal[i], gat, fuser, cgx_config)
        cal_probs.append([1.0 - p_col, p_col])
        
    calibrator = ConformalPredictor()
    calibrator.fit(cal_outputs=np.array(cal_probs), cal_labels=y_cal.astype(int))
    
    # 5. Full Evaluation Pipeline
    engine = DecisionEngine(
        config=cgx_config,
        belief=fuser,
        attribution="epistemic_shapley",
        graph=gat,
        calibrator=calibrator,
        mode="research"
    )
    # Inject calibrator properly since engine normally instantiates it
    engine.pipeline.calibrator = calibrator
    
    preds = []
    all_pred_sets = []
    epistemics = []
    aleatorics = []
    
    for i in range(len(X_test)):
        x_i = X_test[i]
        res = engine.decide(agents, x_i, agent_reliabilities=agent_reliabilities)
        
        preds.append(res.confidence)
        epistemics.append(res.epistemic_uncertainty)
        aleatorics.append(res.aleatoric_uncertainty)
        
        cal_info = res.calibration_metrics or {}
        ps = cal_info.get("prediction_set", [])
        all_pred_sets.append(ps)
        
    preds = np.array(preds)
    
    # Coverage
    coverage = None
    mean_set_size = 0.0
    if all_pred_sets:
        covered = sum(1 for i, ps in enumerate(all_pred_sets) if int(y_test[i]) in ps)
        coverage = float(covered / len(y_test)) if len(y_test) > 0 else 0.0
        mean_set_size = float(np.mean([len(ps) for ps in all_pred_sets]))
        
    nll = -np.mean(y_test * np.log(preds + 1e-15) + (1 - y_test) * np.log(1 - preds + 1e-15))
        
    return {
        "accuracy": accuracy(preds, y_test),
        "brier_score": brier_score(preds, y_test),
        "nll": float(nll),
        "ece": calculate_ece(preds, y_test),
        "mean_epistemic": float(np.mean(epistemics)),
        "mean_aleatoric": float(np.mean(aleatorics)),
        "empirical_coverage": coverage if coverage else 0.0,
        "mean_set_size": mean_set_size,
        "latency_ms": 0.0, # Will instrument later
        "escalation_rate": 0.0,
        "failure_conditions": [],
        "metadata": {}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML benchmark config")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        yaml_config = yaml.safe_load(f)
        
    # Support overriding graph_type via a list in YAML to auto-run ablations
    graph_types = yaml_config.pop("graph_types", [yaml_config.get("graph_type", "EpistemicGAT")])
    if "graph_type" in yaml_config:
        del yaml_config["graph_type"]
        
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{yaml_config['experiment_id']}_summary.json")
    csv_path = os.path.join(args.output_dir, f"{yaml_config['experiment_id']}_raw_results.csv")
    
    reporters = [JSONReporter(json_path), CSVReporter(csv_path)]
    runner = BenchmarkRunner(reporters)
    
    all_results = []
    for g_type in graph_types:
        cfg = BenchmarkConfig(graph_type=g_type, **yaml_config)
        results = runner.run(cfg, evaluation_fn)
        all_results.extend(results)
        
if __name__ == "__main__":
    main()
