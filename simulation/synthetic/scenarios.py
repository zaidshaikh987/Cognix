"""
Synthetic simulation scenarios for COGNIX validation.
These are SIMULATED conditions, not physical hardware.
"""

def generate_scenario(name):
    base_scenario = {
        "agents": ["Camera", "LiDAR", "Radar"],
        "input_data": {"feature_vector": [1, 2, 3]},
        "context": "Intersection",
        "expected_outcome": "ACT"
    }
    
    scenarios = {
        "NORMAL": base_scenario,
        "CAMERA_DEGRADATION": {**base_scenario, "expected_outcome": "ACT", "note": "High camera uncertainty"},
        "LIDAR_DEGRADATION": {**base_scenario, "expected_outcome": "ACT", "note": "High lidar uncertainty"},
        "SENSOR_CONFLICT": {**base_scenario, "expected_outcome": "ESCALATE"},
        "OOD_ENVIRONMENT": {**base_scenario, "expected_outcome": "ABSTAIN"},
        "COMMUNICATION_FAILURE": {**base_scenario, "expected_outcome": "ACT"},
        "MULTI_AGENT_DISAGREEMENT": {**base_scenario, "expected_outcome": "ESCALATE"},
        "HIGH_COLLECTIVE_UNCERTAINTY": {**base_scenario, "expected_outcome": "ESCALATE"},
        "PARTIAL_OBSERVABILITY": {**base_scenario, "expected_outcome": "ABSTAIN"},
        "RECOVERY": {**base_scenario, "expected_outcome": "ACT"}
    }
    
    return scenarios.get(name, base_scenario)

def get_all_scenarios():
    return [
        "NORMAL", "CAMERA_DEGRADATION", "LIDAR_DEGRADATION", 
        "SENSOR_CONFLICT", "OOD_ENVIRONMENT", "COMMUNICATION_FAILURE",
        "MULTI_AGENT_DISAGREEMENT", "HIGH_COLLECTIVE_UNCERTAINTY", 
        "PARTIAL_OBSERVABILITY", "RECOVERY"
    ]
