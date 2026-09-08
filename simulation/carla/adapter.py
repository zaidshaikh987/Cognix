"""
CARLA Simulation Adapter for COGNIX.

Requires: CARLA 0.9.15 server running and carla Python package installed.
DO NOT import carla at module level — guard the import.

This adapter is NOT a mandatory dependency for the core Cognix framework.
"""

class AgentPrediction:
    def __init__(self, pred, unc):
        self.pred = pred
        self.unc = unc

class CARLASensorAdapter:
    def __init__(self, host="127.0.0.1", port=2000):
        self.host = host
        self.port = port
        self.client = None
        self.world = None

    def connect(self):
        try:
            import carla
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            print("Connected to CARLA")
        except ImportError:
            print("Error: CARLA Python package not installed. Skipping connection.")
        except Exception as e:
            print(f"Error connecting to CARLA: {e}")

class CARLACameraAgent:
    def __init__(self, adapter):
        self.adapter = adapter
    def predict(self, data):
        return AgentPrediction("GO", 0.1)

class CARLALiDARAgent:
    def __init__(self, adapter):
        self.adapter = adapter
    def predict(self, data):
        return AgentPrediction("BRAKE", 0.05)

class CARLAV2VAgent:
    def __init__(self, adapter):
        self.adapter = adapter
    def predict(self, data):
        return AgentPrediction("GO", 0.5)
