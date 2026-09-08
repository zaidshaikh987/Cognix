from .hardware import MotorController

class CognixCar:
    def __init__(self, engine, hardware: MotorController, agents):
        self.engine = engine
        self.hardware = hardware
        self.agents = agents

    def run_cycle(self, sensor_data):
        print("Running cycle...")
        # 1. Get predictions
        for a in self.agents:
            a.update_sensor_data(sensor_data)
            
        # 2. Send to Cognix engine
        try:
            decision = self.engine.decide(self.agents)
        except Exception:
            decision = {"outcome": "ESCALATE"}
            
        # 3 & 4. Translate decision to motor commands & execute safely
        self.decision_to_action(decision)
        return decision

    def decision_to_action(self, decision):
        outcome = getattr(decision.get("outcome", ""), "value", str(decision.get("outcome", "")))
        if outcome in ("ABSTAIN", "ESCALATE"):
            self.hardware.emergency_stop()
            return
            
        action = decision.get("action", "STOP")
        if action == "GO":
            self.hardware.forward(0.5)
        elif action == "BRAKE" or action == "STOP":
            self.hardware.stop()
        elif action == "TURN_LEFT":
            self.hardware.turn_left(0.5)
        elif action == "TURN_RIGHT":
            self.hardware.turn_right(0.5)
        else:
            self.hardware.emergency_stop()
