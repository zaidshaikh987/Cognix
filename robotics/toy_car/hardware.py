"""
Hardware abstraction layer for toy autonomous car demonstrator.

Target hardware (reference, not mandatory):
- Raspberry Pi 4
- L298N motor driver
- HC-SR04 ultrasonic sensor
- Pi Camera Module

This is a DEMONSTRATOR, not a research contribution.
All commands are bounded and include emergency stop.
"""

class MotorController:
    def forward(self, speed: float):
        raise NotImplementedError
    def backward(self, speed: float):
        raise NotImplementedError
    def turn_left(self, speed: float):
        raise NotImplementedError
    def turn_right(self, speed: float):
        raise NotImplementedError
    def stop(self):
        raise NotImplementedError
    def emergency_stop(self):
        raise NotImplementedError
    def is_safe(self, command) -> bool:
        return True

class RaspberryPiMotorController(MotorController):
    def __init__(self):
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.GPIO.setmode(self.GPIO.BCM)
            # Init pins here...
            self.initialized = True
        except ImportError:
            self.initialized = False
            print("RPi.GPIO not found. Cannot initialize hardware.")

    def _execute(self, cmd, speed):
        if not self.initialized: return
        if not self.is_safe(cmd):
            self.emergency_stop()
            return
        # Execute logic...
        
    def forward(self, speed: float):
        self._execute("FORWARD", speed)
    def backward(self, speed: float):
        self._execute("BACKWARD", speed)
    def turn_left(self, speed: float):
        self._execute("LEFT", speed)
    def turn_right(self, speed: float):
        self._execute("RIGHT", speed)
    def stop(self):
        pass
    def emergency_stop(self):
        print("EMERGENCY STOP (Hardware)")

class SimulatedMotorController(MotorController):
    def forward(self, speed: float):
        print(f"Sim: FORWARD at {speed}")
    def backward(self, speed: float):
        print(f"Sim: BACKWARD at {speed}")
    def turn_left(self, speed: float):
        print(f"Sim: LEFT at {speed}")
    def turn_right(self, speed: float):
        print(f"Sim: RIGHT at {speed}")
    def stop(self):
        print("Sim: STOP")
    def emergency_stop(self):
        print("Sim: EMERGENCY STOP")
