import json
import csv
import os
from abc import ABC, abstractmethod
from typing import List
from cognix.evaluation.benchmark import ExperimentResult

class BenchmarkReporter(ABC):
    @abstractmethod
    def report(self, results: List[ExperimentResult]) -> None:
        pass

class JSONReporter(BenchmarkReporter):
    def __init__(self, output_path: str):
        self.output_path = output_path
        
    def report(self, results: List[ExperimentResult]) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        data = [r.to_dict() for r in results]
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)

class CSVReporter(BenchmarkReporter):
    def __init__(self, output_path: str):
        self.output_path = output_path
        
    def report(self, results: List[ExperimentResult]) -> None:
        if not results:
            return
            
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Flatten dictionary
        fields = list(results[0].to_dict().keys())
        
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                d = r.to_dict()
                # Ensure complex types are stringified for CSV
                d["failure_conditions"] = json.dumps(d["failure_conditions"])
                d["metadata"] = json.dumps(d["metadata"])
                writer.writerow(d)
