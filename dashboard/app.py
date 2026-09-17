import json
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

app = FastAPI(title="COGNIX Research Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path("results/universal_evaluation")
SUMMARIES_DIR = RESULTS_DIR / "summaries"
LATENCY_DIR = RESULTS_DIR / "latency"
STATS_DIR = RESULTS_DIR / "stats"

def safe_load_json(path: Path):
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

@app.get("/api/metrics")
def get_all_metrics():
    """Serves all aggregated metrics for the dashboard."""
    all_summaries = safe_load_json(SUMMARIES_DIR / "all_summaries.json") or {}
    latency = safe_load_json(LATENCY_DIR / "latency_comparison.json") or {}
    stats = safe_load_json(STATS_DIR / "statistical_tests.json") or {}
    
    # Calculate run status based on raw files if summaries are incomplete
    raw_dir = RESULTS_DIR / "raw"
    total_expected = 20 * 6 * 3  # 20 seeds * 6 scenarios * 3 models
    completed_runs = 0
    scenarios_status = {
        "NORMAL": "PENDING",
        "HIGH_NOISE": "PENDING",
        "MISSING_AGENT": "PENDING",
        "OOD_SHIFT": "PENDING",
        "CONFLICTING": "PENDING",
        "MULTI_FAILURE": "PENDING",
    }
    
    if raw_dir.exists():
        files = list(raw_dir.glob("*.json"))
        completed_runs = len(files)
        for s in scenarios_status.keys():
            s_files = [f for f in files if f.name.startswith(s + "_")]
            if len(s_files) >= 60:  # 20 seeds * 3 models
                scenarios_status[s] = "COMPLETED"
            elif len(s_files) > 0:
                scenarios_status[s] = "RUNNING"
                
    status = {
        "total_expected": total_expected,
        "completed_runs": completed_runs,
        "failed_runs": 0, # Cannot track easily statically without reliability log
        "pending_runs": max(0, total_expected - completed_runs),
        "scenarios": scenarios_status
    }
    
    # Read metadata if exists
    metadata = safe_load_json(RESULTS_DIR / "metadata.json") or {}

    return {
        "summaries": all_summaries,
        "latency": latency,
        "stats": stats,
        "status": status,
        "metadata": metadata,
        "communication": {
            "status": "NOT IMPLEMENTED",
            "type": "THEORETICAL / IN-PROCESS",
            "reduction_vs_standardgat": 0.0,
            "cost_formula": "N × (N-1) × feature_bytes",
            "future_work": "Future Extension: Uncertainty-aware pre-transmission communication pruning (~37% target)"
        }
    }

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    response = FileResponse(os.path.join(STATIC_DIR, "index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  COGNIX Research Analytics Dashboard")
    print("  http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
