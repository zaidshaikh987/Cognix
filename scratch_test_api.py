import requests
import time

print("Setting scenario to CAMERA_BLACKOUT...")
r = requests.post("http://localhost:8000/api/set_scenario", json={"scenario": "CAMERA_BLACKOUT"})
print("Response:", r.status_code, r.text)

print("Fetching latest payload...")
for i in range(3):
    r = requests.get("http://localhost:8000/api/latest")
    data = r.json()
    print(f"Tick: {data['tick']} | Scenario: {data['scenario']}")
    # print agent epistemic
    for a in data["agents"]:
        print(f"  Agent: {a['name']}, Epi: {a['epistemic']:.4f}, Weight: {a['weight']:.4f}")
    time.sleep(1)
