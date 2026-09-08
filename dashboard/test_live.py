"""Quick live performance test for the COGNIX dashboard."""
import urllib.request
import json
import time

BASE = "http://localhost:8000"

print("COGNIX Dashboard - Live Performance Test")
print("=" * 72)
print(f"{'Tick':<5} {'Scenario':<24} {'Decision':<20} {'Conf':>6} {'Epi':>6} {'Risk':<10} {'ms':>6}")
print("-" * 72)

for i in range(8):
    try:
        r = urllib.request.urlopen(f"{BASE}/api/latest", timeout=4)
        d = json.loads(r.read().decode())
        print(
            f"{str(d['tick']):<5} "
            f"{d['scenario']:<24} "
            f"{d['decision']:<20} "
            f"{d['confidence']:>6.3f} "
            f"{d['epistemic']:>6.3f} "
            f"{d['risk_level']:<10} "
            f"{d['latency']['total']:>6.2f}"
        )
    except Exception as e:
        print(f"  [ERROR] {e}")
    time.sleep(1.6)

print("=" * 72)

# Summary from history
try:
    r = urllib.request.urlopen(f"{BASE}/api/history", timeout=4)
    hist = json.loads(r.read().decode())
    decisions = [h["decision"] for h in hist]
    risks     = [h["risk_level"] for h in hist]
    confs     = [h["confidence"] for h in hist]
    epis      = [h["epistemic"]  for h in hist]
    lats      = [h["latency"]["total"] for h in hist]

    from collections import Counter
    print(f"\nHistory summary over {len(hist)} ticks:")
    print(f"  Decisions  : {dict(Counter(decisions))}")
    print(f"  Risk levels: {dict(Counter(risks))}")
    print(f"  Avg confidence  : {sum(confs)/len(confs):.3f}")
    print(f"  Avg epistemic   : {sum(epis)/len(epis):.3f}")
    print(f"  Avg latency     : {sum(lats)/len(lats):.2f} ms")
    print(f"  Min/Max latency : {min(lats):.2f} / {max(lats):.2f} ms")
    print(f"\nScenarios seen: {sorted(set(h['scenario'] for h in hist))}")
except Exception as e:
    print(f"  [History ERROR] {e}")

print("\nDashboard is LIVE at http://localhost:8000")
