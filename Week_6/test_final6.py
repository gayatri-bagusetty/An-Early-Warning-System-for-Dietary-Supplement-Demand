"""Sanity checks on Week 6 results (run: python test_final6.py)."""
import json, math
D = json.load(open("processed/final6_results.json"))
H = D["headline"]; FM = "Ridge (F2 + trend)"
def test_final_beats_baselines():
    for h in ("h1", "h2", "h3"):
        assert H[FM][h]["MAPE"] < H["Seasonal naive"][h]["MAPE"] < H["Last value"][h]["MAPE"]
def test_counts():
    a = D["alert"]["Logistic regression"]["h2"]
    assert a["tp"] + a["fp"] + a["fn"] + a["tn"] == a["n"] == 216
    assert math.isclose(a["recall"], a["tp"] / (a["tp"] + a["fn"]))
def test_band_coverage_reasonable():
    assert 0.70 <= D["intervals"]["h2"]["coverage"] <= 0.90
def test_inventory_order():
    i = D["inventory"]; assert i[FM]["cost_per_pm"] < i["Seasonal naive"]["cost_per_pm"] < i["Last value"]["cost_per_pm"]
if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f()
    print("4 tests passed")
