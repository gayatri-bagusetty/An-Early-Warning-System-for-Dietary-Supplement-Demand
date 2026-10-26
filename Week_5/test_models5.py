"""Week 5 tests: metrics are correct on hand-checked numbers, and the rolling-origin split cannot leak."""
import numpy as np, pandas as pd
import metrics5 as M

def test_metrics_hand_checked():
    a, p = np.array([100., 200.]), np.array([110., 180.])
    assert abs(M.mape(a, p) - 10.0) < 1e-9 and abs(M.mae(a, p) - 15.0) < 1e-9
    assert abs(M.rmse(a, p) - np.sqrt((100 + 400) / 2)) < 1e-9
    assert abs(M.bias_pct(a, p) - 0.0) < 1e-9          # +10% and -10% cancel

def test_classification_counts():
    s = M.classification_scores([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (2, 1, 1, 1)
    assert abs(s["precision"] - 2 / 3) < 1e-9 and abs(s["recall"] - 2 / 3) < 1e-9

def test_partial_f_detects_real_signal_and_ignores_noise():
    ps_signal, ps_noise = [], []
    for seed in range(40):
        rng = np.random.default_rng(seed); n = 200; x = rng.normal(size=n); z = rng.normal(size=n); y = 2 * x + rng.normal(size=n)
        ps_signal.append(M.partial_f_test(y, z[:, None], np.column_stack([z, x]))[3])    # adding the real driver
        ps_noise.append(M.partial_f_test(y, x[:, None], np.column_stack([x, z]))[3])     # adding pure noise
    assert max(ps_signal) < 0.001
    assert 0.3 < np.mean(ps_noise) < 0.7 and np.mean(np.array(ps_noise) < 0.05) < 0.15   # p-values roughly uniform under no effect

def test_rolling_predictions_target_months_are_after_the_origin():
    R = pd.read_csv("processed/predictions_rolling.csv", parse_dates=["origin", "month"])
    gap = (R["month"].dt.year - R["origin"].dt.year) * 12 + (R["month"].dt.month - R["origin"].dt.month)
    assert (gap == R["h"]).all(), "every prediction must be exactly h months after its origin"
    assert R.groupby(["origin", "h", "model"]).size().min() == 8, "8 products per origin and horizon"

def test_each_origin_is_scored_once_per_model():
    R = pd.read_csv("processed/predictions_rolling.csv")
    assert not R.duplicated(["origin", "h", "model", "product"]).any()

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(len(fns), "tests passed")
