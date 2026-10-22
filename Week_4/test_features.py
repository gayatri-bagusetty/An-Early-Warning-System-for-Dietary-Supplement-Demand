"""Leakage and correctness tests for features.py.  Run: python test_features.py   (also works with pytest)"""
import numpy as np, pandas as pd
import features as F

PRODS = ["A", "B"]


def toy(n=48, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=n, freq="MS")
    rows = []
    for p in PRODS:
        for i, m in enumerate(months):
            rows.append(dict(**{"Product Name": p}, month=m, units=float(rng.integers(500, 1500)), n_records=int(rng.integers(4, 12)),
                             price=float(rng.uniform(20, 40)), discount=float(rng.choice([0, .1, .2])), search_index=float(rng.uniform(20, 80)),
                             review_count=float(rng.integers(5, 30)), avg_rating=float(rng.uniform(3.5, 5))))
    panel = pd.DataFrame(rows)
    beh = panel[["Product Name", "month"]].copy()
    beh["return_rate"] = 0.03; beh["discount_share"] = 0.3; beh["platform_hhi"] = 0.34; beh["location_hhi"] = 0.4
    return panel, beh


ORIGIN_COLS = ["lu_o", "lu_roll3_o", "lu_mom3_o", "search_o", "search_o1", "search_o2", "search_anom_o", "search_mom3_o", "search_roll3_o",
               "return_rate_o", "rev_count_o", "rating_o"]


def test_origin_features_ignore_the_future():
    panel, beh = toy(); train_end = "2022-12-01"
    h = 2
    base = F.build_features(panel, beh, h, train_end)
    changed = panel.copy()
    fut = changed["month"] > train_end
    for c in ["units", "search_index", "review_count", "avg_rating", "price"]:
        changed.loc[fut, c] = changed.loc[fut, c] * 3.0          # wildly different future
    new = F.build_features(changed, beh, h, train_end)
    ok = base["month"] <= pd.Timestamp(train_end) + pd.DateOffset(months=h)    # target months whose origin is still <= train_end
    for c in ORIGIN_COLS + ["lu_yoy_t"]:
        a, b = base.loc[ok, c], new.loc[ok, c]
        assert np.allclose(a.fillna(-1), b.fillna(-1)), f"{c} changed when only future months changed"


def test_learned_statistics_use_training_months_only():
    panel, beh = toy(); train_end = "2022-12-01"
    base = F.build_features(panel, beh, 1, train_end)
    changed = panel.copy(); fut = changed["month"] > train_end
    changed.loc[fut, "units"] = changed.loc[fut, "units"] * 10.0
    changed.loc[fut, "search_index"] = 99.0
    new = F.build_features(changed, beh, 1, train_end)
    assert np.allclose(base["season_idx"], new["season_idx"])          # seasonal index must not see test months
    assert np.allclose(base.loc[base["month"] <= train_end, "search_anom"], new.loc[new["month"] <= train_end, "search_anom"])


def test_shifts_do_not_cross_products():
    panel, beh = toy()
    f = F.build_features(panel, beh, 2, "2022-12-01")
    first = f.groupby("Product Name").head(2)                        # first two months of each product have no origin value
    assert first["lu_o"].isna().all()


def test_cyclical_month_puts_december_next_to_january():
    def pt(m): return np.array([np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])
    assert np.linalg.norm(pt(12) - pt(1)) < 0.6                      # 0.52
    assert np.linalg.norm(pt(12) - pt(6)) > 1.9                      # 2.0, opposite sides of the year


def test_product_by_season_interaction_columns():
    panel, beh = toy()
    f = F.build_features(panel, beh, 1, "2022-12-01")
    X = F.design_matrix(f, ["memory", "calendar"], PRODS)
    assert np.allclose(X["prod_A_x_sin"], (f["Product Name"] == "A") * f["month_sin"])
    assert (X[["prod_A", "prod_B"]].sum(axis=1) == 1).all()           # one-hot: exactly one product per row


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
