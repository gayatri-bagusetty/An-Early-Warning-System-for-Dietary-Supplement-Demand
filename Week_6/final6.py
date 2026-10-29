"""Week 6 - final evaluation of the frozen forecasting and alert models.

Frozen choices from Week 5 (not changed here):  forecast = Ridge regression on sales memory + calendar/season + trend (pooled);
alert = logistic regression.  Everything below is computed from the Week 5 rolling-origin predictions (28 monthly origins,
Dec 2022 - Mar 2025) except the feature-impact study and the shock test, which refit the ridge model at every origin.

Sections: 1 headline metrics | 2 slices | 3 prediction intervals | 4 inventory simulation | 5 surge alert | 6 feature impact | 7 shock stress test
"""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import features as F
import metrics5 as M
warnings.filterwarnings("ignore")

R = pd.read_csv("processed/predictions_rolling.csv", parse_dates=["origin", "month"])
C = pd.read_csv("processed/predictions_surge.csv", parse_dates=["origin", "month"])
panel = pd.read_csv("../Week_3/processed/panel_monthly.csv", parse_dates=["month"])
weekly = pd.read_csv("../Week_3/processed/sales_clean_weekly.csv", parse_dates=["Date"])
beh = F.monthly_behaviour(weekly)
PRODUCTS = sorted(panel["Product Name"].unique())
FINAL, LAST, SNAIVE = "Ridge (F2 + trend)", "Last value", "Seasonal naive"
OUT = {}
hit = lambda a, p, tol: float(np.mean(np.abs(p - a) / a <= tol))

# ------------------------------------------------------------------ 1. headline metrics
SHOW = [LAST, SNAIVE, "Seasonal profile", "Ridge (F2)", FINAL, "Random forest (F2)", "Gradient boosting (F2)"]
OUT["headline"] = {}
for n in SHOW:
    OUT["headline"][n] = {}
    for h in (1, 2, 3):
        d = R[(R.model == n) & (R.h == h)]
        s = M.summarize(d.true, d.pred, d.n_records)
        s.update(hit5=hit(d.true, d.pred, .05), hit10=hit(d.true, d.pred, .10), hit15=hit(d.true, d.pred, .15),
                 p90_ape=float(np.percentile(d.ape, 90) * 100), max_ape=float(d.ape.max() * 100))
        OUT["headline"][n][f"h{h}"] = s
OUT["mean_units_per_product_month"] = float((R[(R.model == FINAL) & (R.h == 2)].true * R[(R.model == FINAL) & (R.h == 2)].n_records).mean())

# ------------------------------------------------------------------ 2. slices (h = 2)
d2 = {n: R[(R.model == n) & (R.h == 2)].copy() for n in [LAST, SNAIVE, FINAL]}
for n in d2:
    d2[n]["period"] = d2[n].month.dt.year.astype(str) + " H" + ((d2[n].month.dt.month > 6) + 1).astype(str)
    d2[n]["quarter"] = "Q" + d2[n].month.dt.quarter.astype(str)
OUT["by_period"] = {n: {k: M.mape(g.true, g.pred) for k, g in d2[n].groupby("period")} for n in d2}
OUT["by_quarter"] = {n: {k: M.mape(g.true, g.pred) for k, g in d2[n].groupby("quarter")} for n in d2}
OUT["bias_by_quarter"] = {k: M.bias_pct(g.true, g.pred) for k, g in d2[FINAL].groupby("quarter")}
prod = {}
for p, g in d2[FINAL].groupby("product"):
    gl = d2[LAST][d2[LAST]["product"] == p]; gs = d2[SNAIVE][d2[SNAIVE]["product"] == p]
    s = M.summarize(g.true, g.pred, g.n_records)
    prod[p] = {"MAPE": s["MAPE"], "MAE_units": s["MAE_units"], "RMSE_units": s["RMSE_units"], "bias": s["bias"], "hit10": hit(g.true, g.pred, .10),
               "mean_units": float((g.true * g.n_records).mean()), "MAPE_last": M.mape(gl.true, gl.pred), "MAPE_snaive": M.mape(gs.true, gs.pred), "n": s["n"]}
OUT["by_product"] = prod
# seasonal profile of the demand rate per product (month average / product average), all 64 months
pr = pd.read_csv("../Week_3/processed/panel_monthly.csv", parse_dates=["month"]); pr["rate"] = pr["units"] / pr["n_records"]
pr["moy"] = pr.month.dt.month; pr["rel"] = pr.rate / pr.groupby("Product Name")["rate"].transform("mean")
si = pr.groupby(["Product Name", "moy"])["rel"].mean().unstack()
OUT["season_profile"] = {p: {"peak_month": int(si.loc[p].idxmax()), "peak": float(si.loc[p].max()), "trough_month": int(si.loc[p].idxmin()), "trough": float(si.loc[p].min())} for p in si.index}
# worst errors (h = 2): specific cases
w = d2[FINAL].assign(err=lambda d: (d.pred - d.true) / d.true).sort_values("ape", ascending=False).head(8)
OUT["worst_errors"] = [{"product": r["product"], "month": str(r["month"].date())[:7], "actual_units": float(r.true * r.n_records), "forecast_units": float(r.pred * r.n_records), "err_pct": float(r.err * 100), "n_records": int(r.n_records)} for _, r in w.iterrows()]

# ------------------------------------------------------------------ 3. prediction intervals (empirical, expanding window, 80%)
def intervals(model, h, q=(0.10, 0.90), min_n=40):
    d = R[(R.model == model) & (R.h == h)].sort_values(["origin", "product"]).copy()
    d["lr"] = np.log(d.true / d.pred); d["lo"] = np.nan; d["hi"] = np.nan
    for o in d.origin.unique():
        past = d[d.month <= o]                       # errors already realised when the forecast is made
        if len(past) >= min_n:
            ql, qh = np.quantile(past.lr, q); m = d.origin == o
            d.loc[m, "lo"] = d.loc[m, "pred"] * np.exp(ql); d.loc[m, "hi"] = d.loc[m, "pred"] * np.exp(qh)
    return d
OUT["intervals"] = {}
IV = {}
for h in (1, 2, 3):
    d = intervals(FINAL, h); IV[h] = d; e = d.dropna(subset=["lo"])
    ql, qh = np.quantile(np.log(d.true / d.pred), [0.10, 0.90])
    OUT["intervals"][f"h{h}"] = {"coverage": float(((e.true >= e.lo) & (e.true <= e.hi)).mean()), "n": int(len(e)), "n_all": int(len(d)),
                                  "band_low_pct": float((np.exp(ql) - 1) * 100), "band_high_pct": float((np.exp(qh) - 1) * 100),
                                  "below": float((e.true < e.lo).mean()), "above": float((e.true > e.hi).mean())}
IV[2].to_csv("processed/intervals_final_h2.csv", index=False)

# ------------------------------------------------------------------ 4. inventory simulation (h = 2, order = forecast x buffer for a 90% in-stock target)
price = float(panel["price"].mean()); MARGIN, HOLD = 0.40, 0.05          # assumptions (stated in the report): 40% margin, holding/waste cost 5% of price per excess unit
def inventory(model, svc=0.90, h=2, min_n=40):
    d = R[(R.model == model) & (R.h == h)].sort_values(["origin", "product"]).copy(); d["ratio"] = d.true / d.pred; rows = []
    for o, g in d.groupby("origin"):
        past = d[d.month <= o]
        if len(past) < min_n:
            continue
        m = float(np.quantile(past.ratio, svc)); order = g.pred * m * g.n_records; dem = g.true * g.n_records
        rows.append(pd.DataFrame({"short": np.maximum(dem - order, 0), "excess": np.maximum(order - dem, 0), "dem": dem, "buf": m - 1, "ok": (order >= dem).astype(float)}))
    r = pd.concat(rows)
    return {"in_stock": float(r.ok.mean()), "buffer_pct": float(r.buf.mean() * 100), "excess_units": float(r.excess.mean()), "excess_pct_of_demand": float(r.excess.sum() / r.dem.sum() * 100),
            "short_units": float(r.short.mean()), "n": int(len(r)), "mean_demand": float(r.dem.mean()),
            "cost_per_pm": float(MARGIN * price * r.short.mean() + HOLD * price * r.excess.mean())}
OUT["inventory"] = {n: inventory(n) for n in [LAST, SNAIVE, FINAL]}
OUT["inventory_assumptions"] = {"price": price, "margin": MARGIN, "hold_pct_of_price": HOLD, "service_target": 0.90}

# ------------------------------------------------------------------ 5. surge alert evaluation
LRN = "Logistic regression"
clsn = [LRN, "Decision tree", "Random forest", "Gradient boosting", "Seasonal rule (surge a year ago)", "Persistence (last month a surge)", "Predict never"]
OUT["alert"] = {}
for n in clsn:
    OUT["alert"][n] = {}
    for h in (1, 2, 3):
        d = C[(C.model == n) & (C.h == h)]
        s = M.classification_scores(d.y, d.pred, d.score if n != "Predict never" else None)
        s["accuracy"] = float((d.y == d.pred).mean()); s["n"] = int(len(d))
        if n == LRN:
            s["brier"] = float(brier_score_loss(d.y, d.score)); s["brier_never"] = float(brier_score_loss(d.y, np.full(len(d), d.y.mean())))
        OUT["alert"][n][f"h{h}"] = s
lr2 = C[(C.model == LRN) & (C.h == 2)].copy()
bins = [0, .1, .3, .5, .7, .9, 1.0001]; lr2["bin"] = pd.cut(lr2.score, bins, right=False)
OUT["calibration_h2"] = [{"bin": str(k), "n": int(len(g)), "mean_pred": float(g.score.mean()), "observed": float(g.y.mean())} for k, g in lr2.groupby("bin", observed=True)]
cost = {}
for ratio in (1, 3, 5):
    best = None; curve = []
    for t in np.round(np.linspace(0.05, 0.95, 19), 2):
        pred = (lr2.score >= t).astype(int); fn = int(((lr2.y == 1) & (pred == 0)).sum()); fp = int(((lr2.y == 0) & (pred == 1)).sum()); c = ratio * fn + fp
        curve.append((float(t), c, fn, fp)); best = (float(t), c, fn, fp) if best is None or c < best[1] else best
    at50 = [x for x in curve if abs(x[0] - 0.5) < 1e-9][0]
    cost[f"ratio{ratio}"] = {"curve": curve, "best_t": best[0], "best_cost": best[1], "best_fn": best[2], "best_fp": best[3], "cost_at_05": at50[1], "never": int(ratio * (lr2.y == 1).sum())}
OUT["cost_curve"] = cost
pp = {}
for p, g in lr2.groupby("product"):
    s = M.classification_scores(g.y, g.pred); pp[p] = {k: s[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall")}
OUT["alert_by_product"] = pp
fm = R[(R.model == FINAL) & (R.h == 2)][["month", "product", "true", "pred", "n_records"]]
ex = lr2.merge(fm, on=["month", "product"], how="left")
def pick(df, n): return [{"product": r["product"], "month": str(r["month"].date())[:7], "prob": float(r.score), "actual_rate": float(r.true), "forecast_rate": float(r.pred_y)} for _, r in df.head(n).iterrows()]
OUT["alert_examples"] = {"caught": pick(ex[(ex.y == 1) & (ex.pred_x == 1)].sort_values("score", ascending=False), 4), "false_alarm": pick(ex[(ex.y == 0) & (ex.pred_x == 1)].sort_values("score", ascending=False), 3),
                         "missed": pick(ex[(ex.y == 1) & (ex.pred_x == 0)].sort_values("score"), 3)} if "pred_x" in ex else {}
lr2.to_csv("processed/alert_lr_h2.csv", index=False)

# ------------------------------------------------------------------ 6. feature-by-feature impact (rolling-origin refits of the final ridge model)
F.GROUPS["trend"] = ["t_idx"]
NEED = [c for g_ in ["memory", "calendar", "search_raw", "search_eng", "price_promo", "behaviour"] for c in F.GROUPS[g_]] + ["lu_mean12_o"]
ORIGINS = list(pd.date_range("2022-12-01", "2025-03-01", freq="MS")); LAST_M = panel.month.max()
def frame(pn, o, h):
    f = F.build_features(pn, beh, h, o.strftime("%Y-%m-%d"))
    f["lu_mean12_o"] = f.groupby("Product Name")["lu"].transform(lambda s: s.rolling(12).mean().shift(h))
    f["t_idx"] = (f["month"].dt.year - 2020) * 12 + f["month"].dt.month
    return f
def split(f, o, h):
    ok = f[NEED].notna().all(axis=1)
    return f[(f.month <= o) & ok].sort_values("month"), f[(f.month == o + pd.DateOffset(months=h)) & ok]
def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); m = RidgeCV(alphas=[0.1, 1, 10, 100, 300], cv=TimeSeriesSplit(4), scoring="neg_mean_squared_error").fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))
BASE = ["memory", "calendar", "trend"]
prodcols = lambda cols: [c for c in cols if c.startswith("prod_") and not (c.endswith("_x_sin") or c.endswith("_x_cos"))]
DROP = {"Same month last year": lambda cols: ["lu_yoy_t"], "Last month's level": lambda cols: ["lu_o"], "3-month average level": lambda cols: ["lu_roll3_o"],
        "3-month momentum": lambda cols: ["lu_mom3_o"], "Learned season index": lambda cols: ["season_idx"],
        "Seasonal curve (month sin/cos, by product)": lambda cols: ["month_sin", "month_cos"] + [c for c in cols if c.endswith("_x_sin") or c.endswith("_x_cos")],
        "Holiday flags (Jan, summer, Nov-Dec)": lambda cols: ["flag_newyear", "flag_summer", "flag_gifting"], "Product identity": prodcols, "Trend (time index)": lambda cols: ["t_idx"], "ALL sales-memory features (4)": lambda cols: ["lu_o", "lu_roll3_o", "lu_mom3_o", "lu_yoy_t"], "ALL season features (curve, index, flags)": lambda cols: ["month_sin", "month_cos", "flag_newyear", "flag_summer", "flag_gifting", "season_idx"] + [c for c in cols if c.endswith("_x_sin") or c.endswith("_x_cos")], "Memory and season together": lambda cols: ["lu_o", "lu_roll3_o", "lu_mom3_o", "lu_yoy_t"] + ["month_sin", "month_cos", "flag_newyear", "flag_summer", "flag_gifting", "season_idx"] + [c for c in cols if c.endswith("_x_sin") or c.endswith("_x_cos")]}
ADD = ["search_o", "search_o1", "search_o2", "search_anom_o", "search_mom3_o", "search_roll3_o", "price_idx_t", "discount_t", "disc_x_searchanom", "return_rate_o", "platform_hhi_o", "discount_share_o"]
store = {}   # (variant, h) -> list of (origin, product, month, true, pred, n_records)
for h in (1, 2, 3):
    for o in ORIGINS:
        if o + pd.DateOffset(months=h) > LAST_M:
            continue
        f = frame(panel, o, h); tr, te = split(f, o, h)
        if len(te) == 0:
            continue
        Xtr0, Xte0 = F.design_matrix(tr, BASE, PRODUCTS), F.design_matrix(te, BASE, PRODUCTS); cols = list(Xtr0.columns)
        info = pd.DataFrame({"origin": o, "product": te["Product Name"].values, "month": te["month"].values, "true": np.exp(te["y"].values), "n": te["n_records"].values})
        def save(name, p): store.setdefault((name, h), []).append(info.assign(pred=np.exp(p)))
        save("BASE", ridge_pred(Xtr0.values, tr["y"].values, Xte0.values))
        for name, fn in DROP.items():
            dc = [c for c in fn(cols) if c in cols]
            save("drop|" + name, ridge_pred(Xtr0.drop(columns=dc).values, tr["y"].values, Xte0.drop(columns=dc).values))
        for c_ in ADD:
            a = Xtr0.assign(**{c_: tr[c_].values}); b = Xte0.assign(**{c_: te[c_].values})
            save("add|" + c_, ridge_pred(a.values, tr["y"].values, b.values))
impact = {"drop": {}, "add": {}}
for (name, h), parts in store.items():
    if name == "BASE":
        continue
    v = pd.concat(parts, ignore_index=True).sort_values(["origin", "product"]).reset_index(drop=True)
    b = pd.concat(store[("BASE", h)], ignore_index=True).sort_values(["origin", "product"]).reset_index(drop=True)
    ea, eb = np.abs(v.pred - v.true) / v.true, np.abs(b.pred - b.true) / b.true
    mean, lo, hi = M.block_bootstrap_diff(ea.values, eb.values, v.month.astype(str).values, n_boot=800)
    kind, label = name.split("|")
    impact[kind].setdefault(label, {})[f"h{h}"] = {"mape": M.mape(v.true, v.pred), "d_mape": mean * 100, "lo": lo * 100, "hi": hi * 100,
        "d_rmse": M.rmse(v.true * v.n, v.pred * v.n) - M.rmse(b.true * b.n, b.pred * b.n), "p": float(stats.wilcoxon((ea - eb).values).pvalue)}
for h in (1, 2, 3):
    b = pd.concat(store[("BASE", h)], ignore_index=True)
    impact.setdefault("base", {})[f"h{h}"] = {"mape": M.mape(b.true, b.pred), "rmse": M.rmse(b.true * b.n, b.pred * b.n), "n": int(len(b))}
OUT["impact"] = impact
# which feature does the season index make redundant?  correlation of season_idx with the same-month-last-year value at the last origin
f = frame(panel, pd.Timestamp("2024-12-01"), 2); tr_, _ = split(f, pd.Timestamp("2024-12-01"), 2)
OUT["corr_season_vs_yoy"] = float(np.corrcoef(tr_["season_idx"], tr_["lu_yoy_t"] - tr_.groupby("Product Name")["lu_yoy_t"].transform("mean"))[0, 1])

# ------------------------------------------------------------------ 7. shock stress test: a +60% surge in Ashwagandha for Jan-Mar 2024 that no feature can anticipate
SH = "Ashwagandha"; SHOCK = pd.date_range("2024-01-01", "2024-03-01", freq="MS"); MULT = 1.6
pn = panel.copy(); mask = (pn["Product Name"] == SH) & pn.month.isin(SHOCK); pn.loc[mask, "units"] *= MULT
ql, qh = np.quantile(np.log(IV[2].true / IV[2].pred), [0.10, 0.90]); rows = []
for t in pd.date_range("2024-01-01", "2024-07-01", freq="MS"):
    o = t - pd.DateOffset(months=2); f = frame(pn, o, 2); tr, te = split(f, o, 2); te = te[te["Product Name"] == SH]
    Xtr, Xte = F.design_matrix(tr, BASE, PRODUCTS), F.design_matrix(te, BASE, PRODUCTS)
    pr_ = float(np.exp(ridge_pred(Xtr.values, tr["y"].values, Xte.values))[0]); act = float(np.exp(te["y"].values[0]))
    last = float(np.exp(te["lu_o"].values[0])); sn = float(np.exp(te["lu_yoy_t"].values[0]))
    clean = R[(R.model == FINAL) & (R.h == 2) & (R["product"] == SH) & (R.month == t)]
    rows.append({"month": str(t.date())[:7], "in_shock": bool(t in SHOCK), "actual": act, "ridge": pr_, "last": last, "snaive": sn, "ape_ridge": abs(pr_ - act) / act * 100, "ape_last": abs(last - act) / act * 100,
                 "outside_band": bool(act > pr_ * np.exp(qh) or act < pr_ * np.exp(ql)), "clean_ridge_ape": float(clean.ape.iloc[0] * 100) if len(clean) else None, "n_records": int(te["n_records"].values[0])})
OUT["stress"] = {"product": SH, "multiplier": MULT, "months": [str(m.date())[:7] for m in SHOCK], "rows": rows}
json.dump(OUT, open("processed/final6_results.json", "w"), indent=1, default=str)
print("saved")
h = OUT["headline"]; print({n: (round(h[n]["h2"]["MAPE"], 2), round(h[n]["h2"]["hit10"], 3)) for n in h})
print("intervals", {k: (round(v["coverage"], 3), round(v["band_low_pct"], 1), round(v["band_high_pct"], 1)) for k, v in OUT["intervals"].items()})
print("inventory", {k: {a: round(b, 2) for a, b in v.items()} for k, v in OUT["inventory"].items()})
print("alert LR", {k: (round(v["accuracy"], 3), round(v["precision"], 2), round(v["recall"], 2), round(v["F1"], 2), round(v.get("ROC_AUC", 0), 3)) for k, v in OUT["alert"][LRN].items()}, "never acc", round(OUT["alert"]["Predict never"]["h2"]["accuracy"], 3))
print("cost", {k: (v["best_t"], v["best_cost"], v["cost_at_05"], v["never"]) for k, v in cost.items()})
for kind in ("drop", "add"):
    print(kind)
    for k, v in impact[kind].items(): print(f"  {k:44s}", " ".join(f"h{h}: {v[f'h{h}']['d_mape']:+.2f} [{v[f'h{h}']['lo']:+.2f},{v[f'h{h}']['hi']:+.2f}]" for h in (1, 2, 3)))
print("base", impact["base"])
for r in rows: print(r)
