"""Week 4 - Do the engineered features actually help forecasts?  Time-based evaluation.

For each forecast origin (a training cut-off) and horizon h = 1, 2, 3 months, fit Ridge regression on
months <= origin and score on later months. Two baselines are included:
  seasonal naive = units of the same month last year        last value = units at the origin month ("rearview mirror")
Target = log(units per record), i.e. demand per listing, see features.py.
Everything learned from data (scaler, seasonal indices, ridge strength) uses training months only.
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import features as F

panel = pd.read_csv("../Week_3/processed/panel_monthly.csv", parse_dates=["month"])
weekly = pd.read_csv("../Week_3/processed/sales_clean_weekly.csv", parse_dates=["Date"])
beh = F.monthly_behaviour(weekly)
PRODUCTS = sorted(panel["Product Name"].unique())

SETS = {
    "S1 sales memory": ["memory"],
    "S2 + calendar and season": ["memory", "calendar"],
    "S3 + raw search lags": ["memory", "calendar", "search_raw"],
    "S4 + engineered search": ["memory", "calendar", "search_eng"],
    "S5 + raw and engineered search": ["memory", "calendar", "search_raw", "search_eng"],
    "S6 + price and promotion": ["memory", "calendar", "search_eng", "price_promo"],
    "S7 + behaviour and channel": ["memory", "calendar", "search_eng", "price_promo", "behaviour"],
}
ORIGINS = {"2022-12-01": ("2023-01-01", "2023-12-01"), "2023-12-01": ("2024-01-01", "2025-04-01")}
R = {"sets": {k: v for k, v in SETS.items()}, "runs": {}}


def mape(a, b):
    return float(np.mean(np.abs(a - b) / b) * 100)


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)                                   # fitted on training rows only
    order = np.argsort(np.arange(len(Xtr)))
    m = RidgeCV(alphas=[0.1, 1, 10, 100, 300], cv=TimeSeriesSplit(4), scoring="neg_mean_squared_error")
    m.fit(sc.transform(Xtr), ytr)
    return m, sc, m.predict(sc.transform(Xte))


def run(origin, h, groups_list, need_cols, test_months, extra_filter=None):
    f = F.build_features(panel, beh, h, origin)
    ok = f[need_cols].notna().all(axis=1)
    if extra_filter is not None:
        ok &= extra_filter(f)
    tr = f[(f["month"] <= origin) & ok].sort_values("month")          # time order for TimeSeriesSplit
    te = f[(f["month"] >= test_months[0]) & (f["month"] <= test_months[1]) & ok]
    return f, tr, te


allcols = [c for g in ["memory", "calendar", "search_raw", "search_eng", "price_promo", "behaviour"] for c in F.GROUPS[g]]
paired = {}
for origin, tmonths in ORIGINS.items():
    for h in (1, 2, 3):
        f, tr, te = run(origin, h, SETS, allcols, tmonths)
        yte = np.exp(te["y"].values)
        res = {"n_train": int(len(tr)), "n_test": int(len(te))}
        res["seasonal_naive"] = mape(np.exp(te["lu_yoy_t"].values), yte)
        res["last_value"] = mape(np.exp(te["lu_o"].values), yte)
        preds = {}
        for name, groups in SETS.items():
            Xtr = F.design_matrix(tr, groups, PRODUCTS); Xte = F.design_matrix(te, groups, PRODUCTS)
            m, sc, yh = fit_predict(Xtr.values, tr["y"].values, Xte.values)
            preds[name] = np.exp(yh); res[name] = mape(preds[name], yte); res[name + " alpha"] = float(m.alpha_)
        # paired comparison of absolute percentage errors against S2 (does search / price / behaviour add anything?)
        ape = {k: np.abs(v - yte) / yte for k, v in preds.items()}
        for name in SETS:
            if name in ("S1 sales memory", "S2 + calendar and season"):
                continue
            d = ape[name] - ape["S2 + calendar and season"]
            w = stats.wilcoxon(d) if np.any(d != 0) else None
            res[name + " vs S2"] = {"mean_diff_pp": float(d.mean() * 100), "wilcoxon_p": float(w.pvalue) if w else None,
                                    "share_rows_better": float((d < 0).mean())}
        d = ape["S2 + calendar and season"] - ape["S1 sales memory"]
        res["S2 vs S1"] = {"mean_diff_pp": float(d.mean() * 100), "wilcoxon_p": float(stats.wilcoxon(d).pvalue), "share_rows_better": float((d < 0).mean())}
        # per product MAPE (S1 vs S2)
        pp = {}
        for pr in PRODUCTS:
            msk = (te["Product Name"] == pr).values
            pp[pr] = {"S1": mape(preds["S1 sales memory"][msk], yte[msk]), "S2": mape(preds["S2 + calendar and season"][msk], yte[msk]),
                      "last_value": mape(np.exp(te["lu_o"].values[msk]), yte[msk])}
        res["per_product"] = pp
        # diagnostic: does search anomaly explain what S2 leaves unexplained?
        Xtr2 = F.design_matrix(tr, SETS["S2 + calendar and season"], PRODUCTS); Xte2 = F.design_matrix(te, SETS["S2 + calendar and season"], PRODUCTS)
        m2, sc2, yh2 = fit_predict(Xtr2.values, tr["y"].values, Xte2.values)
        rtr = tr["y"].values - m2.predict(sc2.transform(Xtr2.values)); rte = te["y"].values - yh2
        res["resid_corr_searchanom_train"] = float(np.corrcoef(tr["search_anom_o"], rtr)[0, 1])
        res["resid_corr_searchanom_test"] = float(np.corrcoef(te["search_anom_o"], rte)[0, 1])
        res["resid_corr_searchraw_test"] = float(np.corrcoef(te["search_o"], rte)[0, 1])
        if origin == "2023-12-01" and h == 2:
            Xs = F.design_matrix(tr, SETS["S4 + engineered search"], PRODUCTS)
            m4, sc4, _ = fit_predict(Xs.values, tr["y"].values, F.design_matrix(te, SETS["S4 + engineered search"], PRODUCTS).values)
            coefs = pd.Series(m4.coef_, index=Xs.columns)
            nonprod = coefs[[c for c in coefs.index if not c.startswith("prod_")]]
            keep = te[["Product Name", "month", "n_records", "y"]].copy()
            keep["true_rate"] = np.exp(keep.pop("y")); keep["last_value"] = np.exp(te["lu_o"].values)
            keep["S1"] = preds["S1 sales memory"]; keep["S2"] = preds["S2 + calendar and season"]; keep["S4"] = preds["S4 + engineered search"]
            keep.to_csv("processed/forecasts_h2_origin2023-12.csv", index=False)
            res["top_coef_S4"] = nonprod.reindex(nonprod.abs().sort_values(ascending=False).index).head(8).round(4).to_dict()
        R["runs"][f"{origin}|h{h}"] = res

# ---- H4 (exploratory): do review features add anything?  Only months covered by the review source
revcols = allcols[:]  # same rows as above, plus reviews must be known
H4 = {}
for h in (1, 2, 3):
    f, tr, te = run("2022-12-01", h, SETS, allcols + F.GROUPS["reviews"], ("2023-01-01", "2023-09-01"))
    yte = np.exp(te["y"].values)
    base = SETS["S4 + engineered search"]; withrev = base + ["reviews"]
    outs = {}
    for name, groups in [("S4", base), ("S4 + reviews", withrev)]:
        Xtr = F.design_matrix(tr, groups, PRODUCTS); Xte = F.design_matrix(te, groups, PRODUCTS)
        m, sc, yh = fit_predict(Xtr.values, tr["y"].values, Xte.values); outs[name] = np.exp(yh)
    d = np.abs(outs["S4 + reviews"] - yte) / yte - np.abs(outs["S4"] - yte) / yte
    H4[f"h{h}"] = {"n_train": int(len(tr)), "n_test": int(len(te)), "S4": mape(outs["S4"], yte), "S4 + reviews": mape(outs["S4 + reviews"], yte),
                   "mean_diff_pp": float(d.mean() * 100), "wilcoxon_p": float(stats.wilcoxon(d).pvalue)}
R["H4_reviews"] = H4

json.dump(R, open("processed/model_compare.json", "w"), indent=1, default=str)
for k, v in R["runs"].items():
    print(k, {a: round(b, 2) for a, b in v.items() if isinstance(b, float) and "alpha" not in a and "resid" not in a}, "n", v["n_train"], v["n_test"])
print("H4", {k: (round(v["S4"], 2), round(v["S4 + reviews"], 2), round(v["wilcoxon_p"], 3)) for k, v in H4.items()})
