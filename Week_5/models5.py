"""Week 5 - Predictive modelling and algorithm selection.

Question (Week 1): can we forecast demand 1 to 3 months ahead better than looking at last month's sales?
This script compares candidate algorithms under ONE validation design so that the comparison is fair:

  rolling-origin (expanding window) evaluation: for every origin month o from Dec 2022 to Mar 2025, fit on months <= o,
  predict month o + h (h = 1, 2, 3) for all 8 products, then move the origin forward one month.
  Anything learned from data (seasonal indices, product means, scalers, clusters, model weights) uses months <= o only.

Tasks
  A. Regression  : forecast the log demand rate (units per record), reported as MAPE, sMAPE, bias, MAE and RMSE in units.
  B. Modelling strategy : one pooled model vs one model per product vs one model per seasonal cluster (k-means).
  C. Classification : early-warning flag "demand rate will be >= 15% above its trailing 12-month average"
                      (precision, recall, F1, PR-AUC).
Hyperparameters are fixed in advance (not tuned on the test months); see MODELS below.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Week_4"))

import features as F
import metrics5 as M

warnings.filterwarnings("ignore", category=UserWarning)
T0 = time.time()
panel = pd.read_csv("../Week_3/processed/panel_monthly.csv", parse_dates=["month"])
weekly = pd.read_csv("../Week_3/processed/sales_clean_weekly.csv", parse_dates=["Date"])
beh = F.monthly_behaviour(weekly)
PRODUCTS = sorted(panel["Product Name"].unique())
LAST_MONTH = panel["month"].max()                                          # Apr 2025
ORIGINS = list(pd.date_range("2022-12-01", "2025-03-01", freq="MS"))        # 28 monthly origins
HORIZONS = (1, 2, 3)
SURGE = float(np.log(1.15))                                                # +15% over trailing 12-month average

F.GROUPS["trend"] = ["t_idx"]                                                # linear time index (months since Jan 2020), known for any future month
F2 = ["memory", "calendar"]                                                # sales memory + calendar and season (Week 4 winner)
F7 = ["memory", "calendar", "search_eng", "price_promo", "behaviour"]      # everything Week 4 built, except raw search and reviews
NEED = [c for g in ["memory", "calendar", "search_raw", "search_eng", "price_promo", "behaviour"] for c in F.GROUPS[g]] + ["lu_mean12_o"]


def frame(origin, h):
    """Feature frame for forecast horizon h with all learned statistics fitted on months <= origin."""
    f = F.build_features(panel, beh, h, origin.strftime("%Y-%m-%d"))
    f["lu_mean12_o"] = f.groupby("Product Name")["lu"].transform(lambda s: s.rolling(12).mean().shift(h))
    f["t_idx"] = (f["month"].dt.year - 2020) * 12 + f["month"].dt.month
    f["lu_dev12_o"] = f["lu_o"] - f["lu_mean12_o"]
    f["surge"] = (f["lu"] - f["lu_mean12_o"] >= SURGE).astype(float)
    return f


def split(f, origin, h):
    ok = f[NEED].notna().all(axis=1)
    tr = f[(f["month"] <= origin) & ok].sort_values("month")
    target = origin + pd.DateOffset(months=h)
    te = f[(f["month"] == target) & ok]
    return tr, te


def lin_matrix(df, groups, extra=()):
    X = F.design_matrix(df, groups, PRODUCTS)
    for c in extra:
        X[c] = df[c]
    return X


def tree_matrix(df, groups, extra=()):
    cols = [c for g in groups for c in F.GROUPS[g]] + list(extra)
    X = df[cols].copy()
    for pr in PRODUCTS:
        X[f"prod_{pr}"] = (df["Product Name"] == pr).astype(float)
    return X


# ---- candidate regressors (hyperparameters fixed a priori)
def ridge():
    return RidgeCV(alphas=[0.1, 1, 10, 100, 300], cv=TimeSeriesSplit(4), scoring="neg_mean_squared_error")


MODELS = {
    # name: (estimator factory, feature groups, matrix kind, needs scaling)
    "OLS (F2)":              (lambda: LinearRegression(), F2, "lin", True),
    "Ridge (F2)":            (ridge, F2, "lin", True),
    "Ridge (F7)":            (ridge, F7, "lin", True),
    "Ridge (F2 + raw search)": (ridge, F2 + ["search_raw"], "lin", True),
    "Ridge (F2 + trend)":    (ridge, F2 + ["trend"], "lin", True),
    "Ridge (F2 + trend + raw search)": (ridge, F2 + ["trend", "search_raw"], "lin", True),
    "Decision tree (F2)":    (lambda: DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=0), F2, "tree", False),
    "Random forest (F2)":    (lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=-1), F2, "tree", False),
    "Random forest (F7)":    (lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=-1), F7, "tree", False),
    "Gradient boosting (F2)": (lambda: GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0), F2, "tree", False),
    "Gradient boosting (F7)": (lambda: GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0), F7, "tree", False),
}


def fit_predict(spec, tr, te):
    make, groups, kind, scale = spec
    mat = lin_matrix if kind == "lin" else tree_matrix
    Xtr, Xte = mat(tr, groups).values, mat(te, groups).values
    if scale:
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = make().fit(Xtr, tr["y"].values)
    return m.predict(Xtr), m.predict(Xte)


def seasonal_profile(f, tr, te):
    """Baseline: product training mean of the log rate + learned season index (no regression at all)."""
    base = f[f["month"] <= tr["month"].max()].groupby("Product Name")["lu"].mean()
    return (te["Product Name"].map(base) + te["season_idx"]).values


def cluster_labels(f, origin, k=3):
    """k-means on each product's 12-month seasonal profile, fitted on months <= origin only."""
    t = f[f["month"] <= origin]
    dev = t["lu"] - t.groupby("Product Name")["lu"].transform("mean")
    prof = dev.groupby([t["Product Name"], t["month"].dt.month]).mean().unstack()
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(prof.values)
    return dict(zip(prof.index, km.labels_))


def strategy_predict(f, tr, te, origin, how):
    """Ridge on F2 fitted pooled, per product, or per k-means cluster. Product dummies are dropped when a model sees one product."""
    cols = [c for g in F2 for c in F.GROUPS[g]]
    if how == "pooled":
        return fit_predict(MODELS["Ridge (F2)"], tr, te)[1]
    key = (lambda d: d["Product Name"]) if how == "per product" else None
    if how == "per cluster":
        lab = cluster_labels(f, origin); key = lambda d: d["Product Name"].map(lab)
    out = pd.Series(index=te.index, dtype=float)
    for g in sorted(key(te).unique()):
        a, b = tr[key(tr) == g], te[key(te) == g]
        sc = StandardScaler().fit(a[cols]); m = ridge().fit(sc.transform(a[cols]), a["y"])
        out[b.index] = m.predict(sc.transform(b[cols]))
    return out.values


# ====================================================================================== A + B: regression
rows, train_rows = [], []
for h in HORIZONS:
    for o in ORIGINS:
        if o + pd.DateOffset(months=h) > LAST_MONTH:
            continue
        f = frame(o, h)
        tr, te = split(f, o, h)
        if len(te) == 0:
            continue
        base = {"origin": o, "h": h, "month": te["month"].values, "product": te["Product Name"].values,
                "n_records": te["n_records"].values, "true": np.exp(te["y"].values)}
        def add(name, pred):
            rows.append(pd.DataFrame({**base, "model": name, "pred": np.exp(pred)}))
        add("Last value", te["lu_o"].values)
        add("Seasonal naive", te["lu_yoy_t"].values)
        add("Seasonal profile", seasonal_profile(f, tr, te))
        for name, spec in MODELS.items():
            ptr, pte = fit_predict(spec, tr, te)
            add(name, pte)
            if name == "Ridge (F2)":                       # retransformation bias: exp(mean of log) under-forecasts the mean (Duan smearing)
                add("Ridge (F2), bias-corrected", pte + np.log(np.mean(np.exp(tr["y"].values - ptr))))
            train_rows.append({"origin": o, "h": h, "model": name, "train_MAPE": M.mape(np.exp(tr["y"]), np.exp(ptr)), "n_train": len(tr)})
        for how in ("per product", "per cluster"):
            add(f"Ridge F2, {how}", strategy_predict(f, tr, te, o, how))
R = pd.concat(rows, ignore_index=True)
R["ape"] = np.abs(R["pred"] - R["true"]) / R["true"]
TR = pd.DataFrame(train_rows)
R.to_csv("processed/predictions_rolling.csv", index=False)
TR.to_csv("processed/train_error.csv", index=False)
print(f"regression done in {time.time() - T0:.0f}s: {len(R):,} predictions, {R['model'].nunique()} models, {len(ORIGINS)} origins")

out = {"origins": [str(o.date()) for o in ORIGINS], "horizons": list(HORIZONS), "surge_threshold_pct": 15, "regression": {}}
order = ["Last value", "Seasonal naive", "Seasonal profile", "OLS (F2)", "Ridge (F2)", "Ridge (F7)", "Ridge (F2 + raw search)", "Ridge (F2 + trend)", "Ridge (F2 + trend + raw search)", "Ridge (F2), bias-corrected", "Decision tree (F2)", "Random forest (F2)",
         "Random forest (F7)", "Gradient boosting (F2)", "Gradient boosting (F7)", "Ridge F2, per product", "Ridge F2, per cluster"]
for name in order:
    out["regression"][name] = {}
    for h in HORIZONS:
        d = R[(R["model"] == name) & (R["h"] == h)]
        s = M.summarize(d["true"], d["pred"], d["n_records"])
        tr_ = TR[(TR["model"] == name) & (TR["h"] == h)]
        s["train_MAPE"] = float(tr_["train_MAPE"].mean()) if len(tr_) else None
        out["regression"][name][f"h{h}"] = s

# ---- paired comparison against the reference model (Ridge F2): Wilcoxon + block bootstrap over target months
REF = "Ridge (F2)"
out["vs_ref"] = {"reference": REF}
for name in order:
    if name == REF:
        continue
    out["vs_ref"][name] = {}
    for h in HORIZONS:
        a = R[(R["model"] == name) & (R["h"] == h)].sort_values(["origin", "product"]).reset_index(drop=True)
        b = R[(R["model"] == REF) & (R["h"] == h)].sort_values(["origin", "product"]).reset_index(drop=True)
        d = (a["ape"] - b["ape"]).values
        mean, lo, hi = M.block_bootstrap_diff(a["ape"].values, b["ape"].values, a["month"].astype(str).values)
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else None
        out["vs_ref"][name][f"h{h}"] = {"diff_pp": mean * 100, "ci_lo_pp": lo * 100, "ci_hi_pp": hi * 100, "wilcoxon_p": p}
def pair(a_name, b_name, h):
    a = R[(R["model"] == a_name) & (R["h"] == h)].sort_values(["origin", "product"]).reset_index(drop=True)
    b = R[(R["model"] == b_name) & (R["h"] == h)].sort_values(["origin", "product"]).reset_index(drop=True)
    d = (a["ape"] - b["ape"]).values
    mean, lo, hi = M.block_bootstrap_diff(a["ape"].values, b["ape"].values, a["month"].astype(str).values)
    return {"diff_pp": mean * 100, "ci_lo_pp": lo * 100, "ci_hi_pp": hi * 100, "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
            "share_a_better": float((d < 0).mean())}
out["pairs"] = {f"{a} vs {b}": {f"h{h}": pair(a, b, h) for h in HORIZONS} for a, b in [
    ("Ridge (F2 + trend)", "Ridge (F2)"), ("Ridge (F2 + trend + raw search)", "Ridge (F2 + trend)"),
    ("Random forest (F2)", "Ridge (F2 + trend)"), ("Gradient boosting (F2)", "Ridge (F2 + trend)"), ("Decision tree (F2)", "Ridge (F2 + trend)"),
    ("Seasonal naive", "Ridge (F2 + trend)")]}
n_cmp = sum(1 for k, v in out["vs_ref"].items() if k != "reference" for _ in v)
out["n_comparisons_vs_ref"] = n_cmp

# ---- error by test year, and learning curve (error vs amount of training history)
R["year"] = pd.to_datetime(R["month"]).dt.year
out["by_year"] = {n: {str(y): M.mape(d["true"], d["pred"]) for y, d in R[(R["model"] == n) & (R["h"] == 2)].groupby("year")} for n in order}
# ---- bias by calendar month and by product (does the model under-forecast at peaks?)
d = R[(R["model"] == REF) & (R["h"] == 2)].copy()
d["moy"] = pd.to_datetime(d["month"]).dt.month
out["bias_by_month_ridge_h2"] = {int(k): M.bias_pct(g["true"], g["pred"]) for k, g in d.groupby("moy")}
out["bias_by_product_ridge_h2"] = {k: M.bias_pct(g["true"], g["pred"]) for k, g in d.groupby("product")}
dt_ = R[(R["model"] == "Ridge (F2 + trend)") & (R["h"] == 2)].copy(); dt_["moy"] = pd.to_datetime(dt_["month"]).dt.month
out["bias_by_month_trend_h2"] = {int(k): M.bias_pct(g["true"], g["pred"]) for k, g in dt_.groupby("moy")}
out["bias_by_year_h2"] = {n: {str(y): M.bias_pct(g["true"], g["pred"]) for y, g in R[(R["model"] == n) & (R["h"] == 2)].groupby("year")} for n in ["Ridge (F2)", "Ridge (F2 + trend)"]}
out["mape_by_product_h2"] = {n: {k: M.mape(g["true"], g["pred"]) for k, g in R[(R["model"] == n) & (R["h"] == 2)].groupby("product")} for n in ["Last value", "Seasonal naive", "Ridge (F2)", "Random forest (F2)", "Gradient boosting (F2)"]}
# residual autocorrelation of the reference model (are errors still carrying a pattern?)
res = d.assign(r=np.log(d["true"]) - np.log(d["pred"])).sort_values(["product", "month"])
ac = []
for _, g in res.groupby("product"):
    r = g["r"].values - g["r"].values.mean()
    if len(r) > 6:
        ac.append(float((r[1:] * r[:-1]).sum() / (r ** 2).sum()))
out["residual_lag1_autocorr_ridge_h2_mean"] = float(np.mean(ac))
print(f"summaries done {time.time() - T0:.0f}s")

# ====================================================================================== validation curves (single split, h = 2)
O_VC = pd.Timestamp("2023-12-01")
def split_all(f, origin):
    ok = f[NEED].notna().all(axis=1)
    return f[(f["month"] <= origin) & ok].sort_values("month"), f[(f["month"] > origin) & ok]

f = frame(O_VC, 2); tr, te = split_all(f, O_VC)
# decision-tree depth
vc_tree = {}
for depth in range(1, 11):
    m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=5, random_state=0).fit(tree_matrix(tr, F2).values, tr["y"].values)
    vc_tree[depth] = {"train": M.mape(np.exp(tr["y"]), np.exp(m.predict(tree_matrix(tr, F2).values))),
                      "test": M.mape(np.exp(te["y"]), np.exp(m.predict(tree_matrix(te, F2).values)))}
# ridge penalty
vc_ridge = {}
Xtr = lin_matrix(tr, F2).values; Xte = lin_matrix(te, F2).values; sc = StandardScaler().fit(Xtr)
from sklearn.linear_model import Ridge
for a in [0.001, 0.01, 0.1, 1, 10, 100, 300, 1000, 10000]:
    m = Ridge(alpha=a).fit(sc.transform(Xtr), tr["y"].values)
    vc_ridge[a] = {"train": M.mape(np.exp(tr["y"]), np.exp(m.predict(sc.transform(Xtr)))), "test": M.mape(np.exp(te["y"]), np.exp(m.predict(sc.transform(Xte))))}
# boosting rounds (staged predictions)
gb = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0).fit(tree_matrix(tr, F2).values, tr["y"].values)
vc_gb = {}
for i, (ptr, pte) in enumerate(zip(gb.staged_predict(tree_matrix(tr, F2).values), gb.staged_predict(tree_matrix(te, F2).values)), start=1):
    if i in (5, 10, 25, 50, 100, 150, 200, 300, 400):
        vc_gb[i] = {"train": M.mape(np.exp(tr["y"]), np.exp(ptr)), "test": M.mape(np.exp(te["y"]), np.exp(pte))}
out["validation_curves"] = {"origin": str(O_VC.date()), "h": 2, "n_train": len(tr), "n_test": len(te), "tree_depth": vc_tree, "ridge_alpha": vc_ridge, "gb_rounds": vc_gb}

# ---- learning curve: fixed test set (Jan 2024 - Apr 2025), training window = the last k months before the origin
lc2 = {}
for name in ["Ridge (F2)", "Random forest (F2)", "Gradient boosting (F2)", "Decision tree (F2)"]:
    lc2[name] = {}
    for k in (12, 18, 24, 30, 36):
        trk = tr[tr["month"] > O_VC - pd.DateOffset(months=k)]
        ptr, pte = fit_predict(MODELS[name], trk, te)
        lc2[name][k] = {"train": M.mape(np.exp(trk["y"]), np.exp(ptr)), "test": M.mape(np.exp(te["y"]), np.exp(pte)), "n_train": int(len(trk))}
out["learning_curve_fixed_test"] = lc2

# ---- clustering robustness: silhouette by k and stability of cluster membership across origins
from sklearn.metrics import adjusted_rand_score, silhouette_score
f_vc = frame(O_VC, 2); t_ = f_vc[f_vc["month"] <= O_VC]
dev = t_["lu"] - t_.groupby("Product Name")["lu"].transform("mean")
prof = dev.groupby([t_["Product Name"], t_["month"].dt.month]).mean().unstack()
out["cluster_silhouette_by_k"] = {k: float(silhouette_score(prof.values, KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(prof.values))) for k in (2, 3, 4, 5)}
final_lab = cluster_labels(frame(ORIGINS[-1], 1), ORIGINS[-1])
out["cluster_ari_vs_final"] = {str(o.date()): float(adjusted_rand_score([final_lab[p] for p in PRODUCTS], [cluster_labels(frame(o, 1), o)[p] for p in PRODUCTS])) for o in ORIGINS[::3]}
out["cluster_members_final"] = {str(k): sorted([p for p in PRODUCTS if final_lab[p] == k]) for k in sorted(set(final_lab.values()))}

# ====================================================================================== interpretation (single split, h = 2)
def grouped_permutation(model, mat_fn, te_, groups, n_rep=30, seed=0, scaler=None):
    """Increase in MAPE when the columns of one feature group are shuffled together (rows permuted jointly)."""
    rng = np.random.default_rng(seed)
    X = mat_fn(te_, groups)
    predict = lambda Z: model.predict(scaler.transform(Z) if scaler is not None else Z)
    base = M.mape(np.exp(te_["y"]), np.exp(predict(X.values)))
    cols = list(X.columns)
    sets = {
        "Season (month, flags, season index)": [i for i, c in enumerate(cols) if c in F.GROUPS["calendar"] or c.endswith("_x_sin") or c.endswith("_x_cos")],
        "Sales memory (lags, momentum, year-ago)": [i for i, c in enumerate(cols) if c in F.GROUPS["memory"]],
        "Product identity (one-hot)": [i for i, c in enumerate(cols) if c.startswith("prod_") and not (c.endswith("_x_sin") or c.endswith("_x_cos"))],
    }
    imp = {}
    for label, idx in sets.items():
        incr = []
        for _ in range(n_rep):
            Z = X.values.copy(); perm = rng.permutation(len(Z)); Z[:, idx] = Z[perm][:, idx]
            incr.append(M.mape(np.exp(te_["y"]), np.exp(predict(Z))) - base)
        imp[label] = {"mape_increase_pp": float(np.mean(incr)), "sd": float(np.std(incr)), "n_columns": len(idx)}
    return base, imp

interp = {}
for name in ["Ridge (F2)", "Random forest (F2)", "Gradient boosting (F2)"]:
    make, groups, kind, scale = MODELS[name]
    mat = lin_matrix if kind == "lin" else tree_matrix
    Xa = mat(tr, groups); sc_ = StandardScaler().fit(Xa.values) if scale else None
    m = make().fit(sc_.transform(Xa.values) if scale else Xa.values, tr["y"].values)
    base, imp = grouped_permutation(m, mat, te, groups, scaler=sc_)
    interp[name] = {"test_MAPE": base, "importance": imp}
    if name == "Ridge (F2)":
        coefs = pd.Series(m.coef_, index=Xa.columns)
        interp[name]["alpha"] = float(m.alpha_)
        interp[name]["top_coef"] = coefs[[c for c in coefs.index if not c.startswith("prod_")]].reindex(coefs[[c for c in coefs.index if not c.startswith("prod_")]].abs().sort_values(ascending=False).index).head(6).round(4).to_dict()
    if name == "Random forest (F2)":
        fi = pd.Series(m.feature_importances_, index=Xa.columns).sort_values(ascending=False)
        interp[name]["impurity_importance_top"] = fi.head(6).round(4).to_dict()
out["interpretation"] = {"origin": str(O_VC.date()), "h": 2, "n_test": int(len(te)), **interp}

# ---- H3 / H4 revisited with the Week 1 partial F-test (in-sample, restricted vs full OLS), origin Dec 2023
ft = {}
for h in HORIZONS:
    f = frame(O_VC, h); tr_h, _ = split(f, O_VC, h)
    base_cols = [c for g in F2 for c in F.GROUPS[g]]
    prod = pd.get_dummies(tr_h["Product Name"]).astype(float).values
    Xr = np.column_stack([tr_h[base_cols].values, prod[:, 1:]])
    Xs = np.column_stack([Xr, tr_h[F.GROUPS["search_raw"]].values])
    Fv, d1, d2, p = M.partial_f_test(tr_h["y"].values, Xr, Xs)
    rv = f[(f["month"] <= O_VC) & f[NEED + F.GROUPS["reviews"]].notna().all(axis=1)].sort_values("month")
    prodv = pd.get_dummies(rv["Product Name"]).astype(float).values
    Xr2 = np.column_stack([rv[base_cols].values, prodv[:, 1:], rv[F.GROUPS["search_raw"]].values])
    Xv = np.column_stack([Xr2, rv[F.GROUPS["reviews"]].values])
    Fv2, e1, e2, p2 = M.partial_f_test(rv["y"].values, Xr2, Xv)
    ft[f"h{h}"] = {"H3_search": {"F": Fv, "df1": d1, "df2": d2, "p": p, "n": int(len(tr_h))}, "H4_reviews": {"F": Fv2, "df1": e1, "df2": e2, "p": p2, "n": int(len(rv))}}
out["partial_F"] = ft
print(f"validation + interpretation done {time.time() - T0:.0f}s")

# ====================================================================================== C: surge early-warning classifier
CLF = {
    "Logistic regression": (lambda: LogisticRegression(C=1.0, max_iter=2000), "lin", True),
    "Decision tree": (lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=0), "tree", False),
    "Random forest": (lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=-1), "tree", False),
    "Gradient boosting": (lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0), "tree", False),
}
EXTRA = ("lu_mean12_o", "lu_dev12_o")
crow = []
for h in HORIZONS:
    for o in ORIGINS:
        if o + pd.DateOffset(months=h) > LAST_MONTH:
            continue
        f = frame(o, h); tr, te = split(f, o, h)
        if len(te) == 0:
            continue
        ytr, yte = tr["surge"].values, te["surge"].values
        b = {"origin": o, "h": h, "month": te["month"].values, "product": te["Product Name"].values, "y": yte}
        crow.append(pd.DataFrame({**b, "model": "Persistence (last month a surge)", "score": (te["lu_dev12_o"].values >= SURGE).astype(float), "pred": (te["lu_dev12_o"].values >= SURGE).astype(float)}))
        seas = (te["lu_yoy_t"].values - te["lu_mean12_o"].values >= SURGE).astype(float)
        crow.append(pd.DataFrame({**b, "model": "Seasonal rule (surge a year ago)", "score": seas, "pred": seas}))
        crow.append(pd.DataFrame({**b, "model": "Predict never", "score": 0.0, "pred": 0.0}))
        if len(set(ytr)) < 2:
            continue
        for name, (make, kind, scale) in CLF.items():
            mat = lin_matrix if kind == "lin" else tree_matrix
            Xtr, Xte = mat(tr, F2, EXTRA).values, mat(te, F2, EXTRA).values
            if scale:
                sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            m = make().fit(Xtr, ytr); s = m.predict_proba(Xte)[:, 1]
            crow.append(pd.DataFrame({**b, "model": name, "score": s, "pred": (s >= 0.5).astype(float)}))
C = pd.concat(crow, ignore_index=True)
C.to_csv("processed/predictions_surge.csv", index=False)
out["classification"] = {}
for name in C["model"].unique():
    out["classification"][name] = {}
    for h in HORIZONS:
        d = C[(C["model"] == name) & (C["h"] == h)]
        out["classification"][name][f"h{h}"] = M.classification_scores(d["y"], d["pred"], d["score"] if name in CLF else None)
json.dump(out, open("processed/models5_results.json", "w"), indent=1, default=str)
print(f"all done {time.time() - T0:.0f}s")
for n in order:
    print(f"{n:26s}", "  ".join(f"h{h}: MAPE {out['regression'][n][f'h{h}']['MAPE']:5.2f} RMSE {out['regression'][n][f'h{h}']['RMSE_units']:6.1f}" for h in HORIZONS))
for n, v in out["classification"].items():
    print(f"{n:34s}", "  ".join(f"h{h}: P {v[f'h{h}']['precision']:.2f} R {v[f'h{h}']['recall']:.2f} F1 {v[f'h{h}']['F1']:.2f}" for h in HORIZONS), " base", round(v["h2"]["base_rate"], 2))
