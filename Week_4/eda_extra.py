"""Week 4 - EDA stage 2: coverage check, demand-rate view, Week 2 validation gate, autocorrelation, transformations.

Stage 1 (eda.py) explored raw monthly units. This stage follows up on what stage 1 showed:
units are dominated by how many records fall in each month, so the analysis is repeated on the
coverage-normalised "demand rate" (units per record). Writes figures_w4/*.png and processed/eda_extra.json.
"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTE, GRID = "#1F2933", "#6B7785", "#E3E7EB"
C1, C2, C3, C4 = "#2F6F6B", "#C8553D", "#8A94A6", "#E2A93B"
sns.set_theme(style="white", rc={"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                                 "xtick.color": MUTE, "ytick.color": MUTE, "axes.spines.top": False, "axes.spines.right": False,
                                 "axes.titlesize": 10, "axes.titleweight": "regular", "axes.titlecolor": INK})
panel = pd.read_csv("processed/panel_monthly.csv", parse_dates=["month"]).sort_values(["Product Name", "month"]).reset_index(drop=True)
weekly = pd.read_csv("processed/sales_clean_weekly.csv", parse_dates=["Date"])
summ = json.load(open("processed/run_summary.json"))
PRODUCTS = sorted(panel["Product Name"].unique()); MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
panel["upr"] = panel["units"] / panel["n_records"]
R = {}
zs = lambda s: (s - s.groupby(panel["Product Name"]).transform("mean")) / s.groupby(panel["Product Name"]).transform("std")
cv = lambda s: s.std() / s.mean()

# ---------------------------------------------------------------- 1. coverage artefact
g = panel.groupby("Product Name")
R["n_records"] = dict(mean=float(panel["n_records"].mean()), sd=float(panel["n_records"].std()), min=int(panel["n_records"].min()), max=int(panel["n_records"].max()))
R["cv_units"] = float(g["units"].apply(cv).mean()); R["cv_n_records"] = float(g["n_records"].apply(cv).mean()); R["cv_upr"] = float(g["upr"].apply(cv).mean())
R["corr_units_nrecords"] = float(np.corrcoef(zs(panel["units"]), zs(panel["n_records"]))[0, 1])
R["corr_units_upr"] = float(np.corrcoef(zs(panel["units"]), zs(panel["upr"]))[0, 1])
dm = lambda s: s - s.groupby(panel["Product Name"]).transform("mean")
R["var_share_nrecords"] = float(np.var(dm(np.log(panel["n_records"]))) / np.var(dm(np.log(panel["units"]))))
R["var_share_upr"] = float(np.var(dm(np.log(panel["upr"]))) / np.var(dm(np.log(panel["units"]))))
fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.0))
ax[0].scatter(zs(panel["n_records"]), zs(panel["units"]), s=9, color=C3, alpha=0.6)
b = np.polyfit(zs(panel["n_records"]), zs(panel["units"]), 1); xx = np.linspace(-2.5, 3.5, 20); ax[0].plot(xx, b[0] * xx + b[1], color=C2, lw=1.8)
ax[0].set_xlabel("records in the month (z-score)"); ax[0].set_ylabel("monthly units (z-score)")
ax[0].set_title(f"Units follow the record count\nr = {R['corr_units_nrecords']:.2f}")
vals = [R["cv_units"], R["cv_n_records"], R["cv_upr"]]
bars = ax[1].bar(["units", "records", "units per\nrecord"], vals, color=[C3, C4, C1], width=0.55)
for r_, v in zip(bars, vals): ax[1].text(r_.get_x() + r_.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
ax[1].set_ylabel("coefficient of variation"); ax[1].set_title("Variability of each series")
fig.tight_layout(); fig.savefig("figures_w4/fig_coverage.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 2. seasonal structure: units vs rate
def decompose(s):
    s = s.astype(float); trend = s.rolling(12, center=True).mean().rolling(2).mean().shift(-1); return trend, s / trend
def strength_and_index(col):
    fs, idx, slope = {}, {}, {}
    for p in PRODUCTS:
        s = panel[panel["Product Name"] == p].set_index("month")[col]; trend, ratio = decompose(s)
        moy = ratio.groupby(ratio.index.month).mean(); moy = moy / moy.mean(); idx[p] = moy.values
        lr = np.log(ratio.dropna()); rem = lr - np.log(moy.reindex(lr.index.month).values); fs[p] = float(max(0, 1 - rem.var() / lr.var()))
        tt = trend.dropna(); slope[p] = float(np.expm1(np.polyfit(np.arange(len(tt)), np.log(tt.values), 1)[0] * 12))
    return fs, pd.DataFrame(idx, index=MON).T, slope
fs_u, S_u, _ = strength_and_index("units"); fs_r, S_r, sl_r = strength_and_index("upr")
R["seasonal_strength_units_mean"] = float(np.mean(list(fs_u.values()))); R["seasonal_strength_upr_mean"] = float(np.mean(list(fs_r.values())))
R["seasonal_strength_upr"] = {k: round(v, 3) for k, v in fs_r.items()}
R["peak_month_upr"] = {p: MON[int(np.argmax(S_r.loc[p].values))] for p in PRODUCTS}; R["trough_month_upr"] = {p: MON[int(np.argmin(S_r.loc[p].values))] for p in PRODUCTS}
R["amp_ratio_upr"] = {p: round(float(S_r.loc[p].max() / S_r.loc[p].min()), 2) for p in PRODUCTS}
R["trend_pct_per_year_upr"] = {k: round(v * 100, 2) for k, v in sl_r.items()}
fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
for a, S, ttl in [(ax[0], S_u, "Raw monthly units"), (ax[1], S_r, "Units per record (demand rate)")]:
    for p in PRODUCTS: a.plot(range(1, 13), S.loc[p].values, color=C3, lw=0.9, alpha=0.8)
    a.set_title(ttl); a.set_xticks(range(1, 13)); a.set_xticklabels([m[0] for m in MON]); a.axhline(1, color=MUTE, lw=0.6)
ax[0].set_ylabel("seasonal index (1 = average)"); ax[0].text(0.03, 0.04, f"seasonal strength {R['seasonal_strength_units_mean']:.2f}", transform=ax[0].transAxes, color=C2, fontsize=8)
ax[1].text(0.03, 0.04, f"seasonal strength {R['seasonal_strength_upr_mean']:.2f}", transform=ax[1].transAxes, color=C1, fontsize=8)
fig.tight_layout(); fig.savefig("figures_w4/fig_seasonal_units_vs_rate.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 3. search vs demand rate: raw and de-seasonalised
def residualise(col):
    out = pd.Series(index=panel.index, dtype=float)
    for p, gg in panel.groupby("Product Name"):
        X = np.column_stack([np.arange(len(gg))] + [(gg["month"].dt.month.values == m).astype(float) for m in range(1, 13)])
        beta, *_ = np.linalg.lstsq(X, gg[col].values.astype(float), rcond=None); out.loc[gg.index] = gg[col].values - X @ beta
    return out
panel["upr_res"] = residualise("upr"); panel["search_res"] = residualise("search_index")
def ccf(a, b, maxlag=6):
    return np.array([[gg[a].corr(gg[b].shift(k)) for _, gg in panel.groupby("Product Name")] for k in range(maxlag + 1)])
raw_c, res_c = ccf("upr", "search_index"), ccf("upr_res", "search_res")
n_obs = int(panel.groupby("Product Name").size().iloc[0]); bound = 1.96 / np.sqrt(n_obs)
R["ccf_upr_raw_mean"] = [round(float(np.nanmean(r)), 3) for r in raw_c]; R["ccf_upr_res_mean"] = [round(float(np.nanmean(r)), 3) for r in res_c]
R["ccf_bound"] = float(bound); R["ccf_upr_res_n_sig"] = int((np.abs(res_c) > bound).sum()); R["ccf_upr_res_n_tests"] = int(res_c.size)
R["ccf_upr_res_expected_by_chance"] = float(res_c.size * 0.05)
R["upr_share_var_trend_season"] = float(1 - panel["upr_res"].var() / (panel["upr"] - panel.groupby("Product Name")["upr"].transform("mean")).var())
fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=True); lags = np.arange(raw_c.shape[0])
for a, M, ttl, col in [(ax[0], raw_c, "Raw demand rate", C2), (ax[1], res_c, "After removing trend and seasonality", C1)]:
    for i in range(M.shape[1]): a.plot(lags, M[:, i], color=C3, lw=0.8, alpha=0.7)
    a.plot(lags, np.nanmean(M, axis=1), color=col, lw=2.4, marker="o", ms=4, label="mean of 8 products")
    a.axhspan(-bound, bound, color=GRID, alpha=0.7, zorder=0); a.axhline(0, color=MUTE, lw=0.6)
    a.set_title(ttl); a.set_xlabel("search interest lag (months)"); a.legend(frameon=False, fontsize=7, loc="upper right")
ax[0].set_ylabel("correlation with demand rate"); fig.tight_layout(); fig.savefig("figures_w4/fig_ccf_rate.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 4. Week 2 validation gate
wk = weekly.groupby(["Product Name", "Date"])["Units Sold"].mean().reset_index()          # mean units per record, per product-week
R["gate1_empty_product_months"] = int(panel["units"].isna().sum()); R["gate1_min_records"] = int(panel["n_records"].min())
R["gate2_cv_weekly_records"] = float(weekly.groupby("Product Name")["Units Sold"].apply(cv).mean())
R["gate3_seasonal_strength_mean"] = R["seasonal_strength_upr_mean"]
acf = []
for p, gg in panel.groupby("Product Name"):
    s = np.log(gg["upr"].values); s = s - s.mean(); den = (s ** 2).sum()
    acf.append([float((s[k:] * s[:len(s) - k]).sum() / den) for k in range(1, 13)])
acf = np.array(acf); R["gate5_acf_mean"] = [round(float(x), 3) for x in acf.mean(axis=0)]
R["gate5_acf_lags_over_bound"] = int((acf.mean(axis=0) > 1.96 / np.sqrt(n_obs)).sum()); R["acf_bound"] = float(1.96 / np.sqrt(n_obs))
R["gate6"] = {"revenue_identity_share": summ["checks"]["revenue = units x price (tol 0.01 per unit)"], "discount_in_0_1": summ["checks"]["discount within 0-1"], "returns_le_units": summ["checks"]["returns <= units"]}
R["gate_decision"] = "PASS: structure present in checks 2, 3 and 5" if (R["gate2_cv_weekly_records"] > 0.1 and R["gate3_seasonal_strength_mean"] > 0.2 and R["gate5_acf_lags_over_bound"] > 0) else "FAIL"
fig, ax = plt.subplots(figsize=(6.2, 2.9)); k = np.arange(1, 13)
for i in range(acf.shape[0]): ax.plot(k, acf[i], color=C3, lw=0.8, alpha=0.7)
ax.plot(k, acf.mean(axis=0), color=C1, lw=2.4, marker="o", ms=4, label="mean of 8 products")
ax.axhspan(-R["acf_bound"], R["acf_bound"], color=GRID, alpha=0.7, zorder=0); ax.axhline(0, color=MUTE, lw=0.6)
ax.set_xlabel("lag (months)"); ax.set_ylabel("autocorrelation"); ax.set_xticks(k); ax.legend(frameon=False, fontsize=8)
ax.set_title("Autocorrelation of the monthly demand rate")
fig.tight_layout(); fig.savefig("figures_w4/fig_acf.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 5. transformations (distribution shape)
u, r = panel["units"].values, panel["upr"].values
R["skew"] = {"units": float(stats.skew(u)), "log_units": float(stats.skew(np.log(u))), "sqrt_units": float(stats.skew(np.sqrt(u))),
             "upr": float(stats.skew(r)), "log_upr": float(stats.skew(np.log(r))), "sqrt_upr": float(stats.skew(np.sqrt(r)))}
bc, lam = stats.boxcox(u); R["boxcox_lambda_units"] = float(lam); R["skew_boxcox_units"] = float(stats.skew(bc))
bc2, lam2 = stats.boxcox(r); R["boxcox_lambda_upr"] = float(lam2); R["skew_boxcox_upr"] = float(stats.skew(bc2))
try:
    from statsmodels.tsa.stattools import adfuller
    pv = [adfuller(np.log(gg["upr"].values), regression="ct")[1] for _, gg in panel.groupby("Product Name")]
    R["adf_p_median"] = float(np.median(pv)); R["adf_n_stationary_5pct"] = int(sum(x < 0.05 for x in pv))
except Exception as e:
    R["adf_p_median"] = None

# ---------------------------------------------------------------- 6. reviews on the rate
rv = panel[panel["reviews_covered"]]
for c in ["review_count", "avg_rating"]:
    rho, pv_ = stats.spearmanr(zs(panel[c])[rv.index], zs(panel["upr"])[rv.index]); R[f"spearman_{c}_upr"] = {"rho": round(float(rho), 3), "p": round(float(pv_), 3)}


# ---------------------------------------------------------------- 8. seasonal archetypes on the demand rate
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
Zl = linkage(S_r.values, method="ward"); best = None
for kk in (2, 3, 4):
    lab = fcluster(Zl, kk, criterion="maxclust")
    if len(set(lab)) > 1:
        sc = silhouette_score(S_r.values, lab)
        if best is None or sc > best[1]: best = (kk, sc, lab)
R["cluster_upr"] = {"k": int(best[0]), "silhouette": float(best[1]), "members": {p: int(l) for p, l in zip(PRODUCTS, best[2])}}
cg = sns.clustermap(S_r, row_linkage=Zl, col_cluster=False, cmap="RdBu_r", center=1, figsize=(6.4, 3.8), linewidths=0.4, linecolor="white",
                    cbar_pos=None, dendrogram_ratio=(0.14, 0.001), annot=True, fmt=".2f", annot_kws={"size": 6})
cg.ax_heatmap.set_ylabel(""); cg.ax_heatmap.tick_params(labelsize=8); cg.savefig("figures_w4/fig_seasonal_heatmap_rate.png", dpi=200); plt.close("all")

# ---------------------------------------------------------------- 7. EDA workflow flowchart
fig, ax = plt.subplots(figsize=(7.4, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 78); ax.axis("off")
def box(x, y, w, h, t, s, ec=C1, fc="#FFFFFF"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h * 0.70, t, ha="center", va="center", fontsize=8.2, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, s, ha="center", va="center", fontsize=6.9, color=MUTE, linespacing=1.15)
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=MUTE, lw=1.1))
steps = [("1 Profile", "shape, missing,\nranges, types"), ("2 Distributions", "skew, outliers,\ntransforms"), ("3 Time structure", "trend, season,\nautocorrelation"),
         ("4 Relationships", "lead-lag, price,\npromo, channel"), ("5 Joint view", "correlation, VIF,\nPCA, clusters")]
xs = [1, 21, 41, 61, 81]
for (t, s), x in zip(steps, xs): box(x, 56, 18, 13, t, s)
for i in range(4): arr(xs[i] + 18.5, 63.5, xs[i + 1] - 0.5, 63.5)
box(1, 36, 44, 13, "Does a finding look too good, too weak or odd?", "coverage artefact? shared seasonality?\nchance findings from many tests?", ec=C4, fc="#FDF1E3")
box(55, 36, 44, 13, "Re-run on the corrected view", "units per record;\nde-seasonalised series", ec=C4, fc="#FDF1E3")
arr(45.5, 43.5, 54.5, 43.5)
for x in (10, 30, 50, 70, 90): pass
arr(50, 55.5, 23, 49.7)
box(1, 18, 44, 11, "6 Feature plan", "each finding names the feature it motivates", ec=C1, fc="#EAF3F2")
box(55, 18, 44, 11, "7 Time-based test of features", "train to a cut-off, score later months", ec=C1, fc="#EAF3F2")
arr(77, 35.6, 23, 29.7); arr(45.5, 23.5, 54.5, 23.5)
box(1, 1, 98, 9, "8 Business reading", "what decision does each finding support, and how sure are we?", ec=C4, fc="#FDF1E3")
arr(77, 17.5, 50, 10.7)
fig.savefig("figures_w4/fig_eda_flow.png", dpi=200, bbox_inches="tight"); plt.close(fig)

panel.to_csv("processed/panel_with_rate.csv", index=False)
json.dump(R, open("processed/eda_extra.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in R.items()}, indent=1, default=str))
