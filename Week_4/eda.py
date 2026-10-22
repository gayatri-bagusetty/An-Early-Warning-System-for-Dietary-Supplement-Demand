"""Week 4 - Exploratory data analysis on the cleaned Week 3 panel.

Reads  processed/panel_monthly.csv and processed/sales_clean_weekly.csv  (outputs of pipeline.py)
Writes figures_w4/*.png and processed/eda_results.json (every number quoted in the report).

NOTE: the panel comes from SIMULATED data. The EDA workflow is real and reusable; the findings
describe the simulation and must be re-checked on the real Kaggle / Trends / review files.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

Path("figures_w4").mkdir(exist_ok=True)
INK, MUTE, GRID = "#1F2933", "#6B7785", "#E3E7EB"
C1, C2, C3, C4 = "#2F6F6B", "#C8553D", "#8A94A6", "#E2A93B"
sns.set_theme(style="white", rc={"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                                 "xtick.color": MUTE, "ytick.color": MUTE, "axes.spines.top": False, "axes.spines.right": False,
                                 "axes.titlesize": 10, "axes.titleweight": "regular", "axes.titlecolor": INK})
R = {}   # results

panel = pd.read_csv("processed/panel_monthly.csv", parse_dates=["month"]).sort_values(["Product Name", "month"]).reset_index(drop=True)
weekly = pd.read_csv("processed/sales_clean_weekly.csv", parse_dates=["Date"])
PRODUCTS = sorted(panel["Product Name"].unique())
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ============================================================ 1. profile
R["shape"] = list(panel.shape)
R["n_products"] = len(PRODUCTS); R["n_months"] = int(panel["month"].nunique())
miss = panel.isna().sum(); miss = miss[miss > 0]
R["missing_by_column"] = {k: int(v) for k, v in miss.items()}
R["review_missing_months"] = int((~panel.loc[panel["Product Name"] == PRODUCTS[0], "reviews_covered"]).sum())
prof = panel.groupby("Product Name").agg(mean_units=("units", "mean"), sd_units=("units", "std"), min_units=("units", "min"),
                                         max_units=("units", "max"), price=("price", "mean"), discount=("discount", "mean"),
                                         search=("search_index", "mean"))
prof["cv"] = prof["sd_units"] / prof["mean_units"]
R["profile"] = prof.round(3).reset_index().to_dict("records")

# ============================================================ 2. distributions
u = panel["units"]
R["skew_units_pooled"] = float(stats.skew(u)); R["skew_logunits_pooled"] = float(stats.skew(np.log1p(u)))
R["kurt_units_pooled"] = float(stats.kurtosis(u))
ps = panel.groupby("Product Name")["units"].apply(lambda s: stats.skew(s)); pl = panel.groupby("Product Name")["units"].apply(lambda s: stats.skew(np.log1p(s)))
R["skew_units_by_product_median"] = float(ps.median()); R["skew_log_by_product_median"] = float(pl.median())
sh = panel.groupby("Product Name")["units"].apply(lambda s: stats.shapiro(s).pvalue)
R["shapiro_units_p_gt_05"] = int((sh > 0.05).sum()); R["shapiro_units_p_min"] = float(sh.min())
R["cv_range"] = [float(prof["cv"].min()), float(prof["cv"].max())]
fig, ax = plt.subplots(1, 3, figsize=(8.2, 2.9))
sns.histplot(u, bins=30, color=C1, ax=ax[0], edgecolor="white"); ax[0].set_title(f"Units, all product-months\nskew {R['skew_units_pooled']:.2f}"); ax[0].set_xlabel("units per month")
sns.histplot(np.log1p(u), bins=30, color=C4, ax=ax[1], edgecolor="white"); ax[1].set_title(f"log(1 + units)\nskew {R['skew_logunits_pooled']:.2f}"); ax[1].set_xlabel("log units"); ax[1].set_ylabel("")
order = prof.sort_values("mean_units").index
sns.boxplot(data=panel, y="Product Name", x="units", order=order, color=C1, fliersize=2, linewidth=0.8, ax=ax[2]); ax[2].set_title("Units by product"); ax[2].set_ylabel(""); ax[2].set_xlabel("units per month")
fig.tight_layout(); fig.savefig("figures_w4/fig_distributions.png", dpi=200); plt.close(fig)

# ============================================================ 3. trend and seasonality
def decompose(s):
    """Classical multiplicative decomposition with a centred 2x12 moving average."""
    s = s.astype(float)
    trend = s.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
    ratio = s / trend
    return trend, ratio

season = {}; trend_slope = {}; fs = {}; trend_series = {}
for p in PRODUCTS:
    g = panel[panel["Product Name"] == p].set_index("month")["units"]
    trend, ratio = decompose(g)
    moy = ratio.groupby(ratio.index.month).mean()
    moy = moy / moy.mean()
    season[p] = moy.values
    lr = np.log(ratio.dropna()); rem = lr - np.log(moy.reindex(lr.index.month).values)
    fs[p] = float(max(0, 1 - rem.var() / lr.var()))
    tt = trend.dropna(); x = np.arange(len(tt))
    slope = np.polyfit(x, np.log(tt.values), 1)[0]; trend_slope[p] = float(np.expm1(slope * 12))
    trend_series[p] = trend
S = pd.DataFrame(season, index=MON).T
R["seasonal_strength"] = {k: round(v, 3) for k, v in fs.items()}
R["trend_pct_per_year"] = {k: round(v * 100, 2) for k, v in trend_slope.items()}
R["peak_month"] = {p: MON[int(np.argmax(S.loc[p].values))] for p in PRODUCTS}
R["trough_month"] = {p: MON[int(np.argmin(S.loc[p].values))] for p in PRODUCTS}
R["seasonal_amp_ratio"] = {p: round(float(S.loc[p].max() / S.loc[p].min()), 2) for p in PRODUCTS}
R["seasonal_index"] = S.round(3).reset_index().rename(columns={"index": "Product Name"}).to_dict("records")

fig, axs = plt.subplots(2, 4, figsize=(8.4, 3.9), sharex=True)
for ax, p in zip(axs.ravel(), PRODUCTS):
    g = panel[panel["Product Name"] == p]
    ax.plot(g["month"], g["units"], color=C3, lw=0.9); ax.plot(trend_series[p].index, trend_series[p].values, color=C2, lw=1.6)
    ax.set_title(p, fontsize=8.5); ax.tick_params(labelsize=7); ax.set_xlabel("")
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2)); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
axs[0, 0].set_ylabel("units / month"); axs[1, 0].set_ylabel("units / month")
fig.tight_layout(); fig.savefig("figures_w4/fig_trend_small_multiples.png", dpi=200); plt.close(fig)

# seasonal clustering (product archetypes)
Z = linkage(S.values, method="ward")
best = None
for k in (2, 3, 4):
    lab = fcluster(Z, k, criterion="maxclust")
    if len(set(lab)) > 1:
        sc = silhouette_score(S.values, lab)
        if best is None or sc > best[1]: best = (k, sc, lab)
R["cluster_k"] = int(best[0]); R["cluster_silhouette"] = float(best[1])
R["clusters"] = {p: int(l) for p, l in zip(PRODUCTS, best[2])}
cg = sns.clustermap(S, row_linkage=Z, col_cluster=False, cmap="RdBu_r", center=1, figsize=(6.6, 3.8), linewidths=0.4, linecolor="white",
                    cbar_kws={"label": "seasonal index (1 = average month)"}, dendrogram_ratio=(0.14, 0.001), annot=True, fmt=".2f", annot_kws={"size": 6})
cg.ax_heatmap.set_ylabel(""); cg.ax_heatmap.tick_params(labelsize=8)
cg.savefig("figures_w4/fig_seasonal_heatmap.png", dpi=200); plt.close("all")

# ============================================================ 4. search vs sales
def residualise(df, col):
    """Remove a linear trend and month-of-year means (product-specific) from a series."""
    out = pd.Series(index=df.index, dtype=float)
    for p, g in df.groupby("Product Name"):
        y = g[col].values.astype(float)
        X = np.column_stack([np.arange(len(g))] + [(g["month"].dt.month.values == m).astype(float) for m in range(1, 13)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[g.index] = y - X @ beta
    return out
panel["units_res"] = residualise(panel, "units"); panel["search_res"] = residualise(panel, "search_index")
def ccf(df, a, b, maxlag=6):
    rows = []
    for k in range(0, maxlag + 1):
        rr = []
        for p, g in df.groupby("Product Name"):
            rr.append(g[a].corr(g[b].shift(k)))
        rows.append(rr)
    return np.array(rows)   # lags x products
raw_c = ccf(panel, "units", "search_index"); res_c = ccf(panel, "units_res", "search_res")
n_obs = panel.groupby("Product Name").size().iloc[0]; bound = 1.96 / np.sqrt(n_obs)
R["ccf_bound_single"] = float(bound)
R["ccf_raw_mean"] = [round(float(np.nanmean(r)), 3) for r in raw_c]
R["ccf_res_mean"] = [round(float(np.nanmean(r)), 3) for r in res_c]
R["ccf_raw_best_lag_by_product"] = {p: int(np.nanargmax(raw_c[:, i])) for i, p in enumerate(PRODUCTS)}
R["ccf_raw_best_r_by_product"] = {p: round(float(np.nanmax(raw_c[:, i])), 3) for i, p in enumerate(PRODUCTS)}
R["ccf_res_products_sig_any_lag"] = int(sum((np.abs(res_c[:, i]) > bound).any() for i in range(len(PRODUCTS))))
R["ccf_res_max_abs"] = float(np.nanmax(np.abs(res_c)))
R["ccf_raw_products_sig_lag1or2"] = int(sum(((raw_c[1:3, i]) > bound).any() for i in range(len(PRODUCTS))))
# share of units variance explained by product x month-of-year pattern
R["r2_seasonal_units"] = float(1 - panel["units_res"].var() / panel.groupby("Product Name")["units"].transform(lambda s: s - s.mean()).var())
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
lags = np.arange(raw_c.shape[0])
for a, M, ttl, col in [(ax[0], raw_c, "Raw series", C2), (ax[1], res_c, "After removing trend and seasonality", C1)]:
    for i in range(M.shape[1]): a.plot(lags, M[:, i], color=C3, lw=0.8, alpha=0.7)
    a.plot(lags, np.nanmean(M, axis=1), color=col, lw=2.4, marker="o", ms=4, label="mean of 8 products")
    a.axhspan(-bound, bound, color=GRID, alpha=0.7, zorder=0); a.axhline(0, color=MUTE, lw=0.6)
    a.set_title(ttl); a.set_xlabel("search interest lag (months)"); a.legend(frameon=False, fontsize=7, loc="upper right")
ax[0].set_ylabel("correlation with units")
fig.tight_layout(); fig.savefig("figures_w4/fig_ccf.png", dpi=200); plt.close(fig)

# ============================================================ 5. price, discount, platform, location (weekly records)
w = weekly.copy()
w["units_rel"] = w["Units Sold"] / w.groupby("Product Name")["Units Sold"].transform("mean")
w["disc_pct"] = (w["Discount"] * 100).round().astype(int)
R["weekly_rows"] = len(w)
def kw(col):
    groups = [g["units_rel"].values for _, g in w.groupby(col)]
    h, p = stats.kruskal(*groups)
    return float(h), float(p)
for col in ["disc_pct", "Platform", "Location"]:
    h, p = kw(col); R[f"kruskal_{col}"] = {"H": round(h, 3), "p": round(p, 4)}
R["group_means_discount"] = w.groupby("disc_pct")["units_rel"].mean().round(3).to_dict()
R["group_means_platform"] = w.groupby("Platform")["units_rel"].mean().round(3).to_dict()
R["group_means_location"] = w.groupby("Location")["units_rel"].mean().round(3).to_dict()
R["share_records_platform"] = (w["Platform"].value_counts(normalize=True) * 100).round(1).to_dict()
R["share_records_location"] = (w["Location"].value_counts(normalize=True) * 100).round(1).to_dict()
w["price_rel"] = w["Price"] / w.groupby("Product Name")["Price"].transform("mean")
rho, p = stats.spearmanr(w["price_rel"], w["units_rel"]); R["spearman_price_units"] = {"rho": round(float(rho), 3), "p": round(float(p), 3)}
rho, p = stats.spearmanr(w["Discount"], w["units_rel"]); R["spearman_discount_units"] = {"rho": round(float(rho), 3), "p": round(float(p), 3)}
fig, ax = plt.subplots(1, 3, figsize=(8.2, 2.9), sharey=True)
sns.boxplot(data=w, x="disc_pct", y="units_rel", color=C1, fliersize=1.5, linewidth=0.8, ax=ax[0]); ax[0].set_xlabel("discount (%)"); ax[0].set_ylabel("units vs product average")
sns.boxplot(data=w, x="Platform", y="units_rel", color=C4, fliersize=1.5, linewidth=0.8, ax=ax[1], order=["Amazon", "Walmart", "iHerb"]); ax[1].set_ylabel("")
sns.boxplot(data=w, x="Location", y="units_rel", color=C3, fliersize=1.5, linewidth=0.8, ax=ax[2], order=["USA", "UK", "Canada"]); ax[2].set_ylabel("")
ax[0].set_title(f"By discount  (p = {R['kruskal_disc_pct']['p']:.2f})"); ax[1].set_title(f"By platform  (p = {R['kruskal_Platform']['p']:.2f})"); ax[2].set_title(f"By location  (p = {R['kruskal_Location']['p']:.2f})")
for a in ax: a.axhline(1, color=C2, lw=1, ls="--")
fig.tight_layout(); fig.savefig("figures_w4/fig_price_promo_channel.png", dpi=200); plt.close(fig)

# ============================================================ 6. reviews (H4)
rv = panel[panel["reviews_covered"]].copy()
R["review_months_per_product"] = int(rv.groupby("Product Name").size().iloc[0])
for c in ["review_count", "avg_rating"]:
    z = lambda s: (s - s.groupby(rv["Product Name"]).transform("mean")) / s.groupby(rv["Product Name"]).transform("std")
    rho, p = stats.spearmanr(z(rv[c]), z(rv["units"])); R[f"spearman_{c}_units"] = {"rho": round(float(rho), 3), "p": round(float(p), 3)}
rv["units_res_c"] = rv["units_res"];
rho, p = stats.spearmanr(rv["review_count"], rv["units_res"]); R["spearman_reviewcount_units_res"] = {"rho": round(float(rho), 3), "p": round(float(p), 3)}
R["review_count_mean"] = float(rv["review_count"].mean()); R["avg_rating_mean"] = float(rv["avg_rating"].mean()); R["avg_rating_sd"] = float(rv["avg_rating"].std())
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
for a, c, ttl in [(ax[0], "review_count", "Review volume vs units"), (ax[1], "avg_rating", "Average rating vs units")]:
    zx = (rv[c] - rv.groupby("Product Name")[c].transform("mean")) / rv.groupby("Product Name")[c].transform("std")
    zy = (rv["units"] - rv.groupby("Product Name")["units"].transform("mean")) / rv.groupby("Product Name")["units"].transform("std")
    a.scatter(zx, zy, s=9, color=C3, alpha=0.6); sns.regplot(x=zx, y=zy, scatter=False, color=C2, ax=a, line_kws={"lw": 1.8})
    a.set_title(f"{ttl}\nSpearman rho = {R['spearman_' + c + '_units']['rho']:.2f}"); a.set_xlabel(f"{c} (z-score within product)"); a.set_ylabel("units (z-score)" if c == "review_count" else "")
fig.tight_layout(); fig.savefig("figures_w4/fig_reviews.png", dpi=200); plt.close(fig)

# ============================================================ 7. multivariate
num = ["units", "price", "discount", "search_index", "search_lag1", "search_lag2", "search_lag3", "units_lag1", "units_lag2", "units_lag3", "month_sin", "month_cos"]
d = panel.dropna(subset=num).copy()
dz = d[num].copy()
for c in num:
    if c not in ("month_sin", "month_cos"): dz[c] = (d[c] - d.groupby("Product Name")[c].transform("mean")) / d.groupby("Product Name")[c].transform("std")
corr = dz.corr(method="spearman")
fig, ax = plt.subplots(figsize=(6.0, 4.9))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6.5}, linewidths=0.4, linecolor="white", cbar_kws={"shrink": 0.7, "label": "Spearman rho"}, ax=ax)
ax.tick_params(labelsize=7.5); fig.tight_layout(); fig.savefig("figures_w4/fig_corr_matrix.png", dpi=200); plt.close(fig)
R["corr_units_search_lag2"] = round(float(corr.loc["units", "search_lag2"]), 3)
R["corr_units_units_lag1"] = round(float(corr.loc["units", "units_lag1"]), 3)
R["corr_search_lag1_lag2"] = round(float(corr.loc["search_lag1", "search_lag2"]), 3)
R["corr_units_price"] = round(float(corr.loc["units", "price"]), 3); R["corr_units_discount"] = round(float(corr.loc["units", "discount"]), 3)
# VIF for the lag block
def vif(df):
    out = {}
    for c in df.columns:
        X = np.column_stack([np.ones(len(df)), df.drop(columns=c).values]); y = df[c].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = float(1 / (1 - r2))
    return out
V = vif(dz[["search_index", "search_lag1", "search_lag2", "search_lag3", "units_lag1", "units_lag2", "units_lag3", "price", "discount"]])
R["vif"] = {k: round(v, 2) for k, v in V.items()}
# PCA on the within-product standardised block
pc_cols = [c for c in num if c != "units"]
pca = PCA().fit(dz[pc_cols].values)
R["pca_explained"] = [round(float(x), 3) for x in pca.explained_variance_ratio_[:6]]
R["pca_n_for_80"] = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.80) + 1)
load = pd.DataFrame(pca.components_[:2].T, index=pc_cols, columns=["PC1", "PC2"])
R["pca_top_loadings_pc1"] = load["PC1"].abs().sort_values(ascending=False).head(4).round(2).to_dict()
R["pca_top_loadings_pc2"] = load["PC2"].abs().sort_values(ascending=False).head(4).round(2).to_dict()
fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.9))
ev = pca.explained_variance_ratio_
ax[0].bar(range(1, len(ev) + 1), ev, color=C1); ax[0].plot(range(1, len(ev) + 1), np.cumsum(ev), color=C2, marker="o", ms=3.5)
ax[0].set_xlabel("principal component"); ax[0].set_ylabel("share of variance"); ax[0].set_title("Scree plot (bars) and cumulative (line)")
sns.heatmap(load.round(2), cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, cbar=False, linewidths=0.4, linecolor="white", ax=ax[1]); ax[1].tick_params(labelsize=7.5); ax[1].set_title("Loadings of the first two components")
fig.tight_layout(); fig.savefig("figures_w4/fig_pca.png", dpi=200); plt.close(fig)

# ============================================================ save
panel.to_csv("processed/panel_with_residuals.csv", index=False)
json.dump(R, open("processed/eda_results.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in R.items() if k not in ("profile", "seasonal_index")}, indent=1, default=str))
