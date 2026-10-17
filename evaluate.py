"""Evaluate the pipeline against the hidden ground truth and a naive baseline; make figures."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

Path("figures").mkdir(exist_ok=True)
INK, MUTE, GRID = "#1F2933", "#6B7785", "#E3E7EB"
C1, C2, C3, C4 = "#2F6F6B", "#C8553D", "#8A94A6", "#E2A93B"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.spines.top": False, "axes.spines.right": False})

truth = pd.read_csv("raw/_truth_sales_clean.csv", parse_dates=["Date"])
clean = pd.read_csv("processed/sales_clean_weekly.csv", parse_dates=["Date"])
panel = pd.read_csv("processed/panel_monthly.csv", parse_dates=["month"])
res = {}

# ---- record-level accuracy against truth (key = date, product, location, platform)
K = ["Date", "Product Name", "Location", "Platform"]
m = clean.merge(truth[K + ["Units Sold", "Price"]], on=K, suffixes=("", "_true"), how="left")
res["rows_matched_to_truth"] = int(m["Units Sold_true"].notna().sum())
res["rows_clean"] = len(m)
m["exact"] = m["Units Sold"] == m["Units Sold_true"]
res["units_exact_pct"] = float(m["exact"].mean() * 100)
imp = m[m["was_missing_units_sold"]]
res["imputed_units_rows"] = len(imp)
res["imputed_units_exact_pct"] = float(imp["exact"].mean() * 100)
res["imputed_units_median_abs_pct_err"] = float(((imp["Units Sold"] - imp["Units Sold_true"]).abs() / imp["Units Sold_true"]).median() * 100)
pm_ = clean.merge(truth[K + ["Price"]], on=K, suffixes=("", "_true"))
pm_ = pm_[pm_["was_missing_price"]]
res["price_missing_rows"] = len(pm_)
pe = (pm_["Price"] - pm_["Price_true"]).abs() / pm_["Price_true"]
res["price_exact_pct"] = float((pe < 0.005).mean() * 100)
res["price_median_abs_pct_err"] = float(pe.median() * 100)
res["price_within5_pct"] = float((pe < 0.05).mean() * 100)
bad = m[~m["exact"]]
res["units_wrong_rows"] = len(bad)
res["units_wrong_flagged_pct"] = float((bad["flag_outlier"] | bad["was_missing_units_sold"] | bad["flag_sign_fixed"]).mean() * 100) if len(bad) else 100.0
res["rows_lost_vs_truth"] = int(len(truth) - m["Units Sold_true"].notna().sum())

# ---- naive baseline: load and aggregate with no cleaning
raw = pd.read_csv("raw/sales_raw.csv", dtype=str, keep_default_na=False)
nv = pd.DataFrame({
    "date": pd.to_datetime(raw["Date"], format="mixed", errors="coerce"),
    "product": raw["Product Name"],
    "units": pd.to_numeric(raw["Units Sold"], errors="coerce"),
})
nv["month"] = nv["date"].dt.to_period("M").dt.to_timestamp()
naive_m = nv.groupby(["product", "month"])["units"].sum().rename("naive").reset_index()
naive_prods = nv["product"].nunique()
res["naive_distinct_product_names"] = int(naive_prods)
tm = truth.assign(month=truth["Date"].dt.to_period("M").dt.to_timestamp()).groupby(["Product Name", "month"])["Units Sold"].sum().rename("truth").reset_index()
cm = panel[["Product Name", "month", "units"]]
cmp_ = tm.merge(cm, on=["Product Name", "month"], how="left").merge(naive_m, left_on=["Product Name", "month"], right_on=["product", "month"], how="left")
cmp_["naive"] = cmp_["naive"].fillna(0)
mape = lambda a, b: float((np.abs(a - b) / b).mean() * 100)
res["monthly_MAPE_naive_pct"] = mape(cmp_["naive"], cmp_["truth"])
res["monthly_MAPE_clean_pct"] = mape(cmp_["units"], cmp_["truth"])

# ---- does cleaning recover the planted lead-lag structure? (simulation: search leads sales by ~2 months)
def lag_corr(units, search, k):
    out = []
    for p, g in pd.DataFrame({"p": units["Product Name"], "u": units["u"], "s": search}).groupby("p"):
        out.append(g["u"].corr(g["s"].shift(k)))
    return float(np.nanmean(out))
tmp = cmp_.merge(panel[["Product Name", "month", "search_index"]], on=["Product Name", "month"])
rows = []
for k in range(0, 5):
    rows.append(dict(lag=k,
                     truth=lag_corr(tmp.rename(columns={"truth": "u"}), tmp["search_index"], k),
                     clean=lag_corr(tmp.rename(columns={"units": "u"}).assign(), tmp["search_index"], k),
                     naive=lag_corr(tmp.rename(columns={"naive": "u"}).assign(), tmp["search_index"], k)))
# note: rename above drops duplicates safely because each call renames a different column
lagdf = pd.DataFrame(rows)
res["lag_corr"] = lagdf.round(3).to_dict("records")

json.dump(res, open("processed/evaluation.json", "w"), indent=2)
print(json.dumps(res, indent=2))

# ================================================================ FIGURES
# Fig A: missing values before / after
summ = json.load(open("processed/run_summary.json"))
cols = ["Units Sold", "Price", "Revenue", "Discount", "Units Returned"]
b = [summ["missing_before_step5"][c] for c in cols]
fig, ax = plt.subplots(figsize=(6.6, 3.0))
x = np.arange(len(cols))
bars = ax.bar(x, b, color=C2, width=0.55)
for r_, v in zip(bars, b):
    ax.text(r_.get_x() + r_.get_width() / 2, v + 3, str(v), ha="center", color=INK, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(cols); ax.set_ylabel("Records with missing value")
ax.set_title("Missing values in the raw sales file, by column (after type cleaning)", loc="left", fontsize=10, color=INK)
ax.text(0.99, 0.93, "After step 5: 0 missing in every column", transform=ax.transAxes, ha="right", color=C1, fontsize=9)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/fig_missing.png", dpi=200); plt.close(fig)

# Fig B: outliers for one product: raw-with-errors vs cleaned
prod = "Biotin"
raw2 = pd.DataFrame({"date": pd.to_datetime(raw["Date"], format="mixed", errors="coerce"), "product": raw["Product Name"].str.strip().str.lower(),
                     "units": pd.to_numeric(raw["Units Sold"].str.replace(",", ""), errors="coerce")})
rp = raw2[raw2["product"] == prod.lower()].dropna()
rp = rp[rp["units"] > 0]
cp = clean[clean["Product Name"] == prod]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
axs[0].scatter(rp["date"], rp["units"], s=6, color=C3, alpha=.7)
hi = rp[rp["units"] > rp["units"].median() * 3]
axs[0].scatter(hi["date"], hi["units"], s=22, color=C2, label="entry-error spikes")
axs[0].set_title("Before: raw units (Biotin)", loc="left", fontsize=10, color=INK)
axs[0].legend(frameon=False, loc="upper right", fontsize=8)
axs[1].scatter(cp["Date"], cp["Units Sold"], s=6, color=C3, alpha=.7)
g = cp[cp["flag_units_reconciled"]]; ge = cp[cp["flag_outlier"] & ~cp["flag_units_reconciled"]]
axs[1].scatter(g["Date"], g["Units Sold"], s=22, color=C1, label="corrected (Revenue/Price)")
axs[1].scatter(ge["Date"], ge["Units Sold"], s=22, color=C4, label="kept: genuine spike")
axs[1].set_title("After cleaning", loc="left", fontsize=10, color=INK)
axs[1].legend(frameon=False, loc="upper right", fontsize=8)
for a in axs:
    a.set_xlabel(""); a.yaxis.grid(True, color=GRID); a.set_axisbelow(True); a.tick_params(axis="x", labelsize=8)
axs[0].set_ylabel("Weekly units per record")
fig.tight_layout(); fig.savefig("figures/fig_outliers.png", dpi=200); plt.close(fig)

# Fig C: monthly series for one product: truth vs naive vs clean
prod = "Vitamin D3"
t = cmp_[cmp_["Product Name"] == prod].sort_values("month")
fig, ax = plt.subplots(figsize=(6.8, 3.0))
ax.plot(t["month"], t["truth"], color=INK, lw=1.8, label="Ground truth (hidden)")
ax.plot(t["month"], t["naive"], color=C2, lw=1.2, label="Naive load + sum")
ax.plot(t["month"], t["units"], color=C1, lw=1.2, ls="--", label="After pipeline")
ax.set_ylabel("Units per month"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("Monthly units, Vitamin D3: naive loading vs the cleaning pipeline", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(); fig.savefig("figures/fig_monthly.png", dpi=200); plt.close(fig)

# Fig D: lead-lag correlation
fig, ax = plt.subplots(figsize=(6.2, 3.0))
w = 0.27; x = lagdf["lag"].values
ax.bar(x - w, lagdf["truth"], w, color=INK, label="Ground truth")
ax.bar(x, lagdf["naive"], w, color=C2, label="Naive")
ax.bar(x + w, lagdf["clean"], w, color=C1, label="Cleaned")
ax.set_xlabel("Search interest lag (months)"); ax.set_ylabel("Mean correlation with units")
ax.set_title("Cleaning preserves the lead-lag signal (simulated data)", loc="left", fontsize=10, color=INK)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
fig.tight_layout(); fig.savefig("figures/fig_lag.png", dpi=200); plt.close(fig)

# Fig E: flowchart
fig, ax = plt.subplots(figsize=(7.4, 9.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 132); ax.axis("off")
def box(x, y, w, h, title, sub, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.6, color=MUTE)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=MUTE, lw=1.1))
src = [(2, "Sales (weekly)\nKaggle CSV"), (35, "Google Trends\n3 CSV downloads"), (68, "Amazon reviews\nJSONL")]
for x, t in src:
    ax.add_patch(FancyBboxPatch((x, 123), 30, 8, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#EEF1F4", ec=C3, lw=1.2))
    ax.text(x + 15, 127, t, ha="center", va="center", fontsize=8.2, color=INK)
    arrow(x + 15, 122.3, x + 15, 118.7)
box(2, 111, 96, 7, "0  INGEST", "read-only raw files, SHA-256 fingerprint, schema + row-count check", "#EAF3F2", C1)
ys = [96, 85, 74, 63, 52, 41, 30]
sales_steps = [("1  Types & dates", "numbers, ISO dates"), ("2  Text standardise", "vocabulary, Monday snap"),
               ("3  De-duplicate", "exact + same-key"), ("4  Validity rules", "scale, sign, returns"),
               ("5  Missing values", "recover > impute"), ("6  Outliers", "robust z + reconcile"),
               ("7  Weekly to monthly", "weighted price/discount")]
for (t, s_), y in zip(sales_steps, ys): box(2, y, 30, 8, t, s_, "#FFFFFF", C1)
for i in range(6): arrow(17, ys[i] - 0.5, 17, ys[i + 1] + 8.6)
for x in (17, 50, 83): arrow(x, 110.3, x, 104.9)
for x, steps, ec in [(35, [("8a  Skip header, parse", "'<1' cross-checked"), ("8b  Average downloads", "reduce sampling noise")], C3),
                     (68, [("9a  Dedupe, rating 1-5", "ms timestamp to month"), ("9b  Match to product", "verified purchases")], C3)]:
    for (t, s_), y in zip(steps, ys[:2]): box(x, y, 30, 8, t, s_, "#FFFFFF", ec)
    arrow(x + 15, ys[0] - 0.5, x + 15, ys[1] + 8.6)
box(2, 14, 96, 8, "10  MERGE, LAG, SPLIT, SCALE", "1:1 validated join | lags within product | time split | z-score fit on train only", "#EAF3F2", C1)
arrow(17, ys[6] - 0.5, 17, 22.9)
arrow(50, ys[1] - 0.5, 50, 22.9); arrow(83, ys[1] - 0.5, 83, 22.9)
box(2, 1, 96, 8, "11  QUALITY GATE + OUTPUTS", "asserts | cleaning_log.csv | run_summary.json | panel_monthly.csv", "#FDF1E3", C4)
arrow(50, 13.5, 50, 9.7)
fig.savefig("figures/fig_flow.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("figures done")
