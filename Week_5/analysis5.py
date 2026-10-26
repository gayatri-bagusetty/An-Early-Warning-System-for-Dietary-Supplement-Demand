"""Week 5 - extra paired comparisons, bias checks and figures, computed from processed/predictions_*.csv and models5_results.json."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import metrics5 as M
R = pd.read_csv("processed/predictions_rolling.csv", parse_dates=["origin", "month"]); R["year"] = R["month"].dt.year
res = json.load(open("processed/models5_results.json")); out = {}
def pair(a, b, h):
    x = R[(R.model == a) & (R.h == h)].sort_values(["origin", "product"]).reset_index(drop=True); y = R[(R.model == b) & (R.h == h)].sort_values(["origin", "product"]).reset_index(drop=True)
    d = (x.ape - y.ape).values; m, lo, hi = M.block_bootstrap_diff(x.ape.values, y.ape.values, x.month.astype(str).values)
    return {"diff_pp": m * 100, "lo": lo * 100, "hi": hi * 100, "p": float(stats.wilcoxon(d).pvalue), "share_a_better": float((d < 0).mean())}
T = "Ridge (F2 + trend)"
out["pairs"] = {f"{a} vs {b}": {f"h{h}": pair(a, b, h) for h in (1, 2, 3)} for a, b in [
    (T, "Ridge (F2)"), ("Ridge (F2 + trend + raw search)", T), ("Random forest (F2)", T), ("Gradient boosting (F2)", T), ("Decision tree (F2)", T), ("Seasonal naive", T), ("Last value", T)]}
bm = lambda n: {int(k): M.bias_pct(g.true, g.pred) for k, g in R[(R.model == n) & (R.h == 2)].assign(m=lambda d: d.month.dt.month).groupby("m")}
out["bias_month"] = {n: bm(n) for n in ["Ridge (F2)", T]}
out["bias_year"] = {n: {str(y): M.bias_pct(g.true, g.pred) for y, g in R[(R.model == n) & (R.h == 2)].groupby("year")} for n in ["Ridge (F2)", T]}
out["by_year"] = {n: {str(y): M.mape(g.true, g.pred) for y, g in R[(R.model == n) & (R.h == 2)].groupby("year")} for n in ["Last value", "Seasonal naive", "Ridge (F2)", T]}
out["mape_prod"] = {n: {k: M.mape(g.true, g.pred) for k, g in R[(R.model == n) & (R.h == 2)].groupby("product")} for n in ["Last value", "Seasonal naive", "Ridge (F2)", T, "Random forest (F2)"]}
json.dump(out, open("processed/analysis5.json", "w"), indent=1)
# ---------------- figures
INK, C1, C2, C3, C4 = "#222B33", "#2C6E68", "#C8553D", "#8A93A6", "#E3A93A"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#C9D3D6", "text.color": INK, "axes.labelcolor": INK, "xtick.color": "#5B6770", "ytick.color": "#5B6770"})
reg = res["regression"]
names = ["Last value", "Seasonal naive", "Decision tree (F2)", "Gradient boosting (F2)", "Random forest (F2)", "OLS (F2)", "Ridge (F2)", T]
lab = ["Last value", "Seasonal naive", "Decision tree", "Gradient boosting", "Random forest", "Linear regression (OLS)", "Ridge", "Ridge + trend"]
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2))
y = np.arange(len(names)); cols = [C2, C3, C3, C3, C3, C3, C1, C1]
for j, (key, title, unit) in enumerate([("MAPE", "Error (MAPE, %)", ""), ("RMSE_units", "Error (RMSE, units a month)", "")]):
    v = [reg[n]["h2"][key] for n in names]
    ax[j].barh(y, v, color=cols, height=0.62); ax[j].set_yticks(y); ax[j].set_yticklabels(lab if j == 0 else [], fontsize=8); ax[j].invert_yaxis(); ax[j].set_title(title, fontsize=9.5)
    for yi, vi in zip(y, v): ax[j].text(vi + max(v) * 0.01, yi, f"{vi:.1f}" if key == "MAPE" else f"{vi:.0f}", va="center", fontsize=7.5)
    ax[j].set_xlim(0, max(v) * 1.15)
fig.tight_layout(); fig.savefig("figures_w5/fig_model_bars.png", dpi=200); plt.close(fig)
vc = res["validation_curves"]
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))
for a, (k, ttl, xl) in zip(ax, [("tree_depth", "Decision tree", "tree depth"), ("ridge_alpha", "Ridge", "penalty strength (log scale)"), ("gb_rounds", "Gradient boosting", "boosting rounds")]):
    xs = [float(x) for x in vc[k]]; tr_ = [vc[k][x]["train"] for x in vc[k]]; te_ = [vc[k][x]["test"] for x in vc[k]]
    a.plot(xs, tr_, color=C3, marker="o", ms=3, label="training"); a.plot(xs, te_, color=C1, marker="o", ms=3, label="test")
    a.set_title(ttl, fontsize=9.5); a.set_xlabel(xl, fontsize=8); a.set_ylim(0, 22)
    if k == "ridge_alpha": a.set_xscale("log"); a.set_xlim(0.05, 3000)
    if k == "tree_depth": a.set_ylabel("MAPE (%)")
ax[0].legend(frameon=False, fontsize=7.5); fig.tight_layout(); fig.savefig("figures_w5/fig_validation_curves.png", dpi=200); plt.close(fig)
d = pd.read_csv("processed/predictions_surge.csv", parse_dates=["month"]); from sklearn.metrics import precision_recall_curve
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
for n, c in [("Logistic regression", C1), ("Random forest", C4), ("Gradient boosting", C3)]:
    g = d[(d.model == n) & (d.h == 2)]; p, r_, _ = precision_recall_curve(g.y, g.score); ax[0].plot(r_, p, color=c, label=n)
base = d[(d.model == "Logistic regression") & (d.h == 2)].y.mean(); ax[0].axhline(base, color=C2, ls="--", lw=1); ax[0].text(0.02, base + 0.02, f"base rate {base:.2f} (no skill)", color=C2, fontsize=7.5)
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_ylim(0, 1.03); ax[0].legend(frameon=False, fontsize=7.5, loc="lower left"); ax[0].set_title("Surge alert, 2 months ahead", fontsize=9.5)
cl = res["classification"]; ms = ["Persistence (last month a surge)", "Seasonal rule (surge a year ago)", "Logistic regression", "Random forest"]; ll = ["Persistence", "Seasonal rule", "Logistic reg.", "Random forest"]
w = 0.26
for i, (k, c) in enumerate([("precision", C3), ("recall", C4), ("F1", C1)]):
    ax[1].bar(np.arange(4) + (i - 1) * w, [cl[m]["h2"][k] for m in ms], w, color=c, label=k)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(ll, fontsize=7.5); ax[1].set_ylim(0, 1.05); ax[1].legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left"); ax[1].set_title("Precision, recall, F1 (h = 2)", fontsize=9.5)
fig.tight_layout(); fig.savefig("figures_w5/fig_surge.png", dpi=200); plt.close(fig)
g = R[(R.model == T) & (R.h == 2) & (R["product"] == "Vitamin D3")].sort_values("month"); l = R[(R.model == "Last value") & (R.h == 2) & (R["product"] == "Vitamin D3")].sort_values("month")
fig, ax = plt.subplots(figsize=(7.6, 2.8)); ax.plot(g.month, g.true, color=INK, lw=1.8, label="actual"); ax.plot(g.month, g.pred, color=C1, lw=1.6, label="Ridge + trend (made 2 months earlier)"); ax.plot(l.month, l.pred, color=C2, lw=1.2, ls="--", label="last value (made 2 months earlier)")
ax.set_ylabel("units per record"); ax.set_title("Vitamin D3: rolling forecasts vs actual", fontsize=9.5); fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=3); fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("figures_w5/fig_vitd.png", dpi=200); plt.close(fig)
r = lambda x: round(x, 2)
for k, v in out["pairs"].items(): print(k, " | ".join(f"{h}: {r(x['diff_pp']):+} [{r(x['lo']):+},{r(x['hi']):+}] p={x['p']:.3f}" for h, x in v.items()))
print(out["bias_year"], {k: round(v, 1) for k, v in out["bias_month"][T].items()}); print(out["by_year"])
for n in ["Ridge (F2)", T, "Ridge (F2 + trend + raw search)"]: print(n, {h: (r(reg[n][h]["MAPE"]), r(reg[n][h]["bias"]), r(reg[n][h]["MAE_units"]), r(reg[n][h]["RMSE_units"])) for h in ["h1", "h2", "h3"]})
