"""Week 6 figures: validation flowchart, forecast fan chart, alert curves, calibration and cost, feature impact, inventory, stress test, dashboard mock-up."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, precision_recall_curve
D = json.load(open("processed/final6_results.json"))
IV = pd.read_csv("processed/intervals_final_h2.csv", parse_dates=["origin", "month"]); AL = pd.read_csv("processed/alert_lr_h2.csv", parse_dates=["month", "origin"])
R = pd.read_csv("processed/predictions_rolling.csv", parse_dates=["origin", "month"])
INK, C1, C2, C3, C4, GREY = "#222B33", "#2C6E68", "#C8553D", "#8A93A6", "#E3A93A", "#5B6770"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#C9D3D6", "text.color": INK, "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY})
def save(fig, n): fig.savefig(f"figures_w6/{n}.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- 1. validation flowchart
fig, ax = plt.subplots(figsize=(7.6, 5.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")
stages = [("Stage 0", "Data and leakage gate", "Is the data sound? Can a forecast\nsee the future?", "Week 2 gate; 10 automated tests", "Done", C1),
          ("Stage 1", "Rolling back-test", "Does it beat last month\nand last year?", "28 monthly origins; 3 horizons", "Done", C1),
          ("Stage 2", "Slices", "Is it good for every product,\nseason and year?", "8 products; 4 quarters; years", "Done", C1),
          ("Stage 3", "Uncertainty", "Are the error bands and\nalert odds honest?", "80% bands; calibration; cost curve", "Done", C1),
          ("Stage 4", "Stress tests", "What happens when something\nunexpected hits?", "+60% shock; feature removal", "Done", C1),
          ("Stage 5", "Pilot and monitoring", "Does it hold on real,\nnew months?", "Shadow run; monthly bias check", "Planned", C4)]
pos = [(2, 36), (35, 36), (68, 36), (68, 6), (35, 6), (2, 6)]
for (st, ti, q, how, status, col), (x, y) in zip(stages, pos):
    ax.add_patch(FancyBboxPatch((x, y), 30, 25, boxstyle="round,pad=0.4,rounding_size=1.5", fc="white", ec=col, lw=1.6))
    ax.text(x + 1.5, y + 22.2, st, fontsize=8, color=col, fontweight="bold"); ax.text(x + 1.5, y + 17.6, ti, fontsize=9.5, fontweight="bold")
    ax.text(x + 1.5, y + 11.4, q, fontsize=7.4, va="center"); ax.text(x + 1.5, y + 5.3, how, fontsize=7.2, color=GREY, va="center")
    ax.text(x + 28.5, y + 22.2, status, fontsize=7.5, color=col, fontweight="bold", ha="right")
for (x0, y0, x1, y1) in [(32.8, 48.5, 34.8, 48.5), (65.8, 48.5, 67.8, 48.5), (83, 35.5, 83, 31.8), (67.8, 18.5, 65.8, 18.5), (34.8, 18.5, 32.8, 18.5)]:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=GREY, lw=1.3))
ax.text(50, 63.5, "Each stage must pass before the next one is trusted", ha="center", fontsize=9, color=GREY)
save(fig, "fig_validation_flow")

# ---- 2. forecast fan chart (Vitamin D3, 2 months ahead)
v = IV[IV["product"] == "Vitamin D3"].sort_values("month")
fig, ax = plt.subplots(figsize=(7.6, 3.1)); e = v.dropna(subset=["lo"])
ax.fill_between(e.month, e.lo, e.hi, color=C1, alpha=0.18, label="80% error band"); ax.plot(v.month, v.true, color=INK, lw=1.8, label="actual"); ax.plot(v.month, v.pred, color=C1, lw=1.6, label="forecast made 2 months earlier")
ax.set_ylabel("demand rate (units per record)"); ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3); ax.set_ylim(80, 250)
ax.set_title("Vitamin D3: forecast, error band and what actually happened", fontsize=9.5); save(fig, "fig_forecast_fan")

# ---- 3. ROC and precision-recall (logistic regression, h = 2) with the simple rules as points
C = pd.read_csv("processed/predictions_surge.csv", parse_dates=["origin", "month"]); c2 = C[C.h == 2]
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2)); lr = c2[c2.model == "Logistic regression"]
fpr, tpr, _ = roc_curve(lr.y, lr.score); ax[0].plot(fpr, tpr, color=C1, lw=2); ax[0].plot([0, 1], [0, 1], color=C3, ls="--", lw=1)
for n, mk, col, lab in [("Seasonal rule (surge a year ago)", "s", C4, "seasonal rule"), ("Persistence (last month a surge)", "^", C2, "last month a surge")]:
    d = c2[c2.model == n]; tp = ((d.y == 1) & (d.pred == 1)).sum() / (d.y == 1).sum(); fp = ((d.y == 0) & (d.pred == 1)).sum() / (d.y == 0).sum(); ax[0].scatter([fp], [tp], marker=mk, color=col, zorder=3, label=lab)
ax[0].set_xlabel("false alarm rate"); ax[0].set_ylabel("share of surges caught"); ax[0].set_title(f"ROC curve (AUC = {D['alert']['Logistic regression']['h2']['ROC_AUC']:.2f})", fontsize=9.5); ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
p, r, _ = precision_recall_curve(lr.y, lr.score); ax[1].plot(r, p, color=C1, lw=2); ax[1].axhline(lr.y.mean(), color=C2, ls="--", lw=1); ax[1].text(0.02, lr.y.mean() + 0.03, f"no-skill level {lr.y.mean():.2f}", color=C2, fontsize=7.5)
ax[1].set_xlabel("recall (share of surges caught)"); ax[1].set_ylabel("precision (share of alerts that are real)"); ax[1].set_ylim(0, 1.03); ax[1].set_title(f"Precision-recall (AUC = {D['alert']['Logistic regression']['h2']['PR_AUC']:.2f})", fontsize=9.5)
fig.tight_layout(); save(fig, "fig_alert_curves")

# ---- 4. calibration + cost curve
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1)); cal = D["calibration_h2"]
ax[0].plot([0, 1], [0, 1], color=C3, ls="--", lw=1); ax[0].scatter([b["mean_pred"] for b in cal], [b["observed"] for b in cal], s=[max(b["n"], 8) * 2.2 for b in cal], color=C1, alpha=0.85)
for b in cal: ax[0].text(b["mean_pred"] + 0.03, b["observed"] - 0.06, f"n={b['n']}", fontsize=6.8, color=GREY)
ax[0].set_xlabel("probability the model gave"); ax[0].set_ylabel("share that really were surges"); ax[0].set_title("Do the probabilities mean what they say?", fontsize=9.5)
for key, col, lab in [("ratio1", C3, "miss costs 1 false alarm"), ("ratio3", C1, "miss costs 3"), ("ratio5", C2, "miss costs 5")]:
    b = D["cost_curve"][key]; cv = np.array([(r[0], r[1]) for r in b["curve"]]); ax[1].plot(cv[:, 0], cv[:, 1], color=col, lw=1.8, label=lab); ax[1].scatter([b["best_t"]], [b["best_cost"]], color=col, zorder=3)
ax[1].set_xlabel("alert threshold"); ax[1].set_ylabel("total cost (false-alarm units)"); ax[1].set_title("Best threshold depends on the cost of a miss", fontsize=9.5); ax[1].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); save(fig, "fig_calibration_cost")

# ---- 5. feature impact
imp = D["impact"]; names_d = list(imp["drop"].keys()); names_a = list(imp["add"].keys())
order_d = sorted(names_d, key=lambda k: imp["drop"][k]["h2"]["d_mape"]); 
fig, ax = plt.subplots(1, 2, figsize=(7.6, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
def bars(a, items, kind, title, lab_fn):
    y = np.arange(len(items)); vals = [imp[kind][k]["h2"]["d_mape"] for k in items]; lo = [imp[kind][k]["h2"]["d_mape"] - imp[kind][k]["h2"]["lo"] for k in items]; hi = [imp[kind][k]["h2"]["hi"] - imp[kind][k]["h2"]["d_mape"] for k in items]
    cols = [C2 if (imp[kind][k]["h2"]["lo"] > 0) else (C1 if imp[kind][k]["h2"]["hi"] < 0 else C3) for k in items]
    a.barh(y, vals, xerr=[lo, hi], color=cols, height=0.62, error_kw={"ecolor": INK, "lw": 0.9, "capsize": 2}); a.set_yticks(y); a.set_yticklabels([lab_fn(k) for k in items], fontsize=7.4); a.axvline(0, color=INK, lw=0.8); a.set_title(title, fontsize=9); a.set_xlabel("change in 2-month MAPE (points)")
sel_d = [k for k in order_d if not k.startswith("Memory and")] + ["Memory and season together"]
sel_d = sorted(set(sel_d), key=lambda k: imp["drop"][k]["h2"]["d_mape"])
bars(ax[0], sel_d, "drop", "Take one feature away\n(right of zero = the feature was helping)", lambda k: k)
bars(ax[1], names_a, "add", "Add one feature to the final model\n(left of zero = the feature helps)", lambda k: k)
ax[0].set_xscale("symlog", linthresh=1.5); ax[0].set_xlim(-1.5, 20); ax[0].set_xticks([-1, 0, 1, 3, 10]); ax[0].set_xticklabels(["-1", "0", "1", "3", "10"]); fig.tight_layout(); save(fig, "fig_feature_impact")

# ---- 6. inventory
inv = D["inventory"]; ms = [("Last value", "Last month\ncarried forward", C2), ("Seasonal naive", "Same month\nlast year", C3), ("Ridge (F2 + trend)", "Final model", C1)]
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.9))
for a, (key, ttl, fmt) in zip(ax, [("buffer_pct", "Safety buffer needed (%)", "{:.0f}%"), ("excess_units", "Spare units held\nper product-month", "{:.0f}"), ("cost_per_pm", "Illustrative cost per\nproduct-month ($)", "${:,.0f}")]):
    vals = [inv[m][key] for m, _, _ in ms]; a.bar(range(3), vals, color=[c for _, _, c in ms], width=0.65); a.set_xticks(range(3)); a.set_xticklabels([l for _, l, _ in ms], fontsize=7)
    for i, vv in enumerate(vals): a.text(i, vv, fmt.format(vv), ha="center", va="bottom", fontsize=8)
    a.set_title(ttl, fontsize=9); a.set_ylim(0, max(vals) * 1.18)
fig.tight_layout(); save(fig, "fig_inventory")

# ---- 7. stress test
st = D["stress"]; rows = st["rows"]; mo = pd.to_datetime([r["month"] + "-01" for r in rows])
fig, ax = plt.subplots(figsize=(7.6, 3.0)); ax.axvspan(mo[0] - pd.Timedelta(days=15), mo[2] + pd.Timedelta(days=15), color=C4, alpha=0.18); ax.text(mo[1], 305, "surge (+60%)", ha="center", fontsize=8, color="#8a6200")
ax.plot(mo, [r["actual"] for r in rows], color=INK, lw=2, marker="o", ms=3.5, label="actual (with surge)"); ax.plot(mo, [r["ridge"] for r in rows], color=C1, lw=1.8, marker="o", ms=3.5, label="final model forecast")
ax.plot(mo, [r["last"] for r in rows], color=C2, lw=1.4, ls="--", marker="o", ms=3, label="last-value forecast"); ax.set_ylabel("Ashwagandha demand rate"); ax.set_ylim(140, 325)
ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3); ax.set_title("Stress test: a surge nobody could see coming (forecasts made 2 months ahead)", fontsize=9.5); save(fig, "fig_stress")

# ---- 8. dashboard mock-up (real back-test numbers for the last forecastable month)
tgt = pd.Timestamp("2025-04-01"); fin = R[(R.model == "Ridge (F2 + trend)") & (R.h == 2) & (R.month == tgt)].set_index("product")
ivt = IV[IV.month == tgt].set_index("product"); alt = AL[AL.month == tgt].set_index("product")
fig = plt.figure(figsize=(7.6, 6.6)); gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 2.4, 2.2], hspace=0.38, wspace=0.25)
H = D["headline"]["Ridge (F2 + trend)"]["h2"]; AL2 = D["alert"]["Logistic regression"]["h2"]
tiles = [("Typical forecast miss", f"{H['MAPE']:.1f}%", "2 months ahead"), ("Forecasts within ±10%", f"{H['hit10'] * 100:.0f}%", "of product-months"), ("Surges caught", f"{AL2['recall'] * 100:.0f}%", "2 months ahead"), ("Alerts that are real", f"{AL2['precision'] * 100:.0f}%", "when the alert fires")]
for i, (t, big, sub) in enumerate(tiles):
    a = fig.add_subplot(gs[0, i]); a.axis("off"); a.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#EAF3F2", ec="none", transform=a.transAxes))
    a.text(0.5, 0.80, t, ha="center", fontsize=7.6, color=GREY, transform=a.transAxes); a.text(0.5, 0.42, big, ha="center", fontsize=17, fontweight="bold", color=C1, transform=a.transAxes); a.text(0.5, 0.10, sub, ha="center", fontsize=7, color=GREY, transform=a.transAxes)
a = fig.add_subplot(gs[1, :]); e = v.dropna(subset=["lo"]); a.fill_between(e.month, e.lo, e.hi, color=C1, alpha=0.18); a.plot(v.month, v.true, color=INK, lw=1.6); a.plot(v.month, v.pred, color=C1, lw=1.4)
a.set_title("Vitamin D3 demand: forecast (teal), 80% band (shaded), actual (black)", fontsize=8.5, loc="left"); a.set_ylabel("units per record", fontsize=8)
a = fig.add_subplot(gs[2, :]); a.axis("off"); a.set_title(f"Product watchlist for {tgt.strftime('%b %Y')} (forecast made 2 months earlier)", fontsize=8.5, loc="left")
cells = []
for p in sorted(fin.index):
    fu = fin.loc[p, "pred"] * fin.loc[p, "n_records"]; lo = ivt.loc[p, "lo"] * fin.loc[p, "n_records"] if p in ivt.index else np.nan; hi = ivt.loc[p, "hi"] * fin.loc[p, "n_records"] if p in ivt.index else np.nan
    pr_ = alt.loc[p, "score"] if p in alt.index else np.nan; status = "ORDER AHEAD" if pr_ >= 0.5 else "normal"
    cells.append([p, f"{fu:,.0f}", f"{lo:,.0f} to {hi:,.0f}", f"{pr_ * 100:.0f}%", status])
tb = a.table(cellText=cells, colLabels=["Product", "Forecast units", "80% band", "Surge chance", "Action"], loc="upper center", cellLoc="center", colWidths=[0.27, 0.17, 0.25, 0.15, 0.16]); tb.auto_set_font_size(False); tb.set_fontsize(7.4); tb.scale(1, 1.45)
for (r_, c_), cell in tb.get_celld().items():
    cell.set_edgecolor("#C9D3D6")
    if r_ == 0: cell.set_facecolor(C1); cell.set_text_props(color="white", fontweight="bold")
    elif c_ == 4 and cells[r_ - 1][4] == "ORDER AHEAD": cell.set_facecolor("#FBE3DC"); cell.set_text_props(fontweight="bold", color=C2)
fig.text(0.01, 0.005, "Mock-up for illustration. Numbers are real back-test values on the simulated data.", fontsize=6.8, color=GREY)
save(fig, "fig_dashboard_mock")
print("figures done")
