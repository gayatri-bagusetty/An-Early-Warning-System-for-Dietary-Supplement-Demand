"""Figures for the feature-engineering and model-comparison sections."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTE, GRID = "#1F2933", "#6B7785", "#E3E7EB"
C1, C2, C3, C4 = "#2F6F6B", "#C8553D", "#8A94A6", "#E2A93B"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
                     "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("processed/model_compare.json"))
fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True)
lines = [("S1 sales memory", "S1 memory only", C3, "o", "-"), ("S2 + calendar and season", "S2 + calendar and season", C1, "o", "-"),
         ("S4 + engineered search", "S4 + engineered search", C4, "s", "-"), ("S6 + price and promotion", "S6 + price and promotion", C2, "^", "-"),
         ("seasonal_naive", "Seasonal naive (same month last year)", INK, None, "--")]
for ax, origin, ttl in [(axs[0], "2022-12-01", "Trained to Dec 2022, tested Jan to Dec 2023"), (axs[1], "2023-12-01", "Trained to Dec 2023, tested Jan 2024 to Apr 2025")]:
    for key, lab, col, mk, ls in lines:
        y = [R["runs"][f"{origin}|h{h}"][key] for h in (1, 2, 3)]
        ax.plot([1, 2, 3], y, color=col, marker=mk, ls=ls, lw=1.8, ms=5, label=lab)
    lv = [R["runs"][f"{origin}|h{h}"]["last_value"] for h in (1, 2, 3)]
    ax.text(0.97, 0.04, f"Last value ('rearview mirror'): {lv[0]:.0f}% to {lv[2]:.0f}%, off the scale", transform=ax.transAxes, ha="right", color=C2, fontsize=7.5)
    ax.set_title(ttl, fontsize=8.5, color=INK); ax.set_xticks([1, 2, 3]); ax.set_xlabel("forecast horizon (months ahead)"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_ylim(4.6, 7.4)
axs[0].set_ylabel("error of demand-rate forecast (MAPE, %)")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig("figures_w4/fig_model_compare.png", dpi=200); plt.close(fig)

# cyclical encoding picture
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
m = np.arange(1, 13); ang = 2 * np.pi * m / 12
ax[0].plot(m, m, color=C3, marker="o", ms=4); ax[0].set_xticks(m); ax[0].set_xlabel("month number"); ax[0].set_ylabel("value fed to a model")
ax[0].annotate("", xy=(12, 12), xytext=(12, 1), arrowprops=dict(arrowstyle="<->", color=C2, lw=1))
ax[0].text(11.6, 3.0, "Dec = 12, Jan = 1:\ndistance 11", fontsize=8, color=C2, ha="right")
ax[0].set_title("Month as a plain number", fontsize=9.5, color=INK)
ax[1].plot(np.sin(ang), np.cos(ang), color=C3, marker="o", ms=4)
lab = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
for i in range(12): ax[1].text(np.sin(ang[i]) * 1.2, np.cos(ang[i]) * 1.2, lab[i], ha="center", va="center", fontsize=8, color=INK)
ax[1].set_xlim(-1.5, 1.5); ax[1].set_ylim(-1.5, 1.5); ax[1].set_aspect("equal"); ax[1].set_xlabel("sin(2π·month/12)"); ax[1].set_ylabel("cos(2π·month/12)")
ax[1].set_title("Month as sine and cosine", fontsize=9.5, color=INK)
ax[1].text(0, 0, "Dec to Jan\ndistance 0.52", ha="center", va="center", fontsize=8, color=C1)
fig.tight_layout(); fig.savefig("figures_w4/fig_cyclical.png", dpi=200); plt.close(fig)
print("ok")
