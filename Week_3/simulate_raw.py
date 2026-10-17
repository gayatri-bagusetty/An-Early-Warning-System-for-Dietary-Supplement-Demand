"""Simulate raw (dirty) data that mimics the three Week 2 sources.

The goal is NOT realistic sales numbers; it is to reproduce the *kinds* of defects
expected in public beauty & wellness data so the cleaning pipeline can be tested.
Every injected defect is recorded in raw/injection_manifest.json, so the pipeline's
detections can be checked against ground truth.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path("raw")
OUT.mkdir(exist_ok=True)

PRODUCTS = {
    "Green Tea Extract": "Herbal",
    "Iron Supplement": "Minerals",
    "Electrolyte Powder": "Sports Nutrition",
    "Biotin": "Vitamins",
    "Ashwagandha": "Herbal",
    "Vitamin D3": "Vitamins",
    "Omega-3": "Fatty Acids",
    "Magnesium": "Minerals",
}
LOCS = ["USA", "UK", "Canada"]
PLATS = ["Amazon", "Walmart", "iHerb"]
mondays = pd.date_range("2020-01-06", "2025-04-28", freq="W-MON")
manifest = {}

# ---------------------------------------------------------------- sales (S1)
# ~16 records per week, as documented in Week 2
rows = []
season = {p: RNG.uniform(0.1, 0.35) for p in PRODUCTS}
phase = {p: RNG.uniform(0, 2 * np.pi) for p in PRODUCTS}
base = {p: RNG.uniform(90, 220) for p in PRODUCTS}
combos = [(p, l, pl) for p in PRODUCTS for l in LOCS for pl in PLATS]
for d in mondays:
    n_rec = int(min(RNG.poisson(15.8), len(combos)))
    for ci in RNG.choice(len(combos), n_rec, replace=False):        # unique key per week
        p, l, pl = combos[ci]
        t = (d.dayofyear / 365.25) * 2 * np.pi
        mu = base[p] * (1 + season[p] * np.sin(t + phase[p])) * (1 + 0.0004 * (d - mondays[0]).days / 7)
        units = max(int(RNG.normal(mu, mu * 0.12)), 1)
        price = round(RNG.uniform(18, 45), 2)
        disc = round(RNG.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20]), 2)
        rows.append(dict(Date=d, **{"Product Name": p}, Category=PRODUCTS[p], **{"Units Sold": units},
                         Price=price, Revenue=round(units * price, 2), Discount=disc,
                         **{"Units Returned": int(RNG.binomial(units, 0.03))},
                         Location=l, Platform=pl))
sales = pd.DataFrame(rows)
clean_truth = sales.copy()          # kept only to measure recovery accuracy
n0 = len(sales)
sales = sales.astype(object)

def pick(n, frac):
    return RNG.choice(np.arange(n0), size=int(n * frac), replace=False)

# 1 Mixed date formats (ISO, "Jan 06, 2020", ambiguous dd/mm/yyyy)
idx = pick(n0, 0.12)
sales.loc[idx[: len(idx) // 2], "Date"] = [d.strftime("%b %d, %Y") for d in clean_truth.loc[idx[: len(idx) // 2], "Date"]]
sales.loc[idx[len(idx) // 2:], "Date"] = [d.strftime("%d/%m/%Y") for d in clean_truth.loc[idx[len(idx) // 2:], "Date"]]
sales.loc[~sales.index.isin(idx), "Date"] = [d.strftime("%Y-%m-%d") for d in clean_truth.loc[~sales.index.isin(idx), "Date"]]
manifest["mixed_date_formats"] = len(idx)

# 2 Product-name inconsistencies
idx = pick(n0, 0.08)
variants = lambda s: RNG.choice([s.lower(), s.upper(), " " + s + " ", s.replace(" ", "-"), s.replace("-", " ")])
sales.loc[idx, "Product Name"] = [variants(s) for s in sales.loc[idx, "Product Name"]]
manifest["inconsistent_names"] = len(idx)

# 3 Location / platform spelling
idx = pick(n0, 0.05)
_alias = {"USA": ["US", "United States", "usa"], "UK": ["U.K.", "uk"], "Canada": ["CA", "canada"]}
sales.loc[idx, "Location"] = [RNG.choice(_alias[l]) for l in sales.loc[idx, "Location"]]   # variants of the TRUE value
_ii = pick(n0, 0.08)
_ii = [i for i in _ii if sales.loc[i, "Platform"] == "Amazon"]
sales.loc[_ii, "Platform"] = "amazon.com"        # alias applied only to true Amazon rows
manifest["inconsistent_locations"] = len(idx)

# 4 Price as text with currency symbol, units with thousands separator / "N/A"
idx = pick(n0, 0.05)
sales.loc[idx, "Price"] = ["$" + str(p) for p in sales.loc[idx, "Price"]]
manifest["price_as_text"] = len(idx)
idx = pick(n0, 0.02)
sales.loc[idx, "Units Sold"] = "N/A"
manifest["units_text_NA"] = len(idx)

# 5 Missing values (units, price, discount, revenue)
miss = {}
for col, frac in [("Units Sold", 0.03), ("Price", 0.025), ("Discount", 0.04), ("Revenue", 0.02), ("Units Returned", 0.03)]:
    ii = pick(n0, frac)
    sales.loc[ii, col] = np.nan
    miss[col] = len(ii)
manifest["missing_injected"] = miss

# 6 Discount on wrong scale (percent instead of fraction)
idx = pick(n0, 0.03)
sales.loc[idx, "Discount"] = [x * 100 if pd.notna(x) else x for x in sales.loc[idx, "Discount"]]
manifest["discount_percent_scale"] = len(idx)

# 7 Impossible values
idx = pick(n0, 0.004)
sales.loc[idx, "Units Sold"] = -abs(clean_truth.loc[idx, "Units Sold"])
manifest["negative_units"] = len(idx)
idx = pick(n0, 0.004)
sales.loc[idx, "Units Returned"] = clean_truth.loc[idx, "Units Sold"] * 2
manifest["returns_exceed_units"] = len(idx)

# 8 Outliers: spikes (x8, e.g. unit entry error) and a promo spike (x2.5, genuine)
idx = pick(n0, 0.006)
sales.loc[idx, "Units Sold"] = [v * 8 if isinstance(v, (int, float, np.integer)) and pd.notna(v) else v for v in sales.loc[idx, "Units Sold"]]
manifest["entry_error_spikes_x8"] = len(idx)
idx = pick(n0, 0.01)
_isnum = lambda v: isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v)
idx = [i for i in idx if _isnum(sales.loc[i, "Units Sold"]) and _isnum(sales.loc[i, "Revenue"])]
sales.loc[idx, "Units Sold"] = [int(round(v * 2.5)) for v in sales.loc[idx, "Units Sold"]]
sales.loc[idx, "Revenue"] = [round(v * 2.5, 2) for v in sales.loc[idx, "Revenue"]]
# a genuine promotion is real demand, so the ground truth contains it too
clean_truth.loc[idx, "Units Sold"] = [int(round(v * 2.5)) for v in clean_truth.loc[idx, "Units Sold"]]
clean_truth.loc[idx, "Revenue"] = [round(v * 2.5, 2) for v in clean_truth.loc[idx, "Revenue"]]
manifest["promo_spikes_x2.5"] = len(idx)

# 9 Duplicates: exact copies and a repeated key with different values
idx = pick(n0, 0.015)
sales = pd.concat([sales, sales.loc[idx]], ignore_index=True)
manifest["exact_duplicates"] = len(idx)
idx = pick(n0, 0.005)
dup = sales.loc[idx].copy()
dup["Units Sold"] = [v + 5 if isinstance(v, (int, float, np.integer)) and pd.notna(v) else v for v in dup["Units Sold"]]
sales = pd.concat([sales, dup], ignore_index=True)
manifest["repeated_key_conflicts"] = len(idx)

# 10 A non-Monday date (late-recorded)
idx = pick(n0, 0.003)
sales.loc[idx, "Date"] = [(d + pd.Timedelta(days=int(RNG.integers(1, 6)))).strftime("%Y-%m-%d") for d in clean_truth.loc[idx, "Date"]]
manifest["non_monday_dates"] = len(idx)

sales = sales.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle
sales.to_csv(OUT / "sales_raw.csv", index=False)
clean_truth.to_csv(OUT / "_truth_sales_clean.csv", index=False)       # ground truth, not used by the pipeline

# ---------------------------------------------------------- Google Trends (S2)
months = pd.date_range("2020-01-01", "2025-04-01", freq="MS")
for dl in (1, 2, 3):                       # three repeated downloads, as planned in Week 2
    wide = {}
    for p in PRODUCTS:
        t = np.arange(len(months))
        lead = 2   # search leads sales by ~2 months in the simulation
        sig = 50 + 25 * season[p] / 0.35 * np.sin((months.dayofyear.values + 61) / 365.25 * 2 * np.pi + phase[p]) + 0.15 * t
        v = np.clip(sig + RNG.normal(0, 4, len(t)), 0, 100).round().astype(int).astype(object)
        small = RNG.choice(len(t), 3, replace=False)
        v[small] = "<1"
        wide[f"{p.lower()}: (United States)"] = v
    df = pd.DataFrame(wide, index=months.strftime("%Y-%m"))
    df.index.name = "Month"
    with open(OUT / f"trends_download{dl}.csv", "w") as f:
        f.write("Category: All categories\n\n")
        df.to_csv(f)
manifest["trends_downloads"] = 3

# --------------------------------------------------------------- reviews (S4)
n_rev = 9000
rm = pd.date_range("2020-01-01", "2023-09-30", freq="D")
ts = RNG.choice(rm, n_rev)
titles = [f"{RNG.choice(list(PRODUCTS))} capsules {RNG.choice(['60 ct', '120 ct', 'bottle'])}" for _ in range(n_rev)]
titles = [("Phone charger cable 2m" if RNG.random() < 0.08 else t) for t in titles]   # irrelevant items
rev = pd.DataFrame(dict(
    asin=[f"B0{RNG.integers(10**7, 10**8)}" for _ in range(n_rev)],
    title=titles,
    rating=RNG.choice([1, 2, 3, 4, 5], n_rev, p=[.06, .05, .09, .22, .58]).astype(float),
    timestamp=pd.to_datetime(ts).astype("datetime64[ms]").astype("int64"),     # Unix milliseconds (unit made explicit)
    user_id=[f"U{RNG.integers(10**5, 10**6)}" for _ in range(n_rev)],
    verified_purchase=RNG.choice([True, False], n_rev, p=[.8, .2]),
))
rev["text"] = RNG.choice(["Works well", "Not noticeable", "Great value", "Upset stomach", "Would buy again"], n_rev)
ii = RNG.choice(n_rev, 180, replace=False); rev.loc[ii, "rating"] = np.nan
ii = RNG.choice(n_rev, 40, replace=False); rev.loc[ii, "rating"] = RNG.choice([0, 6, 7], 40)
rev = pd.concat([rev, rev.sample(220, random_state=3)], ignore_index=True)       # duplicate reviews
rev.to_json(OUT / "reviews_raw.jsonl", orient="records", lines=True)
manifest["review_duplicates"] = 220
manifest["review_rating_missing"] = 180
manifest["review_rating_out_of_range"] = 40

with open(OUT / "injection_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, default=int)
print(json.dumps(manifest, indent=2, default=int))
print("sales rows:", len(sales), "| reviews rows:", len(rev))
