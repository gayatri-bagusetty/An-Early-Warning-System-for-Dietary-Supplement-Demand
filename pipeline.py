"""Week 3 - Data cleaning and preprocessing pipeline for the supplement-demand project.

Design rules
  1. Raw files are read-only; every output is written to processed/.
  2. Each stage is a small pure function: DataFrame in -> DataFrame out.
  3. Every stage logs what it changed (rows affected, rule, action) to a cleaning log.
  4. Problems are FLAGGED first; rows are only removed when the rule is unambiguous.
  5. Anything that learns from data (scalers, imputation statistics) is fit on training months only.
"""
import json, hashlib, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd

RAW, OUT = Path("raw"), Path("processed")
OUT.mkdir(exist_ok=True)
LOG = []   # list of dicts: stage, rule, rows_affected, action


def log(stage, rule, n, action):
    LOG.append(dict(stage=stage, rule=rule, rows_affected=int(n), action=action))


# =========================================================== 0. ingest + fingerprint
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# =========================================================== 1. types and formats
MONEY = re.compile(r"[^\d.\-]")

def parse_dates(s: pd.Series) -> pd.Series:
    """Three formats appear. dd/mm/yyyy vs mm/dd/yyyy is ambiguous, but every valid
    record is a Monday, so we keep whichever reading lands on a Monday."""
    s = s.astype("string").str.strip()
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    txt = pd.to_datetime(s, format="%b %d, %Y", errors="coerce")
    dmy = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    mdy = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    slash = dmy.where(dmy.dt.dayofweek == 0, mdy.where(mdy.dt.dayofweek == 0, dmy))
    return iso.fillna(txt).fillna(slash)


def to_number(s: pd.Series) -> pd.Series:
    """'$31.88' -> 31.88 ; '1,234' -> 1234 ; 'N/A' -> NaN."""
    s = s.astype("string").str.strip().replace({"N/A": pd.NA, "n/a": pd.NA, "": pd.NA, "nan": pd.NA})
    return pd.to_numeric(s.str.replace(MONEY, "", regex=True), errors="coerce")


def standardise_types(df):
    n_text_price = df["Price"].astype("string").str.contains(r"\$", na=False).sum()
    n_na_text = df["Units Sold"].astype("string").str.upper().eq("N/A").sum()
    df = df.copy()
    n_noniso = (~df["Date"].astype("string").str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)).sum()
    df["Date"] = parse_dates(df["Date"])
    for c in ["Units Sold", "Price", "Revenue", "Discount", "Units Returned"]:
        df[c] = to_number(df[c]).astype("float64")
    log("1 types", "Price stored as text with currency symbol", n_text_price, "stripped symbol, cast to float")
    log("1 types", "Text 'N/A' in Units Sold", n_na_text, "converted to missing (NaN)")
    log("1 types", "Dates not in ISO format ('Jan 06, 2020' or dd/mm/yyyy)", n_noniso,
        f"parsed to datetime; ambiguous dd/mm resolved with the Monday rule; {df['Date'].isna().sum()} unparseable")
    return df


# =========================================================== 2. text standardisation
CANON = ["Green Tea Extract", "Iron Supplement", "Electrolyte Powder", "Biotin",
         "Ashwagandha", "Vitamin D3", "Omega-3", "Magnesium"]
LOC_MAP = {"usa": "USA", "us": "USA", "united states": "USA", "u.s.": "USA",
           "uk": "UK", "u.k.": "UK", "united kingdom": "UK", "canada": "Canada", "ca": "Canada"}
PLAT_MAP = {"amazon": "Amazon", "amazon.com": "Amazon", "walmart": "Walmart", "iherb": "iHerb"}


def _key(x):   # 'green-tea  extract ' -> 'green tea extract'
    return re.sub(r"[\s\-_]+", " ", str(x).strip().lower())


def standardise_text(df):
    df = df.copy()
    lookup = {_key(c): c for c in CANON}
    before = df["Product Name"].nunique()
    df["Product Name"] = df["Product Name"].map(lambda x: lookup.get(_key(x), np.nan))
    unmapped = df["Product Name"].isna().sum()
    log("2 text", f"Product-name variants ({before} raw spellings -> {df['Product Name'].nunique()} canonical)",
        before - df["Product Name"].nunique(), "case/space/hyphen normalised and mapped to controlled vocabulary")
    log("2 text", "Product names not in vocabulary", unmapped, "set to missing and flagged for review")
    raw_loc = df["Location"].copy()
    df["Location"] = df["Location"].map(lambda x: LOC_MAP.get(_key(x), np.nan))
    log("2 text", "Location spelling variants", (raw_loc != df["Location"]).sum(), "mapped to USA / UK / Canada")
    raw_p = df["Platform"].copy()
    df["Platform"] = df["Platform"].map(lambda x: PLAT_MAP.get(_key(x), np.nan))
    log("2 text", "Platform spelling variants", (raw_p != df["Platform"]).sum(), "mapped to Amazon / Walmart / iHerb")
    return df


def snap_to_monday(df):
    """Records are weekly (Mondays). A late-recorded date is moved back to its Monday BEFORE
    de-duplication, otherwise it can collide with the genuine record of that week."""
    df = df.copy()
    nm = df["Date"].dt.dayofweek != 0
    df["flag_non_monday"] = nm
    df.loc[nm, "Date"] = df.loc[nm, "Date"] - pd.to_timedelta(df.loc[nm, "Date"].dt.dayofweek, unit="D")
    log("2 text", "Dates that are not Mondays", nm.sum(), "snapped back to the Monday of that week and flagged")
    return df


# =========================================================== 3. duplicates
KEY = ["Date", "Product Name", "Location", "Platform"]

def deduplicate(df):
    n = len(df)
    df = df.drop_duplicates()
    log("3 duplicates", "Exact duplicate rows", n - len(df), "dropped (keep first)")
    dup_key = df.duplicated(KEY, keep=False)
    log("3 duplicates", "Same key (date/product/location/platform), different values", dup_key.sum(),
        "flagged; resolved by keeping the row whose Revenue = Units x Price; otherwise the first")
    df = df.copy()
    df["_consistent"] = (df["Revenue"] - df["Units Sold"] * df["Price"]).abs() < 0.01
    df = df.sort_values("_consistent", ascending=False).drop_duplicates(KEY).drop(columns="_consistent")
    return df.sort_values(KEY).reset_index(drop=True)


# =========================================================== 4. validity rules
def fix_and_flag_validity(df):
    df = df.copy()
    # discount on percent scale
    pct = df["Discount"] > 1
    df.loc[pct & (df["Discount"] <= 100), "Discount"] /= 100
    log("4 validity", "Discount recorded as percent (>1)", pct.sum(), "divided by 100")
    bad_disc = (df["Discount"] < 0) | (df["Discount"] > 1)
    df.loc[bad_disc, "Discount"] = np.nan
    # negative units: sign error is more plausible than negative sales -> absolute value, flagged
    neg = df["Units Sold"] < 0
    df["flag_sign_fixed"] = neg
    df.loc[neg, "Units Sold"] = df.loc[neg, "Units Sold"].abs()
    log("4 validity", "Negative Units Sold", neg.sum(), "sign corrected (abs) and flagged; checked against Revenue/Price where available")
    # returns cannot exceed units
    ret = df["Units Returned"] > df["Units Sold"]
    df["flag_returns_gt_units"] = ret
    df.loc[ret, "Units Returned"] = np.nan
    log("4 validity", "Units Returned > Units Sold", ret.sum(), "set to missing (handled in step 5) and flagged")
    return df


# =========================================================== 5. missing values
def handle_missing(df):
    df = df.copy()
    for c in ["Units Sold", "Price", "Revenue", "Discount", "Units Returned"]:
        df[f"was_missing_{c.lower().replace(' ', '_')}"] = df[c].isna()
    miss_before = df[["Units Sold", "Price", "Revenue", "Discount", "Units Returned"]].isna().sum()
    # (a) deterministic recovery from the identity Revenue = Units x Price
    m = df["Units Sold"].isna() & df["Revenue"].notna() & df["Price"].notna() & (df["Price"] > 0)
    df.loc[m, "Units Sold"] = (df.loc[m, "Revenue"] / df.loc[m, "Price"]).round()
    log("5 missing", "Units Sold missing but Revenue and Price known", m.sum(), "recovered as Revenue / Price (exact, no estimation)")
    m = df["Price"].isna() & df["Revenue"].notna() & df["Units Sold"].notna() & (df["Units Sold"] > 0)
    df.loc[m, "Price"] = (df.loc[m, "Revenue"] / df.loc[m, "Units Sold"]).round(2)
    log("5 missing", "Price missing but Revenue and Units known", m.sum(), "recovered as Revenue / Units")
    # (b) Units: rolling median of the same product-location-platform series
    grp = ["Product Name", "Location", "Platform"]
    df = df.sort_values(grp + ["Date"])
    med = df.groupby(grp)["Units Sold"].transform(lambda s: s.rolling(5, center=True, min_periods=2).median())
    m = df["Units Sold"].isna()
    df.loc[m, "Units Sold"] = med[m]
    still = df["Units Sold"].isna()
    med2 = df.groupby("Product Name")["Units Sold"].transform("median")
    df.loc[still, "Units Sold"] = med2[still]
    log("5 missing", "Units Sold still missing after recovery", (m).sum(), "rolling median of same series (window 5); product median as fallback; flagged was_missing_*")
    # (b2) price: Revenue is known but Units had to be estimated -> Revenue / estimated Units
    m = df["Price"].isna() & df["Revenue"].notna() & (df["Units Sold"] > 0)
    df.loc[m, "Price"] = (df.loc[m, "Revenue"] / df.loc[m, "Units Sold"]).round(2)
    log("5 missing", "Price recovered after Units were estimated (Revenue known)", m.sum(), "Revenue / estimated Units (inherits the units estimate's error)")
    # (c) price: median of same product in the same month
    ym = df["Date"].dt.to_period("M")
    pm = df.groupby(["Product Name", ym])["Price"].transform("median")
    m = df["Price"].isna()
    df.loc[m, "Price"] = pm[m].fillna(df.groupby("Product Name")["Price"].transform("median")[m])
    log("5 missing", "Price still missing", m.sum(), "median price of the same product in the same month")
    # (d) discount: missing means unknown, not 'no discount' -> median of product-month, flagged
    dm = df.groupby(["Product Name", ym])["Discount"].transform("median")
    m = df["Discount"].isna()
    df.loc[m, "Discount"] = dm[m].fillna(0)
    log("5 missing", "Discount missing", m.sum(), "median of product-month (0 if none); flagged")
    # (e) revenue and returns are derived/secondary
    m = df["Revenue"].isna()
    df.loc[m, "Revenue"] = (df.loc[m, "Units Sold"] * df.loc[m, "Price"]).round(2)
    log("5 missing", "Revenue missing", m.sum(), "recomputed as Units x Price")
    m = df["Units Returned"].isna()
    rate = (df["Units Returned"].sum() / df.loc[df["Units Returned"].notna(), "Units Sold"].sum())
    df.loc[m, "Units Returned"] = (df.loc[m, "Units Sold"] * rate).round()
    log("5 missing", "Units Returned missing (or invalid)", m.sum(), f"filled with overall return rate x units ({rate:.3%})")
    over = df["Units Returned"] > df["Units Sold"]
    df.loc[over, "Units Returned"] = df.loc[over, "Units Sold"]
    log("5 missing", "Units Returned above (imputed) Units Sold", over.sum(), "capped at Units Sold")
    df.attrs["missing_before"] = miss_before.to_dict()
    df.attrs["missing_after"] = df[["Units Sold", "Price", "Revenue", "Discount", "Units Returned"]].isna().sum().to_dict()
    return df


# =========================================================== 6. outliers
def detect_outliers(df, z=4.0):
    """Robust z-score on log(units) within each product (median/MAD, not mean/SD, because
    outliers inflate the SD and hide themselves). Flag only; do not delete."""
    df = df.copy()
    lu = np.log1p(df["Units Sold"])
    med = lu.groupby(df["Product Name"]).transform("median")
    mad = ((lu - med).abs().groupby(df["Product Name"]).transform("median") * 1.4826).clip(lower=0.05)  # floor: flat series have MAD = 0
    rz = (lu - med) / mad
    df["robust_z"] = rz
    df["flag_outlier"] = rz.abs() > z
    # Cross-field reconciliation: an extreme value is a data-entry error if Revenue/Price disagrees
    # with Units; it is a genuine spike (e.g. promotion) if Revenue/Price agrees with it.
    both = ~df["was_missing_revenue"] & ~df["was_missing_price"] & ~df["was_missing_units_sold"]
    implied = (df["Revenue"] / df["Price"]).round()
    mismatch = df["flag_outlier"] & both & ((implied - df["Units Sold"]).abs() / df["Units Sold"].clip(lower=1) > 0.05)
    genuine = df["flag_outlier"] & both & ~mismatch
    df["flag_units_reconciled"] = mismatch
    df.loc[mismatch, "Units Sold"] = implied[mismatch]
    log("6 outliers", "Extreme units where Revenue/Price disagrees (entry error)", mismatch.sum(),
        "corrected to Revenue/Price and flagged")
    log("6 outliers", "Extreme units confirmed by Revenue/Price (genuine spikes)", genuine.sum(),
        "kept as real; flagged so models can use a capped copy as a sensitivity check")
    # corrected units can fall below already-validated returns, so re-check that rule
    ret = df["Units Returned"] > df["Units Sold"]
    rate = df["Units Returned"].sum() / df["Units Sold"].sum()
    df.loc[ret, "Units Returned"] = (df.loc[ret, "Units Sold"] * min(rate, 0.05)).round()
    df.loc[ret, "flag_returns_gt_units"] = True
    log("6 outliers", "Returns exceed corrected units (re-check after reconciliation)", ret.sum(), "reset to overall return rate x units; flagged")
    lu = np.log1p(df["Units Sold"])
    rz = (lu - med) / mad
    df["robust_z"] = rz
    df["flag_outlier"] = rz.abs() > z
    # Treatment: tiny shares of rows, so winsorise to the product-level bound in a *separate* column
    lo = (med - z * mad); hi = (med + z * mad)
    df["units_winsorised"] = np.expm1(lu.clip(lo, hi)).round()
    log("6 outliers", f"|robust z| > {z} on log(units), by product", df["flag_outlier"].sum(),
        "flagged; original kept in 'Units Sold', capped copy in 'units_winsorised' for sensitivity analysis")
    return df


# =========================================================== 7. weekly -> monthly
def to_monthly(df):
    d = df.copy()
    d["month"] = d["Date"].dt.to_period("M").dt.to_timestamp()
    d["disc_w"] = d["Discount"] * d["Units Sold"]
    d["price_w"] = d["Price"] * d["Units Sold"]
    g = d.groupby(["Product Name", "month"])
    m = g.agg(units=("Units Sold", "sum"), units_wins=("units_winsorised", "sum"), revenue=("Revenue", "sum"),
              returns=("Units Returned", "sum"), n_records=("Units Sold", "size"), n_weeks=("Date", "nunique"),
              n_outliers=("flag_outlier", "sum"), n_imputed=("was_missing_units_sold", "sum"),
              disc_w=("disc_w", "sum"), price_w=("price_w", "sum")).reset_index()
    m["discount"] = m["disc_w"] / m["units"]            # unit-weighted mean, as in Week 2
    m["price"] = m["price_w"] / m["units"]
    m = m.drop(columns=["disc_w", "price_w"])
    # complete the grid: a missing product-month must be explicit, never silently absent
    full = pd.MultiIndex.from_product([sorted(m["Product Name"].unique()),
                                       pd.date_range(m["month"].min(), m["month"].max(), freq="MS")],
                                      names=["Product Name", "month"])
    m = m.set_index(["Product Name", "month"]).reindex(full).reset_index()
    log("7 aggregate", "Product-months in complete grid", len(m), f"{m['units'].isna().sum()} empty cells left explicit (NaN), not zero")
    return m


# =========================================================== 8. Google Trends
def clean_trends():
    """'<1' means 'below the reporting threshold'. If the other downloads of the same month show a
    clearly higher value (>= 5), the '<1' is a sampling failure, not a real low: treat it as missing
    and average the other downloads. If all downloads agree it is low, use 0.5."""
    vals, masks, names = [], [], []
    for f in sorted(RAW.glob("trends_download*.csv")):
        t = pd.read_csv(f, skiprows=2, index_col=0)
        t.columns = [c.split(":")[0].strip() for c in t.columns]
        t.index = pd.to_datetime(t.index)
        small = (t.astype("string") == "<1").fillna(False).astype(bool)
        vals.append(t.apply(lambda s: pd.to_numeric(s.astype("string").replace("<1", "0.5"), errors="coerce")))
        masks.append(small); names.append(f.name)
        log("8 trends", f"'{f.name}': '<1' entries", small.values.sum(), "detected; resolved against the other downloads (next row)")
    V = np.stack([v.values.astype(float) for v in vals]); Mk = np.stack([m.values for m in masks])
    adj = V.copy(); n_fail = 0
    for i in range(len(V)):
        others = np.delete(V, i, axis=0); om = np.delete(Mk, i, axis=0)
        ok_others = np.where(om, np.nan, others)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slice: every other download is also "<1"
            hi = np.nanmax(ok_others, axis=0)
        fail = Mk[i] & (hi >= 5)
        adj[i][fail] = np.nan; n_fail += int(fail.sum())
    total = int(Mk.sum())
    log("8 trends", "'<1' contradicted by another download (sampling failure)", n_fail, "set to missing and excluded from the average")
    log("8 trends", "'<1' confirmed by all downloads (genuinely low)", total - n_fail, "replaced by 0.5 (sensitivity test planned with 0 and 1)")
    avg = pd.DataFrame(np.nanmean(adj, axis=0), index=vals[0].index, columns=vals[0].columns)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        sd = np.nanstd(adj, axis=0, ddof=1)
    log("8 trends", "Repeated downloads averaged", avg.size, f"mean between-download SD = {np.nanmean(sd):.2f} index points")
    long = avg.stack().rename("search_index").reset_index()
    long.columns = ["month", "Product Name", "search_index"]
    long["Product Name"] = long["Product Name"].map(lambda x: {_key(c): c for c in CANON}[_key(x)])
    return long


# =========================================================== 9. reviews
KEYWORDS = {c: _key(c) for c in CANON}

def clean_reviews():
    r = pd.read_json(RAW / "reviews_raw.jsonl", lines=True, convert_dates=False)   # keep ms timestamps as integers
    n = len(r)
    r = r.drop_duplicates(subset=["user_id", "asin", "timestamp", "text"])
    log("9 reviews", "Duplicate reviews", n - len(r), "dropped")
    bad = ~r["rating"].between(1, 5) & r["rating"].notna()
    log("9 reviews", "Ratings outside 1-5", bad.sum(), "set to missing")
    r.loc[bad, "rating"] = np.nan
    log("9 reviews", "Rating missing", r["rating"].isna().sum(), "left missing; excluded from the average, still counted in review volume")
    r["date"] = pd.to_datetime(r["timestamp"], unit="ms")
    r["month"] = r["date"].dt.to_period("M").dt.to_timestamp()
    title = r["title"].map(_key)
    r["Product Name"] = pd.Series([None] * len(r), index=r.index, dtype="object")
    for prod, kw in KEYWORDS.items():
        r.loc[title.str.contains(kw.split()[0], regex=False) & r["Product Name"].isna(), "Product Name"] = prod
    log("9 reviews", "Items not matching any product keyword", r["Product Name"].isna().sum(), "excluded as irrelevant")
    r = r.dropna(subset=["Product Name"])
    v = r[r["verified_purchase"]]
    out = v.groupby(["Product Name", "month"]).agg(review_count=("rating", "size"), avg_rating=("rating", "mean")).reset_index()
    return out


# =========================================================== 10. merge, lags, split, scale
def build_panel(sales_m, trends, reviews):
    p = sales_m.merge(trends, on=["Product Name", "month"], how="left", validate="1:1")
    p = p.merge(reviews, on=["Product Name", "month"], how="left", validate="1:1")
    p["reviews_covered"] = p["month"] <= "2023-09-01"
    p.loc[~p["reviews_covered"], ["review_count", "avg_rating"]] = np.nan   # source ends Sep 2023: unknown, not zero
    p.loc[p["reviews_covered"], "review_count"] = p.loc[p["reviews_covered"], "review_count"].fillna(0)
    p = p.sort_values(["Product Name", "month"])
    g = p.groupby("Product Name")
    for k in (1, 2, 3):                                     # lags built WITHIN each product only
        p[f"units_lag{k}"] = g["units"].shift(k)
        p[f"search_lag{k}"] = g["search_index"].shift(k)
        p[f"reviews_lag{k}"] = g["review_count"].shift(k)
    p["log_units"] = np.log1p(p["units"])
    p["month_of_year"] = p["month"].dt.month
    p["month_sin"] = np.sin(2 * np.pi * p["month_of_year"] / 12)
    p["month_cos"] = np.cos(2 * np.pi * p["month_of_year"] / 12)
    log("10 panel", "Panel rows (product x month)", len(p), "merged with validate='1:1'; lags created within product")
    return p


def split_and_scale(p, cutoff="2023-12-01"):
    train = p["month"] <= cutoff
    num = ["search_index", "search_lag1", "search_lag2", "search_lag3", "price", "discount", "log_units"]
    mu, sd = p.loc[train, num].mean(), p.loc[train, num].std()
    scaled = (p[num] - mu) / sd
    scaled.columns = [f"{c}_z" for c in num]
    p = pd.concat([p, scaled], axis=1)
    p["split"] = np.where(train, "train", "test")
    json.dump({"mean": mu.to_dict(), "std": sd.to_dict(), "cutoff": cutoff}, open(OUT / "scaler_params.json", "w"), indent=2)
    log("10 scale", f"z-score fitted on months <= {cutoff} only", train.sum(), "parameters saved to scaler_params.json; test rows transformed, never used to fit")
    return p


# =========================================================== 11. quality gate
def quality_report(p, sales_clean):
    checks = {
        "no duplicate keys in sales": not sales_clean.duplicated(KEY).any(),
        "all dates are Mondays": bool((sales_clean["Date"].dt.dayofweek == 0).all()),
        "no negative units/price": bool((sales_clean[["Units Sold", "Price"]] >= 0).all().all()),
        "discount within 0-1": bool(sales_clean["Discount"].between(0, 1).all()),
        "returns <= units": bool((sales_clean["Units Returned"] <= sales_clean["Units Sold"]).all()),
        "revenue = units x price (tol 0.01 per unit)": float(((sales_clean["Revenue"] - sales_clean["Units Sold"] * sales_clean["Price"]).abs() < 1.0).mean()),
        "panel is complete grid (rows)": len(p) == p["Product Name"].nunique() * p["month"].nunique(),
        "search index within 0-100": bool(p["search_index"].between(0, 100).all()),
        "no missing in model targets (units)": int(p["units"].isna().sum()),
    }
    return checks


if __name__ == "__main__":
    raw = pd.read_csv(RAW / "sales_raw.csv", dtype=str, keep_default_na=False)
    raw_n = len(raw)
    fingerprint = {f.name: sha256(f) for f in sorted(RAW.glob("*.csv")) + sorted(RAW.glob("*.jsonl"))}
    miss_raw = to_number(raw["Units Sold"]).isna().sum()

    s = standardise_types(raw)
    miss_stage1 = s[["Units Sold", "Price", "Revenue", "Discount", "Units Returned"]].isna().sum().to_dict()
    s = standardise_text(s)
    s = snap_to_monday(s)
    s = deduplicate(s)
    s = fix_and_flag_validity(s)
    s = handle_missing(s)
    s = detect_outliers(s)
    s.to_csv(OUT / "sales_clean_weekly.csv", index=False)

    sm = to_monthly(s)
    tr = clean_trends()
    rv = clean_reviews()
    panel = split_and_scale(build_panel(sm, tr, rv))
    panel.to_csv(OUT / "panel_monthly.csv", index=False)
    checks = quality_report(panel, s)

    pd.DataFrame(LOG).to_csv(OUT / "cleaning_log.csv", index=False)
    summary = dict(raw_rows=raw_n, clean_rows=len(s), panel_rows=len(panel), missing_before_step5=miss_stage1,
                   missing_after=s.attrs["missing_after"], checks=checks, fingerprints=fingerprint,
                   pandas=pd.__version__, numpy=np.__version__)
    json.dump(summary, open(OUT / "run_summary.json", "w"), indent=2, default=str)
    print(pd.DataFrame(LOG).to_string())
    print(json.dumps(summary, indent=2, default=str))
