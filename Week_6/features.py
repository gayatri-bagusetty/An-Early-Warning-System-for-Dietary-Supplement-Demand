"""Week 4 - Feature engineering for the supplement demand forecasting panel.

Forecast set-up (Week 1): predict demand in month t using only what is known at the
forecast origin o = t - h, for horizons h = 1, 2, 3 months.

Target = "demand rate" = units per record = monthly units / number of weekly records in the month.
EDA showed that raw monthly units mostly reflect how many records the month contains (corr 0.83),
not demand. All "lu" columns below are therefore log(units per record).

Leakage rules enforced here (and tested in test_features.py):
  1. Every feature built from history is shifted by at least h months, within each product.
  2. Anything learned from data (seasonal indices, product means) uses TRAINING months only.
  3. Planned price and discount of month t are treated as known in advance (a brand sets them).
"""
import numpy as np
import pandas as pd

KEYS = ["Product Name", "month"]
TRAIN_END = "2023-12-01"

GROUPS = {
    "memory":    ["lu_o", "lu_roll3_o", "lu_mom3_o", "lu_yoy_t"],
    "calendar":  ["month_sin", "month_cos", "flag_newyear", "flag_summer", "flag_gifting", "season_idx"],
    "search_raw": ["search_o", "search_o1", "search_o2"],
    "search_eng": ["search_anom_o", "search_mom3_o", "search_roll3_o"],
    "price_promo": ["price_idx_t", "discount_t", "disc_x_searchanom"],
    "behaviour": ["return_rate_o", "platform_hhi_o", "discount_share_o"],
    "reviews":   ["rev_count_o", "rating_o"],
}


def monthly_behaviour(weekly: pd.DataFrame) -> pd.DataFrame:
    """Customer-behaviour and channel features from the cleaned weekly records."""
    w = weekly.copy()
    w["month"] = w["Date"].dt.to_period("M").dt.to_timestamp()
    w["disc_units"] = w["Units Sold"] * (w["Discount"] > 0)
    g = w.groupby(KEYS).agg(units=("Units Sold", "sum"), returns=("Units Returned", "sum"), disc_units=("disc_units", "sum")).reset_index()
    out = g[KEYS].copy()
    out["return_rate"] = g["returns"] / g["units"]
    out["discount_share"] = g["disc_units"] / g["units"]
    for col, name in [("Platform", "platform_hhi"), ("Location", "location_hhi")]:
        s = w.pivot_table(index=KEYS, columns=col, values="Units Sold", aggfunc="sum", fill_value=0)
        s = s.div(s.sum(axis=1), axis=0)                      # share of units by channel
        h = (s ** 2).sum(axis=1).rename(name).reset_index()   # Herfindahl index: 1/3 = even split, 1 = one channel
        out = out.merge(h, on=KEYS, how="left")
    return out


def seasonal_table(df: pd.DataFrame, col: str, train_mask: pd.Series) -> pd.Series:
    """Product x month-of-year mean of (col minus the product's training mean), fitted on training rows only."""
    t = df[train_mask]
    dev = t[col] - t.groupby("Product Name")[col].transform("mean")
    return dev.groupby([t["Product Name"], t["month"].dt.month]).mean()


def build_features(panel: pd.DataFrame, beh: pd.DataFrame, h: int, train_end: str = TRAIN_END) -> pd.DataFrame:
    p = panel.merge(beh, on=KEYS, how="left").sort_values(KEYS).reset_index(drop=True)
    p["upr"] = p["units"] / p["n_records"]          # coverage-normalised demand
    p["lu"] = np.log(p["upr"])
    train = p["month"] <= train_end
    g = p.groupby("Product Name")
    moy = p["month"].dt.month

    # --- training-only statistics -------------------------------------------------
    si_units = seasonal_table(p, "lu", train)
    si_search = seasonal_table(p, "search_index", train)
    key = list(zip(p["Product Name"], moy))
    p["season_idx"] = pd.Series(key).map(si_units).values                      # target-encoded seasonality of the target month
    search_season = pd.Series(key).map(si_search).values
    search_mean_train = p[train].groupby("Product Name")["search_index"].mean()
    price_mean_train = p[train].groupby("Product Name")["price"].mean()
    p["search_anom"] = p["search_index"] - (p["Product Name"].map(search_mean_train) + search_season)   # search minus its seasonal norm

    # --- helper: value of a per-product series as known at the origin (shift by h) -----
    def at_origin(series, extra=0):
        return series.groupby(p["Product Name"]).shift(h + extra)

    # A. sales memory
    p["lu_o"] = at_origin(p["lu"])
    p["lu_roll3_o"] = at_origin(g["lu"].transform(lambda s: s.rolling(3).mean()))
    p["lu_mom3_o"] = at_origin(p["lu"] - g["lu"].shift(3))
    p["lu_yoy_t"] = g["lu"].shift(12)                                           # same month last year (known: 12 >= h)
    # B. calendar (known for any future month)
    p["month_sin"] = np.sin(2 * np.pi * moy / 12); p["month_cos"] = np.cos(2 * np.pi * moy / 12)
    p["flag_newyear"] = (moy == 1).astype(float)
    p["flag_summer"] = moy.isin([6, 7, 8]).astype(float)
    p["flag_gifting"] = moy.isin([11, 12]).astype(float)
    # C. search
    p["search_o"] = at_origin(p["search_index"])
    p["search_o1"] = at_origin(p["search_index"], 1)
    p["search_o2"] = at_origin(p["search_index"], 2)
    p["search_anom_o"] = at_origin(p["search_anom"])
    p["search_mom3_o"] = at_origin(p["search_index"] - g["search_index"].shift(3))
    p["search_roll3_o"] = at_origin(g["search_index"].transform(lambda s: s.rolling(3).mean()))
    # D. planned price / promotion for the target month, and an interaction
    p["price_idx_t"] = p["price"] / p["Product Name"].map(price_mean_train)
    p["discount_t"] = p["discount"]
    p["disc_x_searchanom"] = p["discount_t"] * p["search_anom_o"]
    # E. behaviour / channel
    p["return_rate_o"] = at_origin(p["return_rate"])
    p["platform_hhi_o"] = at_origin(p["platform_hhi"])
    p["discount_share_o"] = at_origin(p["discount_share"])
    # F. reviews (only months covered by the source)
    p["rev_count_o"] = at_origin(p["review_count"])
    p["rating_o"] = at_origin(p["avg_rating"])

    p["y"] = p["lu"]; p["horizon"] = h
    p["is_train"] = train
    return p


def design_matrix(df: pd.DataFrame, groups, products) -> pd.DataFrame:
    """Numeric design matrix: chosen feature groups + product one-hot + product x season interactions."""
    cols = [c for gname in groups for c in GROUPS[gname]]
    X = df[cols].copy()
    if "calendar" in groups or "memory" in groups:
        for pr in products:
            ind = (df["Product Name"] == pr).astype(float)
            X[f"prod_{pr}"] = ind
            if "calendar" in groups:
                X[f"prod_{pr}_x_sin"] = ind * df["month_sin"]
                X[f"prod_{pr}_x_cos"] = ind * df["month_cos"]
    return X
