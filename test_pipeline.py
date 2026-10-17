"""Unit tests for the cleaning rules. Run: python test_pipeline.py   (also works with pytest)"""
import numpy as np, pandas as pd
import pipeline as P


def test_money_and_na_parsing():
    s = pd.Series(["$31.88", "1,234", "N/A", "", "42"])
    out = P.to_number(s)
    assert out.iloc[0] == 31.88 and out.iloc[1] == 1234 and out.iloc[4] == 42
    assert out.iloc[2] is pd.NA or np.isnan(out.iloc[2])


def test_ambiguous_date_resolved_by_monday_rule():
    # 06/01/2020 is Monday 6 Jan (dd/mm) but Sunday 1 Jun (mm/dd) -> must pick 6 Jan
    d = P.parse_dates(pd.Series(["06/01/2020", "Jan 13, 2020", "2020-01-20"]))
    assert list(d.dt.strftime("%Y-%m-%d")) == ["2020-01-06", "2020-01-13", "2020-01-20"]


def test_name_variants_collapse():
    df = pd.DataFrame({"Product Name": [" green-tea  extract ", "OMEGA 3", "Biotin"],
                       "Location": ["US", "U.K.", "canada"], "Platform": ["amazon.com", "iherb", "Walmart"]})
    out = P.standardise_text(df)
    assert list(out["Product Name"]) == ["Green Tea Extract", "Omega-3", "Biotin"]
    assert list(out["Location"]) == ["USA", "UK", "Canada"]


def test_revenue_identity_recovers_missing_units():
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-06"]), "Product Name": ["Biotin"], "Location": ["USA"],
                       "Platform": ["Amazon"], "Units Sold": [np.nan], "Price": [20.0], "Revenue": [3000.0],
                       "Discount": [0.1], "Units Returned": [3.0]})
    out = P.handle_missing(df)
    assert out["Units Sold"].iloc[0] == 150


def test_entry_error_corrected_but_genuine_spike_kept():
    n = 40
    base = pd.DataFrame({"Product Name": ["Biotin"] * n, "Units Sold": [100.0] * n, "Price": [20.0] * n})
    base["Revenue"] = 2000.0
    for c in ["units_sold", "price", "revenue"]:
        base[f"was_missing_{c}"] = False
    base.loc[0, "Units Sold"] = 800.0                       # entry error: revenue still says 100 units
    base.loc[1, ["Units Sold", "Revenue"]] = [300.0, 6000.0]  # genuine promo: revenue agrees
    base["Units Returned"] = 2.0
    out = P.detect_outliers(base)
    assert out.loc[0, "Units Sold"] == 100 and out.loc[0, "flag_units_reconciled"]
    assert out.loc[1, "Units Sold"] == 300 and not out.loc[1, "flag_units_reconciled"]


def test_lags_do_not_cross_products():
    m = pd.DataFrame({"Product Name": ["A", "A", "B", "B"], "month": pd.to_datetime(["2020-01-01", "2020-02-01"] * 2),
                      "units": [1, 2, 10, 20], "search_index": [5, 6, 50, 60]})
    sales_m = m.drop(columns="search_index").assign(revenue=0, returns=0, n_records=1, n_weeks=1, n_outliers=0,
                                                    n_imputed=0, discount=0.0, price=1.0, units_wins=m["units"])
    p = P.build_panel(sales_m, m[["Product Name", "month", "search_index"]],
                      pd.DataFrame(columns=["Product Name", "month", "review_count", "avg_rating"]))
    b_first = p[(p["Product Name"] == "B") & (p["month"] == "2020-01-01")]
    assert np.isnan(b_first["units_lag1"].iloc[0])           # not A's February value


def test_trends_lt1_checked_against_other_downloads(tmp_path=None):
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    # month 1: download 3 says "<1" but the others say 34 and 43 -> sampling failure, excluded
    # month 2: all three downloads say "<1" -> genuinely low, becomes 0.5
    rows = {1: ["34", "<1"], 2: ["43", "<1"], 3: ["<1", "<1"]}
    for i, (a, b) in rows.items():
        (d / f"trends_download{i}.csv").write_text(
            "Category: All categories\n\nMonth,biotin: (United States)\n2020-01," + a + "\n2020-02," + b + "\n")
    old = P.RAW
    P.RAW = d
    try:
        out = P.clean_trends().set_index("month")["search_index"]
    finally:
        P.RAW = old
    assert abs(out.iloc[0] - 38.5) < 1e-9      # mean of 34 and 43, the "<1" excluded
    assert abs(out.iloc[1] - 0.5) < 1e-9       # all agree it is low


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
