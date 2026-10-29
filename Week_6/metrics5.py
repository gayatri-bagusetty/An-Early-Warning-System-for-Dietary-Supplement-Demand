"""Week 5 - evaluation metrics and validation helpers (kept separate so they can be unit-tested)."""
import numpy as np


def mape(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs(a - p) / a) * 100)


def smape(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(2 * np.abs(a - p) / (np.abs(a) + np.abs(p))) * 100)


def mae(actual, pred):
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(pred, float))))


def rmse(actual, pred):
    d = np.asarray(actual, float) - np.asarray(pred, float)
    return float(np.sqrt(np.mean(d ** 2)))


def bias_pct(actual, pred):
    """Mean percentage error; negative = under-forecasting (stock-out risk), positive = over-forecasting (excess stock)."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean((p - a) / a) * 100)


def summarize(actual_rate, pred_rate, n_records):
    """All regression metrics. MAPE/sMAPE/bias on the demand rate; MAE/RMSE on units (rate x records in the target month)."""
    au, pu = np.asarray(actual_rate) * np.asarray(n_records), np.asarray(pred_rate) * np.asarray(n_records)
    return {"MAPE": mape(actual_rate, pred_rate), "sMAPE": smape(actual_rate, pred_rate), "bias": bias_pct(actual_rate, pred_rate),
            "MAE_units": mae(au, pu), "RMSE_units": rmse(au, pu), "n": int(len(au))}


def classification_scores(y_true, y_pred, y_score=None):
    """Precision, recall, F1 for the positive (surge) class, plus threshold-free AUCs when scores are given."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum()); fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum()); tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan"); rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    out = {"precision": prec, "recall": rec, "F1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "base_rate": float(y_true.mean())}
    if y_score is not None and len(set(y_true)) == 2:
        out["PR_AUC"] = float(average_precision_score(y_true, y_score)); out["ROC_AUC"] = float(roc_auc_score(y_true, y_score))
    return out


def block_bootstrap_diff(err_a, err_b, groups, n_boot=2000, seed=0):
    """95% interval for mean(err_a - err_b) resampling whole groups (target months), because the 8 products in one
    month share the same calendar and are not independent."""
    rng = np.random.default_rng(seed)
    err_a, err_b, groups = np.asarray(err_a), np.asarray(err_b), np.asarray(groups)
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    d = err_a - err_b; means = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        means.append(np.concatenate([d[idx[g]] for g in pick]).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def partial_f_test(y, X_restricted, X_full):
    """Nested OLS F-test (Week 1 plan for H3 and H4). Returns F, df1, df2, p."""
    from scipy import stats
    y = np.asarray(y, float)
    def rss(X):
        X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ beta
        return float(r @ r), int(np.linalg.matrix_rank(X))
    r0, k0 = rss(X_restricted); r1, k1 = rss(X_full)
    df1, df2 = k1 - k0, len(y) - k1
    F = ((r0 - r1) / df1) / (r1 / df2)
    return float(F), int(df1), int(df2), float(stats.f.sf(F, df1, df2))
