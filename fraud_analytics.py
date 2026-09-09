"""Signal-detection helpers: which columns actually separate fraud from legit.

Everything here is univariate or 2-way and label-aware; nothing depends on the
specific column names of the credit-card file, so the same code runs on any CSV
that carries a binary fraud flag.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Bins with fewer rows than this are unreliable at a ~1.5% base rate and are
# folded away rather than shown as spurious 20x lifts.
MIN_BIN_SUPPORT = 30

# Feature kinds the dashboard's pinned schema uses, mapped to binning behaviour.
#   numeric / currency -> quantile buckets
#   discrete / cyclic  -> one bucket per distinct value, kept in natural order
#   binary             -> no / yes
#   categorical        -> the levels themselves
NUMERIC_KINDS = ("numeric", "currency")
DISCRETE_KINDS = ("discrete", "cyclic")


def detect_kinds(df: pd.DataFrame, label: str, id_cols: list[str]) -> dict[str, str]:
    """Classify each column as 'numeric', 'binary', 'categorical' or 'ignore'."""
    kinds = {}
    for col in df.columns:
        if col == label or col in id_cols:
            kinds[col] = "ignore"
            continue
        s = df[col]
        n_unique = s.nunique(dropna=True)
        if n_unique <= 1:
            kinds[col] = "ignore"
        elif pd.api.types.is_numeric_dtype(s) and n_unique == 2:
            kinds[col] = "binary"
        elif pd.api.types.is_numeric_dtype(s):
            kinds[col] = "numeric"
        elif n_unique <= 50:
            kinds[col] = "categorical"
        else:
            kinds[col] = "ignore"
    return kinds


def bin_series(s: pd.Series, kind: str, n_bins: int = 8) -> pd.Series:
    """Turn any column into ordered, human-readable buckets."""
    if kind in DISCRETE_KINDS:
        # Small-integer columns (velocity 0-9, hour 0-23): every value is its own
        # bucket, ordered numerically rather than alphabetically.
        values = sorted(s.dropna().unique())
        labels = [f"{int(v):02d}:00" for v in values] if kind == "cyclic" \
            else [f"{v:g}" for v in values]
        mapping = dict(zip(values, labels))
        return pd.Series(pd.Categorical(s.map(mapping), categories=labels, ordered=True),
                         index=s.index, name=s.name)
    if kind in NUMERIC_KINDS:
        # Quantile edges, de-duplicated - skewed money columns collapse otherwise.
        edges = np.unique(np.nanquantile(s, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            return s.astype(str)
        binned = pd.cut(s, bins=edges, include_lowest=True, duplicates="drop")
        # pd.cut nudges the first left edge below the minimum; label from the real
        # edges so the lowest bucket reads "0 - 4", not "-0.001 - 4".
        labels = [f"{lo:,.4g} – {hi:,.4g}" for lo, hi in zip(edges[:-1], edges[1:])]
        return binned.cat.rename_categories(labels[: len(binned.cat.categories)])
    if kind == "binary":
        return s.map({0: "0 · no", 1: "1 · yes"}).fillna(s.astype(str))
    return s.astype(str)


def segment_table(df: pd.DataFrame, col: str, label: str, kind: str,
                  n_bins: int = 8) -> pd.DataFrame:
    """Per-bucket fraud rate, lift over the baseline, and weight of evidence."""
    buckets = bin_series(df[col], kind, n_bins)
    y = df[label]
    base = y.mean()

    g = pd.DataFrame({"bucket": buckets, "y": y}).groupby("bucket", observed=True)["y"]
    out = pd.DataFrame({"transactions": g.size(), "fraud": g.sum()})
    out["fraud_rate"] = out["fraud"] / out["transactions"]
    out["lift"] = out["fraud_rate"] / base if base else np.nan
    out["share_of_fraud"] = out["fraud"] / max(int(y.sum()), 1)

    # WoE with the standard 0.5 continuity correction so empty buckets stay finite.
    tot_bad, tot_good = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    dist_bad = (out["fraud"] + 0.5) / tot_bad
    dist_good = (out["transactions"] - out["fraud"] + 0.5) / tot_good
    out["woe"] = np.log(dist_bad / dist_good)
    out["iv_part"] = (dist_bad - dist_good) * out["woe"]

    return out.reset_index().rename(columns={"bucket": "segment"})


def information_value(df: pd.DataFrame, col: str, label: str, kind: str) -> float:
    return float(segment_table(df, col, label, kind)["iv_part"].sum())


def iv_verdict(iv: float) -> str:
    """Standard credit-risk IV bands."""
    if iv < 0.02:
        return "none"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "very strong"


def univariate_auc(df: pd.DataFrame, col: str, label: str, kind: str) -> float:
    """AUC of the column used alone as a score. Folded to >= 0.5 - direction is
    reported separately, so 0.2 and 0.8 are equally informative."""
    y = df[label]
    if y.nunique() < 2:
        return np.nan
    if kind in NUMERIC_KINDS + DISCRETE_KINDS + ("binary",):
        x = df[col]
    else:  # categorical -> score each level by its own fraud rate
        x = df[col].map(df.groupby(col, observed=True)[label].mean())
    try:
        auc = roc_auc_score(y, x)
    except ValueError:
        return np.nan
    return float(max(auc, 1 - auc))


def signal_scan(df: pd.DataFrame, label: str, kinds: dict[str, str]) -> pd.DataFrame:
    """One row per candidate feature, ranked by information value."""
    rows = []
    base = df[label].mean()
    for col, kind in kinds.items():
        if kind == "ignore":
            continue
        seg = segment_table(df, col, label, kind)
        strong = seg[seg["transactions"] >= MIN_BIN_SUPPORT]
        if strong.empty:
            strong = seg
        top = strong.loc[strong["lift"].idxmax()]
        iv = float(seg["iv_part"].sum())
        rows.append({
            "feature": col,
            "type": kind,
            "iv": iv,
            "strength": iv_verdict(iv),
            "auc": univariate_auc(df, col, label, kind),
            "max_lift": float(top["lift"]),
            "riskiest_segment": str(top["segment"]),
            "segment_fraud_rate": float(top["fraud_rate"]),
            "segment_n": int(top["transactions"]),
            "baseline_rate": float(base),
            "buckets": int(len(seg)),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)


def mine_combinations(df: pd.DataFrame, label: str, kinds: dict[str, str],
                      min_support: int = 40, n_bins: int = 5,
                      max_features: int = 8,
                      names: dict[str, str] | None = None) -> pd.DataFrame:
    """Every 2-way bucket intersection, ranked by lift - the rule candidates.

    Restricted to the top `max_features` columns by IV to keep the pair count
    (and the runtime) bounded on wide files.
    """
    usable = [c for c, k in kinds.items() if k != "ignore"]
    if len(usable) > max_features:
        ranked = signal_scan(df, label, {c: kinds[c] for c in usable})
        usable = ranked["feature"].head(max_features).tolist()

    binned = pd.DataFrame(
        {c: bin_series(df[c], kinds[c], n_bins).astype(str) for c in usable}
    )
    binned[label] = df[label].to_numpy()
    base = df[label].mean()

    label_of = (lambda c: names.get(c, c)) if names else (lambda c: c)

    rows = []
    for a, b in itertools.combinations(usable, 2):
        g = binned.groupby([a, b], observed=True)[label]
        agg = pd.DataFrame({"transactions": g.size(), "fraud": g.sum()})
        agg = agg[agg["transactions"] >= min_support]
        if agg.empty:
            continue
        agg["fraud_rate"] = agg["fraud"] / agg["transactions"]
        agg = agg.reset_index()
        for r in agg.itertuples(index=False):
            rows.append({
                "rule": (f"{label_of(a)} = {getattr(r, a)}"
                         f"   AND   {label_of(b)} = {getattr(r, b)}"),
                "feature_a": a, "feature_b": b,
                "transactions": int(r.transactions),
                "fraud": int(r.fraud),
                "fraud_rate": float(r.fraud_rate),
                "lift": float(r.fraud_rate / base) if base else np.nan,
                "share_of_fraud": float(r.fraud / max(int(df[label].sum()), 1)),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("lift", ascending=False, ignore_index=True)


def cross_matrix(df: pd.DataFrame, row_col: str, col_col: str, label: str,
                 kinds: dict[str, str], n_bins: int = 6):
    """Fraud rate for every (row bucket x column bucket) cell, plus its support."""
    r = bin_series(df[row_col], kinds[row_col], n_bins)
    c = bin_series(df[col_col], kinds[col_col], n_bins)
    frame = pd.DataFrame({"r": r.astype(str), "c": c.astype(str), "y": df[label]})
    rate = frame.pivot_table(index="r", columns="c", values="y", aggfunc="mean")
    count = frame.pivot_table(index="r", columns="c", values="y", aggfunc="size")

    # Preserve the natural bucket order rather than the alphabetical default.
    def order_of(binned):
        cats = getattr(getattr(binned, "cat", binned), "categories", None)
        return [str(x) for x in (cats if cats is not None else pd.unique(binned))]

    r_order, c_order = order_of(r), order_of(c)
    r_order = [x for x in r_order if x in rate.index]
    c_order = [x for x in c_order if x in rate.columns]
    return rate.loc[r_order, c_order], count.loc[r_order, c_order]


def flag_count_table(df: pd.DataFrame, label: str,
                     flags: dict[str, pd.Series]) -> pd.DataFrame:
    """Fraud rate by how many risk flags a transaction trips.

    Used to expose a threshold-style generating rule: if the fraud rate jumps from
    ~0 to 100% between two adjacent counts, the label is a deterministic function of
    the flags and any model trained on them will score perfectly for the wrong reason.
    """
    count = sum(f.astype(int) for f in flags.values())
    g = pd.DataFrame({"n_flags": count, "y": df[label]}).groupby("n_flags")["y"]
    out = pd.DataFrame({"transactions": g.size(), "fraud": g.sum()})
    out["fraud_rate"] = out["fraud"] / out["transactions"]
    base = df[label].mean()
    out["lift"] = out["fraud_rate"] / base if base else np.nan
    return out.reset_index()


def deterministic_cut(table: pd.DataFrame, hi: float = 0.99,
                      lo: float = 0.05) -> int | None:
    """The flag count at which the rate crosses from ~never to ~always, if there is
    one. Returns None when no such clean step exists."""
    rows = table.sort_values("n_flags")
    for i in range(1, len(rows)):
        below = rows.iloc[:i]["fraud"].sum() / max(rows.iloc[:i]["transactions"].sum(), 1)
        above_rows = rows.iloc[i:]
        above = above_rows["fraud"].sum() / max(above_rows["transactions"].sum(), 1)
        if below <= lo and above >= hi:
            return int(rows.iloc[i]["n_flags"])
    return None
