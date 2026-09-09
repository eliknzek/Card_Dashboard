"""FRAUD//SCAN - a terminal-styled console for hunting fraud predictors.

Pinned to credit_card_fraud_10k.csv: every column has a declared kind, display
label and bucketing rule in SCHEMA below, so buckets land on the real breakpoints
instead of generic quantiles.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fraud_analytics as fa
import theme as T

DATA_FILE = Path(__file__).parent / "credit_card_fraud_10k.csv"
LABEL = "is_fraud"

# column -> how to bucket it, what to call it, how to format its values.
# "cyclic"/"discrete" give one bucket per value; "currency"/"numeric" quantile-bin.
SCHEMA: dict[str, dict] = {
    "transaction_hour":    dict(kind="cyclic",      label="hour of day",         unit="h"),
    "device_trust_score":  dict(kind="numeric",     label="device trust score",  unit="pts"),
    "foreign_transaction": dict(kind="binary",      label="foreign transaction", unit=""),
    "location_mismatch":   dict(kind="binary",      label="location mismatch",   unit=""),
    "velocity_last_24h":   dict(kind="discrete",    label="txns in last 24h",    unit=""),
    "amount":              dict(kind="currency",    label="amount",              unit="$"),
    "merchant_category":   dict(kind="categorical", label="merchant category",   unit=""),
    "cardholder_age":      dict(kind="numeric",     label="cardholder age",      unit="yrs"),
}
# The five conditions that, taken together, turn out to generate the label in this
# file (see the "generating rule" panel in the model tab). Thresholds were recovered
# from the data, not assumed.
RISK_FLAGS = {
    "night (00:00–03:59)":   lambda d: d["transaction_hour"] < 4,
    "device trust < 40":     lambda d: d["device_trust_score"] < 40,
    "foreign transaction":   lambda d: d["foreign_transaction"] == 1,
    "location mismatch":     lambda d: d["location_mismatch"] == 1,
    "5+ txns in last 24h":   lambda d: d["velocity_last_24h"] >= 5,
}

KINDS = {c: spec["kind"] for c, spec in SCHEMA.items()}
FEATURES = list(SCHEMA)
NAME = {c: spec["label"] for c, spec in SCHEMA.items()}

st.set_page_config(page_title="FRAUD//SCAN", page_icon="🟩",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(T.CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- data layer
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    missing = [c for c in [*FEATURES, LABEL] if c not in df.columns]
    if missing:
        st.error(f"`{DATA_FILE.name}` is missing expected columns: {', '.join(missing)}")
        st.stop()
    return df


@st.cache_data(show_spinner=False)
def cached_scan(df: pd.DataFrame) -> pd.DataFrame:
    return fa.signal_scan(df, LABEL, KINDS)


@st.cache_data(show_spinner=False)
def cached_combos(df: pd.DataFrame, min_support: int) -> pd.DataFrame:
    return fa.mine_combinations(df, LABEL, KINDS, min_support=min_support, names=NAME)


def money(v: float) -> str:
    return f"${v:,.0f}"


# ---------------------------------------------------------------- components
def kpi(label: str, value: str, sub: str = "", alert: bool = False):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value{" alert" if alert else ""}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def lift_color(lift: float) -> str:
    """Diverging: below baseline recedes toward neutral, above it heats up."""
    if not np.isfinite(lift):
        return T.NEUTRAL
    if lift >= 5:
        return T.DIVERGING[2]
    if lift >= 2:
        return T.DIVERGING[3]
    if lift >= 1.2:
        return T.WARNING
    if lift >= 0.8:
        return T.NEUTRAL
    return T.DIVERGING[6]


def lift_bar(seg: pd.DataFrame, title: str, baseline: float) -> go.Figure:
    """Horizontal fraud-rate bars, one per bucket, colored by lift.

    Colour is redundant here - bar length already carries the magnitude - so every
    bar is direct-labelled and no legend box is needed.
    """
    seg = seg.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=seg["fraud_rate"], y=seg["segment"].astype(str), orientation="h",
        marker=dict(color=[lift_color(v) for v in seg["lift"]],
                    cornerradius=4, line=dict(width=0)),
        text=[f"{r*100:.2f}%  ({l:.1f}×)" for r, l in zip(seg["fraud_rate"], seg["lift"])],
        textposition="outside", textfont=dict(color=T.INK_DIM, size=11), cliponaxis=False,
        customdata=np.stack([seg["transactions"], seg["fraud"], seg["lift"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>fraud rate %{x:.2%}"
                       "<br>%{customdata[1]:,} fraud / %{customdata[0]:,} txn"
                       "<br>lift %{customdata[2]:.2f}×<extra></extra>"),
    ))
    fig.add_vline(x=baseline, line=dict(color=T.INK_MUTED, width=1, dash="dot"),
                  annotation_text=f"baseline {baseline:.2%}",
                  annotation_font=dict(color=T.INK_MUTED, size=10),
                  annotation_position="top")
    fig.update_layout(**T.plotly_layout(
        title=title, height=max(250, 34 * len(seg) + 110), bargap=0.3,
        xaxis=dict(tickformat=".1%", gridcolor=T.GRID, zerolinecolor=T.GRID,
                   linecolor=T.BORDER, tickfont=dict(color=T.INK_MUTED, size=11),
                   range=[0, float(seg["fraud_rate"].max()) * 1.38 + 1e-4]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=T.BORDER,
                   tickfont=dict(color=T.INK_DIM, size=11)),
    ))
    return fig


def hour_profile(df: pd.DataFrame, baseline: float) -> go.Figure:
    """Fraud rate across the 24-hour clock. Hour is cyclic and evenly spaced, so it
    reads as an ordered line rather than 24 stacked bars."""
    g = df.groupby("transaction_hour", observed=True)[LABEL]
    prof = pd.DataFrame({"n": g.size(), "fraud": g.sum()}).reindex(range(24), fill_value=0)
    prof["rate"] = np.where(prof["n"] > 0, prof["fraud"] / prof["n"], np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prof.index, y=prof["rate"], mode="lines+markers", name="fraud rate",
        line=dict(color=T.CRITICAL, width=2),
        marker=dict(size=8, color=T.CRITICAL, line=dict(width=2, color=T.BG)),
        customdata=np.stack([prof["n"], prof["fraud"]], axis=-1),
        hovertemplate=("<b>%{x:02d}:00</b><br>fraud rate %{y:.2%}"
                       "<br>%{customdata[1]:,} fraud / %{customdata[0]:,} txn<extra></extra>"),
    ))
    fig.add_hline(y=baseline, line=dict(color=T.INK_MUTED, width=1, dash="dot"),
                  annotation_text=f"baseline {baseline:.2%}",
                  annotation_font=dict(color=T.INK_MUTED, size=10))
    fig.update_layout(**T.plotly_layout(
        title="THE NIGHT WINDOW — fraud rate by hour of day", height=330,
        showlegend=False,
        xaxis=dict(title="hour", dtick=2, tickformat="02d", gridcolor=T.GRID,
                   linecolor=T.BORDER, tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED)),
        yaxis=dict(title="fraud rate", tickformat=".1%", gridcolor=T.GRID,
                   linecolor=T.BORDER, tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED), rangemode="tozero"),
    ))
    return fig


def heatmap(rate: pd.DataFrame, count: pd.DataFrame, row_col: str, col_col: str,
            baseline: float) -> go.Figure:
    """Sequential single hue, dark -> bright: magnitude of the fraud rate."""
    fig = go.Figure(go.Heatmap(
        z=rate.values * 100, x=list(rate.columns), y=list(rate.index),
        colorscale=[[i / (len(T.SEQ_GREEN) - 1), c] for i, c in enumerate(T.SEQ_GREEN)],
        xgap=2, ygap=2, customdata=count.values,
        colorbar=dict(title=dict(text="fraud %", font=dict(color=T.INK_MUTED, size=11)),
                      tickfont=dict(color=T.INK_MUTED, size=10),
                      outlinewidth=0, thickness=10, len=0.8),
        hovertemplate=(f"{NAME[row_col]} <b>%{{y}}</b><br>{NAME[col_col]} <b>%{{x}}</b>"
                       "<br>fraud rate %{z:.2f}%<br>%{customdata:,} txn<extra></extra>"),
    ))
    fig.update_layout(**T.plotly_layout(
        title=f"FRAUD RATE — {NAME[row_col]} × {NAME[col_col]}  (baseline {baseline:.2%})",
        height=min(640, max(330, 30 * len(rate) + 150)),
        xaxis=dict(title=NAME[col_col], tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title=NAME[row_col], tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED), gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


def class_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    """Legit vs fraud shape for one column. Two series -> the reserved status pair,
    each normalised to its own class so the 1.5% minority stays visible."""
    kind = KINDS[col]
    fig = go.Figure()
    for value, name, color in ((0, "legitimate", T.GOOD), (1, "fraud", T.CRITICAL)):
        s = df.loc[df[LABEL] == value, col]
        if s.empty:
            continue
        if kind in ("numeric", "currency"):
            fig.add_trace(go.Histogram(
                x=s, name=name, histnorm="probability density", nbinsx=40, opacity=0.62,
                marker=dict(color=color, line=dict(width=0)),
                hovertemplate=f"<b>{name}</b><br>%{{x}}<br>density %{{y:.4f}}<extra></extra>"))
        else:
            vc = s.value_counts(normalize=True).sort_index()
            fig.add_trace(go.Bar(
                x=vc.index.astype(str), y=vc.values, name=name,
                marker=dict(color=color, cornerradius=4, line=dict(width=2, color=T.BG)),
                hovertemplate=f"<b>{name}</b><br>%{{x}}<br>share %{{y:.1%}}<extra></extra>"))
    fig.update_layout(**T.plotly_layout(
        title=f"DISTRIBUTION — {NAME[col]} · each class normalised to itself",
        height=330, barmode="overlay" if kind in ("numeric", "currency") else "group",
        bargap=0.2,
        xaxis=dict(title=NAME[col], gridcolor=T.GRID, linecolor=T.BORDER,
                   tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED)),
        yaxis=dict(title="density" if kind in ("numeric", "currency") else "share of class",
                   gridcolor=T.GRID, linecolor=T.BORDER,
                   tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED)),
    ))
    return fig


def flag_table(frame: pd.DataFrame) -> pd.DataFrame:
    return fa.flag_count_table(frame, LABEL, {k: f(frame) for k, f in RISK_FLAGS.items()})


def flag_step_fig(tbl: pd.DataFrame, cut: int | None) -> go.Figure:
    """Fraud rate against number of risk flags tripped - the shape that reveals a
    threshold rule: flat near zero, then a vertical wall."""
    fig = go.Figure(go.Bar(
        x=tbl["n_flags"], y=tbl["fraud_rate"],
        marker=dict(color=[T.CRITICAL if (cut is not None and k >= cut) else T.NEUTRAL
                           for k in tbl["n_flags"]],
                    cornerradius=4, line=dict(width=0)),
        text=[f"{v:.1%}" for v in tbl["fraud_rate"]], textposition="outside",
        textfont=dict(color=T.INK_DIM, size=11), cliponaxis=False,
        customdata=np.stack([tbl["transactions"], tbl["fraud"]], axis=-1),
        hovertemplate=("<b>%{x} flags</b><br>fraud rate %{y:.2%}"
                       "<br>%{customdata[1]:,} fraud / %{customdata[0]:,} txn<extra></extra>"),
    ))
    fig.update_layout(**T.plotly_layout(
        title="THE GENERATING RULE — fraud rate by number of risk flags tripped",
        height=330, bargap=0.45, showlegend=False,
        xaxis=dict(title="risk flags tripped", dtick=1, gridcolor="rgba(0,0,0,0)",
                   linecolor=T.BORDER, tickfont=dict(color=T.INK_DIM, size=12),
                   title_font=dict(color=T.INK_MUTED)),
        yaxis=dict(title="fraud rate", tickformat=".0%", range=[0, 1.18],
                   gridcolor=T.GRID, linecolor=T.BORDER,
                   tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED)),
    ))
    return fig


# ---------------------------------------------------------------- load + filter
raw = load_data()

st.sidebar.markdown("### ▚ FILTERS")
if st.sidebar.button("⟲ reset all filters", width="stretch"):
    for k in list(st.session_state):
        if k.startswith("flt_"):
            del st.session_state[k]
    st.rerun()

class_pick = st.sidebar.radio("class", ["all", "fraud only", "legit only"],
                              horizontal=True, key="flt_class")
mask = pd.Series(True, index=raw.index)
if class_pick == "fraud only":
    mask &= raw[LABEL] == 1
elif class_pick == "legit only":
    mask &= raw[LABEL] == 0

active: list[str] = []
for col, spec in SCHEMA.items():
    kind, name = spec["kind"], spec["label"]
    if kind == "binary":
        sel = st.sidebar.radio(name, ["any", "yes", "no"], horizontal=True, key=f"flt_{col}")
        if sel != "any":
            mask &= raw[col] == (1 if sel == "yes" else 0)
            active.append(f"{name} = {sel}")
    elif kind == "categorical":
        opts = sorted(raw[col].dropna().unique().tolist())
        sel = st.sidebar.multiselect(name, opts, default=opts, key=f"flt_{col}")
        if len(sel) != len(opts):
            mask &= raw[col].isin(sel)
            active.append(f"{name} ∈ {{{', '.join(sel) or '∅'}}}")
    else:
        if pd.api.types.is_integer_dtype(raw[col]):
            lo, hi = int(raw[col].min()), int(raw[col].max())
            step = 1
        else:
            lo, hi = float(raw[col].min()), float(raw[col].max())
            step = max(round((hi - lo) / 200, 2), 0.01)
        sel = st.sidebar.slider(name, lo, hi, (lo, hi), step=step, key=f"flt_{col}")
        if sel != (lo, hi):
            mask &= raw[col].between(*sel)
            fmt = money if kind == "currency" else (lambda v: f"{v:,.0f}")
            active.append(f"{name} {fmt(sel[0])}–{fmt(sel[1])}")

df = raw[mask].copy()
n_filters = len(active) + (class_pick != "all")
st.sidebar.markdown(
    f'<div class="term">rows <b>{len(df):,}</b> / {len(raw):,}'
    f'<br>filters active <b>{n_filters}</b></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- header + KPIs
st.markdown("# ▚▚ FRAUD//SCAN")
st.markdown(
    f'<div class="term">source <b>{DATA_FILE.name}</b> · '
    f'<b>{len(raw):,}</b> transactions · <b>{len(FEATURES)}</b> candidate features'
    + (f'<br>filters: {" · ".join(active)}' if active else "")
    + "</div>", unsafe_allow_html=True)
st.write("")

if df.empty:
    st.warning("Filters exclude every row. Widen them or hit reset.")
    st.stop()

n, n_fraud = len(df), int(df[LABEL].sum())
rate, base_rate = n_fraud / n, raw[LABEL].mean()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi("transactions", f"{n:,}", f"{n/len(raw):.1%} of file")
with c2:
    kpi("fraud", f"{n_fraud:,}", f"{n - n_fraud:,} legitimate", alert=True)
with c3:
    kpi("fraud rate", f"{rate:.2%}",
        f"{rate/base_rate:.2f}× the unfiltered {base_rate:.2%}", alert=rate > base_rate)
with c4:
    kpi("exposure", money(df.loc[df[LABEL] == 1, "amount"].sum()),
        "fraudulent value", alert=True)
with c5:
    kpi("avg ticket", money(df.loc[df[LABEL] == 1, "amount"].mean() if n_fraud else 0),
        f"vs {money(df.loc[df[LABEL] == 0, 'amount'].mean())} legitimate")

st.write("")
tab_scan, tab_seg, tab_rules, tab_model, tab_over, tab_data = st.tabs(
    ["◆ signal scan", "◆ segment explorer", "◆ rule miner",
     "◆ model", "◆ overview", "◆ transactions"])

# ---------------------------------------------------------------- 1. signal scan
with tab_scan:
    st.markdown("## which columns predict fraud")
    st.markdown(
        '<div class="term">Every candidate column, scored on the filtered rows. '
        '<b>IV</b> = information value (0.02 none · 0.1 weak · 0.3 medium · 0.5+ strong). '
        '<b>AUC</b> = how well the column alone ranks fraud (0.5 = coin flip). '
        '<b>max lift</b> = the riskiest bucket\'s fraud rate ÷ the baseline. '
        'Click any header to re-sort.</div>', unsafe_allow_html=True)
    st.write("")

    scan = cached_scan(df)
    if scan.empty:
        st.warning("No usable feature columns after filtering.")
    else:
        scan = scan.assign(name=scan["feature"].map(NAME))
        left, right = st.columns([1.7, 1])
        with left:
            st.dataframe(
                scan[["name", "iv", "strength", "auc", "max_lift",
                      "riskiest_segment", "segment_fraud_rate"]],
                width="stretch", hide_index=True,
                height=min(560, 44 + 36 * len(scan)),
                column_config={
                    "name": st.column_config.TextColumn("feature", width="medium"),
                    "iv": st.column_config.NumberColumn("IV", format="%.3f"),
                    "strength": st.column_config.TextColumn("strength", width="medium"),
                    "auc": st.column_config.NumberColumn("AUC", format="%.3f"),
                    "max_lift": st.column_config.NumberColumn("max lift", format="%.2f×"),
                    "riskiest_segment": st.column_config.TextColumn("riskiest bucket",
                                                                   width="medium"),
                    "segment_fraud_rate": st.column_config.NumberColumn("bucket rate",
                                                                       format="percent"),
                })
        with right:
            s = scan.iloc[::-1]
            fig = go.Figure(go.Bar(
                x=s["auc"] - 0.5, y=s["name"], orientation="h", base=0.5,
                marker=dict(color=[lift_color(v) for v in s["max_lift"]],
                            cornerradius=4, line=dict(width=0)),
                text=[f"{v:.3f}" for v in s["auc"]], textposition="outside",
                textfont=dict(color=T.INK_DIM, size=11), cliponaxis=False,
                customdata=np.stack([s["iv"], s["max_lift"]], axis=-1),
                hovertemplate=("<b>%{y}</b><br>AUC %{x:.3f}<br>IV %{customdata[0]:.3f}"
                               "<br>max lift %{customdata[1]:.2f}×<extra></extra>"),
            ))
            fig.update_layout(**T.plotly_layout(
                title="RANKING — univariate AUC (0.5 = no signal)",
                height=min(560, 46 * len(s) + 90), bargap=0.35,
                xaxis=dict(range=[0.5, min(1.02, float(s["auc"].max()) * 1.08)],
                           gridcolor=T.GRID, linecolor=T.BORDER,
                           tickfont=dict(color=T.INK_MUTED, size=11)),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=T.BORDER,
                           tickfont=dict(color=T.INK_DIM, size=11)),
            ))
            st.plotly_chart(fig, width="stretch", key="scan_auc")

        top = scan.iloc[0]
        dead = scan[scan["strength"].isin(["none", "weak"])]["name"].tolist()
        st.markdown(
            f'<div class="term">▸ strongest signal: <b>{top["name"]}</b> — '
            f'AUC <b>{top.auc:.3f}</b>, its riskiest bucket <b>{top.riskiest_segment}</b> '
            f'runs <span class="bad">{top.segment_fraud_rate:.2%}</span> fraud '
            f'({top.max_lift:.1f}× baseline) across {top.segment_n:,} transactions.'
            + (f'<br>▸ little or no signal: <b>{", ".join(dead)}</b>' if dead else "")
            + '<br>▸ these columns do not merely correlate — see <b>◆ model</b> for the '
              'exact rule that generates the label in this file.'
            + "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- 2. segments
with tab_seg:
    st.markdown("## where inside a column the fraud sits")
    scan = cached_scan(df)
    order = scan["feature"].tolist() if not scan.empty else FEATURES
    col = st.selectbox("feature", order, format_func=lambda c: NAME[c], key="seg_feature")
    baseline = df[LABEL].mean()

    if col == "transaction_hour":
        st.plotly_chart(hour_profile(df, baseline), width="stretch", key="seg_hours")

    seg = fa.segment_table(df, col, LABEL, KINDS[col])
    st.plotly_chart(lift_bar(seg, f"FRAUD RATE BY BUCKET — {NAME[col]}", baseline),
                    width="stretch", key="seg_lift")

    sort_by = st.radio("sort table by", ["lift", "fraud rate", "share of fraud",
                                         "transactions", "bucket order"],
                       horizontal=True, key="seg_sort")
    key = {"lift": "lift", "fraud rate": "fraud_rate", "share of fraud": "share_of_fraud",
           "transactions": "transactions"}.get(sort_by)
    view = seg.sort_values(key, ascending=False) if key else seg

    st.dataframe(
        view[["segment", "transactions", "fraud", "fraud_rate", "lift",
              "share_of_fraud", "woe"]],
        width="stretch", hide_index=True,
        column_config={
            "segment": st.column_config.TextColumn("bucket", width="medium"),
            "transactions": st.column_config.NumberColumn("txn", format="%d"),
            "fraud": st.column_config.NumberColumn("fraud", format="%d"),
            "fraud_rate": st.column_config.NumberColumn("fraud rate", format="percent"),
            "lift": st.column_config.NumberColumn("lift", format="%.2f×"),
            "share_of_fraud": st.column_config.NumberColumn("share of all fraud",
                                                           format="percent"),
            "woe": st.column_config.NumberColumn("WoE", format="%.3f"),
        })
    st.caption(f"Buckets under {fa.MIN_BIN_SUPPORT} transactions are shown but are too "
               "thin to trust at this base rate.")
    st.plotly_chart(class_distribution(df, col), width="stretch", key="seg_dist")

# ---------------------------------------------------------------- 3. rule miner
with tab_rules:
    st.markdown("## two-feature rules, ranked by lift")
    st.markdown(
        '<div class="term">Every intersection of two buckets that clears the support '
        'floor, scored against the filtered baseline. These are detection-rule '
        'candidates: high lift <b>and</b> enough volume to matter.</div>',
        unsafe_allow_html=True)
    st.write("")

    r1, r2 = st.columns([1, 3])
    with r1:
        support = st.number_input("min transactions per rule", 10, 2000, 40, step=10,
                                  key="rule_support")
    combos = cached_combos(df, int(support))
    if combos.empty:
        st.warning("No two-feature segment clears that support floor. Lower it.")
    else:
        with r2:
            min_lift = st.slider("min lift", 1.0, float(max(2.0, combos["lift"].max())),
                                 2.0, 0.5, key="rule_lift")
        shown = combos[combos["lift"] >= min_lift]
        st.dataframe(
            shown[["rule", "transactions", "fraud", "fraud_rate", "lift", "share_of_fraud"]],
            width="stretch", hide_index=True, height=420,
            column_config={
                "rule": st.column_config.TextColumn("rule"),
                "transactions": st.column_config.NumberColumn("txn", format="%d"),
                "fraud": st.column_config.NumberColumn("fraud", format="%d"),
                "fraud_rate": st.column_config.NumberColumn("fraud rate", format="percent"),
                "lift": st.column_config.NumberColumn("lift", format="%.2f×"),
                "share_of_fraud": st.column_config.NumberColumn("share of all fraud",
                                                               format="percent"),
            })
        st.caption(f"{len(shown):,} of {len(combos):,} mined rules pass the filters. "
                   "Click a header to re-sort.")

    st.markdown("### cross-tab")
    h1, h2 = st.columns(2)
    scan = cached_scan(df)
    order = scan["feature"].tolist() if not scan.empty else FEATURES
    with h1:
        row_col = st.selectbox("rows", order, index=0,
                               format_func=lambda c: NAME[c], key="hm_row")
    with h2:
        col_col = st.selectbox("columns", order, index=min(1, len(order) - 1),
                               format_func=lambda c: NAME[c], key="hm_col")
    if row_col == col_col:
        st.info("Pick two different features to cross.")
    else:
        rate_m, count_m = fa.cross_matrix(df, row_col, col_col, LABEL, KINDS)
        st.plotly_chart(heatmap(rate_m, count_m, row_col, col_col, df[LABEL].mean()),
                        width="stretch", key="rule_heatmap")
        with st.expander("cell counts (the support behind each cell)"):
            st.dataframe(count_m, width="stretch")

# ---------------------------------------------------------------- 4. model
with tab_model:
    import fraud_model as fm

    st.markdown("## how well can these features actually predict?")
    st.markdown(
        '<div class="term">The tabs above ask <b>which features correlate</b>. This one '
        'asks <b>how much fraud a score built from them would catch</b>. Trained on a '
        'stratified 75/25 split of the filtered rows. '
        'Headline metric is <b>PR-AUC</b> (average precision), not ROC-AUC — at a 1.5% '
        'base rate ROC-AUC flatters everything.</div>', unsafe_allow_html=True)
    st.write("")

    ftab = flag_table(df)
    cut = fa.deterministic_cut(ftab)
    if cut is not None:
        st.markdown(
            f'<div class="term">⚠ <b>READ THIS BEFORE TRUSTING THE SCORE.</b> The label in '
            f'this file is not noisy real-world fraud — it is <b>generated by a rule</b>. '
            f'Every transaction tripping <b>{cut} or more</b> of the five risk flags below is '
            f'fraud, <span class="bad">without exception</span>; almost everything below that '
            f'line is clean. So a model here does not "learn to detect fraud" — it '
            f'reverse-engineers the generator, and a near-perfect PR-AUC is the expected '
            f'result, not a good one.<br>'
            f'▸ flags: <b>{" · ".join(RISK_FLAGS)}</b><br>'
            f'▸ real fraud data never looks like this; treat the numbers below as a check '
            f'that the pipeline works, and use the tabs to the left for the actual analysis.'
            '</div>', unsafe_allow_html=True)
        st.plotly_chart(flag_step_fig(ftab, cut), width="stretch", key="mdl_flags")
        with st.expander("the rule, row by row"):
            st.dataframe(
                ftab, width="stretch", hide_index=True,
                column_config={
                    "n_flags": st.column_config.NumberColumn("flags tripped", format="%d"),
                    "transactions": st.column_config.NumberColumn("txn", format="%d"),
                    "fraud": st.column_config.NumberColumn("fraud", format="%d"),
                    "fraud_rate": st.column_config.NumberColumn("fraud rate", format="percent"),
                    "lift": st.column_config.NumberColumn("lift", format="%.1f×"),
                })
            st.caption(
                "The 6 fraud cases sitting at 2 flags are injected label noise — all six "
                "are high-value transactions ($950–$1,185) in the 00:00–02:00 window.")
        st.markdown("---")

    m1, m2 = st.columns([1, 2])
    with m1:
        algo = st.radio("model", ["gradient boosting", "logistic regression"],
                        key="mdl_algo")
    with m2:
        picked = st.multiselect("features given to the model", FEATURES, default=FEATURES,
                                format_func=lambda c: NAME[c], key="mdl_feats")

    if int(df[LABEL].sum()) < fm.MIN_FRAUD_TO_TRAIN:
        st.warning(f"Only {int(df[LABEL].sum())} fraud cases in the filtered set — "
                   f"need at least {fm.MIN_FRAUD_TO_TRAIN} to train a meaningful split. "
                   "Widen the filters.")
    elif not picked:
        st.info("Give the model at least one feature.")
    else:
        res = fm.train(df, LABEL, picked, KINDS, algo)
        pos = int(res["y_test"].sum())

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("PR-AUC", f"{res['ap']:.3f}",
                f"{res['ap']/res['no_skill']:.0f}× the {res['no_skill']:.3f} no-skill line")
        with k2:
            kpi("ROC-AUC", f"{res['roc']:.3f}", "secondary — inflated by imbalance")
        with k3:
            kpi("test set", f"{len(res['y_test']):,}", f"{pos} of them fraud")
        with k4:
            kpi("train set", f"{res['n_train']:,}", f"{res['n_train_pos']} of them fraud")

        st.markdown(
            f'<div class="term">▸ Only <b>{pos}</b> fraud cases sit in the test set, so '
            'these numbers carry wide error bars. Use this tab to compare features and '
            'pick an operating threshold — not to claim a production-ready score.</div>',
            unsafe_allow_html=True)
        st.write("")

        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(fm.pr_curve_fig(res, T, go, np), width="stretch", key="mdl_pr")
            st.caption("Recall = share of fraud caught. Precision = share of alerts "
                       "that really are fraud. The dotted floor is what guessing gets you.")
        with p2:
            st.plotly_chart(fm.importance_fig(res, NAME, T, go), width="stretch", key="mdl_imp")
            st.caption("How much PR-AUC the model loses when that column alone is "
                       "shuffled — measured on the held-out set, so it rewards columns "
                       "that genuinely carry signal.")

        st.markdown("### operating point")
        st.markdown(
            '<div class="term">Every alert costs a review. Slide the score cut-off to '
            'trade fraud caught against false positives your team has to work through.'
            '</div>', unsafe_allow_html=True)
        thr = st.slider("score threshold — flag a transaction above this", 0.01, 0.95,
                        0.30, 0.01, key="mdl_thr")
        o = fm.operating_point(res, df, thr)

        t1, t2, t3, t4, t5 = st.columns(5)
        with t1:
            kpi("alerts raised", f"{o['alerts']:,}",
                f"{o['alert_rate']:.1%} of the test book")
        with t2:
            kpi("fraud caught", f"{o['tp']:,}", f"recall {o['recall']:.0%}")
        with t3:
            kpi("fraud missed", f"{o['fn']:,}", "slipped through", alert=o["fn"] > 0)
        with t4:
            kpi("false positives", f"{o['fp']:,}",
                f"precision {o['precision']:.0%}" if o["alerts"] else "—")
        with t5:
            kpi("value recovered", money(o["amt_caught"]),
                f"{money(o['amt_leaked'])} leaked")

        st.write("")
        st.plotly_chart(fm.score_hist_fig(res, thr, T, go), width="stretch", key="mdl_scores")

# ---------------------------------------------------------------- 5. overview
with tab_over:
    st.markdown("## the shape of the filtered book")
    baseline = df[LABEL].mean()
    st.plotly_chart(hour_profile(df, baseline), width="stretch", key="ovw_hours")

    o1, o2 = st.columns(2)
    with o1:
        st.plotly_chart(
            lift_bar(fa.segment_table(df, "device_trust_score", LABEL, "numeric"),
                     "FRAUD RATE BY DEVICE TRUST SCORE", baseline),
            width="stretch", key="ovw_trust")
    with o2:
        st.plotly_chart(
            lift_bar(fa.segment_table(df, "velocity_last_24h", LABEL, "discrete"),
                     "FRAUD RATE BY TXNS IN LAST 24H", baseline),
            width="stretch", key="ovw_velocity")

    counts = df.groupby("merchant_category", observed=True)[LABEL].agg(["size", "sum", "mean"])
    counts = counts.sort_values("mean", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts.index, y=counts["size"] - counts["sum"], name="legitimate",
        marker=dict(color=T.GOOD, cornerradius=4, line=dict(width=2, color=T.BG)),
        hovertemplate="<b>%{x}</b><br>legitimate %{y:,}<extra></extra>"))
    fig.add_trace(go.Bar(
        x=counts.index, y=counts["sum"], name="fraud",
        marker=dict(color=T.CRITICAL, cornerradius=4, line=dict(width=2, color=T.BG)),
        text=[f"{v:.2%}" for v in counts["mean"]], textposition="outside",
        textfont=dict(color=T.INK_DIM, size=11), cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>fraud %{y:,}<extra></extra>"))
    fig.update_layout(**T.plotly_layout(
        title="VOLUME BY MERCHANT CATEGORY — labelled with each bar's fraud rate",
        height=370, barmode="stack", bargap=0.35,
        xaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=T.BORDER,
                   tickfont=dict(color=T.INK_DIM, size=11)),
        yaxis=dict(title="transactions", gridcolor=T.GRID, linecolor=T.BORDER,
                   tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED)),
    ))
    st.plotly_chart(fig, width="stretch", key="ovw_category")
    st.plotly_chart(class_distribution(df, "amount"), width="stretch", key="ovw_amount")

# ---------------------------------------------------------------- 6. data
with tab_data:
    st.markdown("## filtered transactions")
    only_fraud = st.checkbox("fraud only", value=False, key="tbl_fraud")
    view = df[df[LABEL] == 1] if only_fraud else df
    s1, s2 = st.columns([2, 1])
    with s1:
        sort_col = st.selectbox("sort by", list(view.columns),
                                format_func=lambda c: NAME.get(c, c), key="tbl_sort")
    with s2:
        desc = st.toggle("descending", value=True, key="tbl_desc")
    view = view.sort_values(sort_col, ascending=not desc)

    st.dataframe(view, width="stretch", hide_index=True, height=520)
    buf = io.StringIO()
    view.to_csv(buf, index=False)
    st.download_button("⤓ export csv", buf.getvalue(),
                       file_name="fraud_scan_export.csv", mime="text/csv")
