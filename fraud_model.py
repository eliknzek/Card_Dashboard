"""Predictive layer: can a score built from these columns actually catch fraud?

Kept deliberately small and honest. At a ~1.5% base rate the metric that matters is
average precision (PR-AUC); ROC-AUC looks impressive for models that are useless in
production, so it is reported second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Below this many positives a 25% test split holds too few fraud cases to say
# anything at all, so the tab refuses to train rather than print noise.
MIN_FRAUD_TO_TRAIN = 40
TEST_SIZE = 0.25
SEED = 42


def _design_matrix(df: pd.DataFrame, features: list[str], kinds: dict[str, str]):
    """One frame the estimators can consume; categoricals become ordered codes."""
    X = pd.DataFrame(index=df.index)
    for col in features:
        if kinds[col] == "categorical":
            X[col] = pd.Categorical(df[col]).codes.astype(float)
        else:
            X[col] = df[col].astype(float)
    return X


@st.cache_data(show_spinner="training…")
def train(df: pd.DataFrame, label: str, features: list[str],
          kinds: dict[str, str], algo: str) -> dict:
    """Fit on a stratified 75/25 split and score the held-out quarter."""
    X = _design_matrix(df, features, kinds)
    y = df[label].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    if algo == "logistic regression":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
    else:
        cat_mask = [kinds[c] == "categorical" for c in features]
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=1.0,
            categorical_features=cat_mask if any(cat_mask) else None,
            class_weight="balanced", random_state=SEED)

    model.fit(X_tr, y_tr)
    scores = model.predict_proba(X_te)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_te, scores)

    # Permutation importance on the TEST set, scored by average precision - impurity
    # importance would reward columns for being high-cardinality rather than useful.
    perm = permutation_importance(model, X_te, y_te, scoring="average_precision",
                                  n_repeats=8, random_state=SEED, n_jobs=1)
    importance = (pd.DataFrame({"feature": features,
                                "importance": perm.importances_mean,
                                "std": perm.importances_std})
                  .sort_values("importance", ascending=False, ignore_index=True))

    return {
        "scores": scores,
        "y_test": y_te.to_numpy(),
        "test_index": X_te.index,
        "precision": precision, "recall": recall, "thresholds": thresholds,
        "ap": float(average_precision_score(y_te, scores)),
        "roc": float(roc_auc_score(y_te, scores)),
        "no_skill": float(y_te.mean()),
        "importance": importance,
        "n_train": int(len(y_tr)), "n_train_pos": int(y_tr.sum()),
    }


def operating_point(res: dict, df: pd.DataFrame, threshold: float) -> dict:
    """Confusion counts and money at one score cut-off, on the held-out set."""
    y, s = res["y_test"], res["scores"]
    flag = s >= threshold
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    alerts = tp + fp

    amounts = df.loc[res["test_index"], "amount"].to_numpy()
    return {
        "tp": tp, "fp": fp, "fn": fn, "alerts": alerts,
        "alert_rate": alerts / len(y) if len(y) else 0.0,
        "precision": tp / alerts if alerts else 0.0,
        "recall": tp / max(int(y.sum()), 1),
        "amt_caught": float(amounts[flag & (y == 1)].sum()),
        "amt_leaked": float(amounts[~flag & (y == 1)].sum()),
    }


# ---------------------------------------------------------------- figures
def pr_curve_fig(res, T, go, np):
    """Precision vs recall, with the no-skill floor. Two series -> legend present."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res["recall"], y=res["precision"], mode="lines", name="model",
        line=dict(color=T.CRITICAL, width=2),
        hovertemplate="recall %{x:.0%}<br>precision %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[res["no_skill"]] * 2, mode="lines", name="no skill",
        line=dict(color=T.INK_MUTED, width=1, dash="dot"),
        hovertemplate=f"no skill {res['no_skill']:.1%}<extra></extra>"))
    fig.update_layout(**T.plotly_layout(
        title=f"PRECISION–RECALL  ·  PR-AUC {res['ap']:.3f}", height=360,
        margin=dict(l=56, r=8, t=66, b=44),
        xaxis=dict(title="recall", tickformat=".0%", range=[0, 1],
                   gridcolor=T.GRID, linecolor=T.BORDER,
                   tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED)),
        yaxis=dict(title="precision", tickformat=".0%", range=[0, 1],
                   gridcolor=T.GRID, linecolor=T.BORDER,
                   tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED)),
    ))
    return fig


def importance_fig(res, names, T, go):
    """Permutation importance: how much PR-AUC is lost when a column is shuffled."""
    imp = res["importance"].iloc[::-1]
    fig = go.Figure(go.Bar(
        x=imp["importance"], y=[names.get(f, f) for f in imp["feature"]],
        orientation="h", error_x=dict(array=imp["std"], color=T.INK_MUTED, width=0,
                                      thickness=1),
        marker=dict(color=T.DIVERGING[3], cornerradius=4, line=dict(width=0)),
        text=[f"{v:.3f}" for v in imp["importance"]], textposition="outside",
        textfont=dict(color=T.INK_DIM, size=11), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>PR-AUC lost when shuffled %{x:.4f}<extra></extra>"))
    fig.update_layout(**T.plotly_layout(
        title="PERMUTATION IMPORTANCE", height=360, bargap=0.35, showlegend=False,
        margin=dict(l=8, r=64, t=66, b=40),
        xaxis=dict(gridcolor=T.GRID, linecolor=T.BORDER, zerolinecolor=T.BORDER,
                   tickfont=dict(color=T.INK_MUTED, size=11)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=T.BORDER,
                   tickfont=dict(color=T.INK_DIM, size=11)),
    ))
    return fig


def score_hist_fig(res, threshold, T, go):
    """Where the two classes land on the score line, and where the cut-off sits."""
    y, s = res["y_test"], res["scores"]
    fig = go.Figure()
    for value, name, color in ((0, "legitimate", T.GOOD), (1, "fraud", T.CRITICAL)):
        fig.add_trace(go.Histogram(
            x=s[y == value], name=name, histnorm="probability density",
            nbinsx=50, opacity=0.62, marker=dict(color=color, line=dict(width=0)),
            hovertemplate=f"<b>{name}</b><br>score %{{x:.2f}}<extra></extra>"))
    fig.add_vline(x=threshold, line=dict(color=T.NEON, width=2),
                  annotation_text=f"threshold {threshold:.2f}",
                  annotation_font=dict(color=T.NEON, size=11))
    fig.update_layout(**T.plotly_layout(
        title="SCORE SEPARATION — held-out transactions by predicted fraud score",
        height=340, barmode="overlay", bargap=0.05,
        xaxis=dict(title="predicted fraud score", gridcolor=T.GRID, linecolor=T.BORDER,
                   tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED)),
        yaxis=dict(title="density (each class to itself)", gridcolor=T.GRID,
                   linecolor=T.BORDER, tickfont=dict(color=T.INK_MUTED, size=11),
                   title_font=dict(color=T.INK_MUTED)),
    ))
    return fig
