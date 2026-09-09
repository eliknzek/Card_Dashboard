"""Hacker-terminal visual system for the fraud dashboard.

All chart colors below were validated with the data-viz palette validator against
the dark chart surface (#0b0f0b):

    node validate_palette.js "#09af51,#fe19a3,#bf860c,#3792fd,#00aa95" \
         --mode dark --surface "#0b0f0b"
    -> lightness band PASS | chroma floor PASS | CVD separation PASS
       (worst adjacent magenta<->green dE 9.5 deutan) | normal-vision PASS | contrast PASS

Scatter / all-pairs forms are capped at the first three slots
(green, magenta, blue), the only trio that clears the all-pairs floors.
"""

# --- surfaces & ink ---------------------------------------------------------
BG          = "#060806"   # page ground
SURFACE     = "#0b0f0b"   # chart surface (the surface the palette was validated against)
PANEL       = "#0f150f"   # cards / sidebar
BORDER      = "#1c2a1c"
BORDER_HOT  = "#00ff41"   # neon accent - UI chrome only, never a data mark

INK         = "#d8ffe4"   # primary text
INK_DIM     = "#7fa88c"   # secondary text
INK_MUTED   = "#4d6b57"   # axis / grid ink
GRID        = "#16211a"

NEON        = "#00ff41"   # the matrix green: headings, borders, glow

# --- categorical palette (adjacent-validated, dark) -------------------------
CATEGORICAL = ["#09af51", "#fe19a3", "#bf860c", "#3792fd", "#00aa95"]
CATEGORICAL_ALLPAIRS = CATEGORICAL[:3]   # scatter / bubble cap

# --- status palette (reserved - never reused as a categorical series) -------
CRITICAL = "#fe4339"   # fraud
WARNING  = "#bf860c"
GOOD     = "#09af51"   # legitimate
NEUTRAL  = "#3d4f42"

# --- sequential: one hue (green), dark -> bright ----------------------------
SEQ_GREEN = ["#003311", "#004b1e", "#03642b", "#0a7e3a",
             "#019a46", "#0cb655", "#17d365", "#0bf272"]

# --- diverging: red <-> green with a neutral gray midpoint ------------------
# used for lift (below / above the baseline fraud rate)
DIVERGING = ["#a40007", "#cb040f", "#f50012", "#fd5f52",
             "#2a332c",
             "#00672c", "#03823a", "#0c9d49", "#00ba56"]

MONO = "'JetBrains Mono','Fira Code','SF Mono',Menlo,Consolas,monospace"


def plotly_layout(**overrides):
    """Shared layout: recessive grid and axes, monospace ink, transparent surface."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO, size=12, color=INK_DIM),
        title=dict(font=dict(family=MONO, size=13, color=INK), x=0, xanchor="left"),
        margin=dict(l=8, r=8, t=44, b=8),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER,
                   tickfont=dict(color=INK_MUTED, size=11), title_font=dict(color=INK_MUTED)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER,
                   tickfont=dict(color=INK_MUTED, size=11), title_font=dict(color=INK_MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_DIM, size=11),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=NEON,
                        font=dict(family=MONO, color=INK, size=12)),
        colorway=CATEGORICAL,
    )
    layout.update(overrides)
    return layout


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

.stApp {{
    background:
        radial-gradient(1200px 600px at 15% -10%, #0a1a0d 0%, transparent 60%),
        {BG};
    color: {INK};
}}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {{
    font-family: {MONO} !important;
}}

/* scanline texture - the terminal tell, kept faint so it never fights the data */
.stApp::before {{
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 999;
    background: repeating-linear-gradient(
        to bottom, rgba(0,255,65,0.028) 0px, rgba(0,255,65,0.028) 1px,
        transparent 1px, transparent 3px);
}}

h1, h2, h3, h4 {{
    color: {NEON} !important;
    letter-spacing: .06em; text-transform: uppercase; font-weight: 700 !important;
    text-shadow: 0 0 10px rgba(0,255,65,.35);
}}
h1 {{ font-size: 1.55rem !important; }}
h2 {{ font-size: 1.1rem !important; }}
h3 {{ font-size: .95rem !important; }}

a, a:visited {{ color: {NEON} !important; }}

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {INK} ; }}
section[data-testid="stSidebar"] label p {{
    color: {INK_DIM} !important; font-size: .74rem !important;
    text-transform: uppercase; letter-spacing: .09em;
}}

/* ---- KPI tiles ---- */
.kpi {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-left: 2px solid {NEON};
    border-radius: 3px; padding: 12px 14px; height: 100%;
}}
.kpi .kpi-label {{
    color: {INK_MUTED}; font-size: .68rem; letter-spacing: .13em;
    text-transform: uppercase; margin-bottom: 6px;
}}
.kpi .kpi-value {{
    color: {NEON}; font-size: 1.7rem; font-weight: 700; line-height: 1.1;
    text-shadow: 0 0 14px rgba(0,255,65,.3);
}}
.kpi .kpi-value.alert {{ color: {CRITICAL}; text-shadow: 0 0 14px rgba(254,67,57,.3); }}
.kpi .kpi-sub {{ color: {INK_DIM}; font-size: .72rem; margin-top: 4px; }}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: {INK_MUTED};
    letter-spacing: .1em; text-transform: uppercase; font-size: .76rem;
    padding: 8px 14px; border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    color: {NEON} !important; border-bottom: 2px solid {NEON};
    background: rgba(0,255,65,.05);
}}

/* ---- widgets ---- */
.stButton button, .stDownloadButton button {{
    background: transparent; color: {NEON};
    border: 1px solid {NEON}; border-radius: 2px;
    text-transform: uppercase; letter-spacing: .1em; font-size: .75rem;
}}
.stButton button:hover, .stDownloadButton button:hover {{
    background: rgba(0,255,65,.12); color: {NEON}; border-color: {NEON};
}}
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
    background: {SURFACE} !important; border-color: {BORDER} !important; color: {INK} !important;
}}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 3px; }}

/* ---- readouts ---- */
.term {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-left: 2px solid {NEON};
    border-radius: 3px; padding: 10px 13px; color: {INK_DIM};
    font-size: .8rem; line-height: 1.65;
}}
.term b {{ color: {NEON}; font-weight: 700; }}
.term .bad {{ color: {CRITICAL}; font-weight: 700; }}
.tag {{
    display: inline-block; padding: 1px 7px; margin-right: 5px; border-radius: 2px;
    font-size: .68rem; letter-spacing: .08em; text-transform: uppercase;
    border: 1px solid currentColor;
}}
.tag.hot  {{ color: {CRITICAL}; }}
.tag.warm {{ color: {WARNING}; }}
.tag.cold {{ color: {INK_MUTED}; }}
</style>
"""
