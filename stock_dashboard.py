import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock Returns Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0f1117;
    color: #e8e8e8;
}

.main { padding: 1.5rem 2rem; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.dashboard-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #f0f0f0;
    margin-bottom: 0.2rem;
}

.dashboard-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    color: #6b7280;
    margin-bottom: 1.5rem;
}

.stDataFrame { border-radius: 8px; overflow: hidden; }

.metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
}

.metric-label {
    font-size: 0.7rem;
    color: #6b7280;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
}

.positive { color: #22c55e; }
.negative { color: #ef4444; }
.neutral  { color: #e8e8e8; }

.last-updated {
    font-size: 0.72rem;
    color: #4b5563;
    font-family: 'IBM Plex Mono', monospace;
    text-align: right;
    margin-top: 0.5rem;
}

div[data-testid="stSidebar"] {
    background-color: #13151f;
    border-right: 1px solid #1e2030;
}

.stButton > button {
    background: #1e40af;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    padding: 0.4rem 1rem;
    width: 100%;
}

.stButton > button:hover { background: #2563eb; }

.stTextArea textarea {
    background: #1a1d27;
    color: #e8e8e8;
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
}

hr { border-color: #1e2030; }
</style>
""", unsafe_allow_html=True)

# ── Default stock list ────────────────────────────────────────────────────────
DEFAULT_STOCKS = {
    "TDPOWERSYS": "TD Power Systems",
    "WAAREEENER": "Waaree Energies",
    "CCL": "CCL Products",
    "SHRIRAMFIN": "Shriram Finance",
    "EMCURE": "Emcure Pharmaceuticals",
    "GALAXYSURF": "Galaxy Surfactants",
    "NATIONALUM": "National Aluminium",
    "TRAVELFOOD": "Travel Food Services",
    "ABCAPITAL": "Aditya Birla Capital",
    "APARINDS": "Apar Industries",
    "APLAPOLLO": "APL Apollo Tubes",
    "DIXON": "Dixon Technologies",
    "GRAVITA": "Gravita India",
    "SAGILITY": "Sagility",
    "SAILIFE": "Sai Life Sciences",
    "LGEINDIA": "LG Electronics India",
    "INOXWIND": "Inox Wind",
    "QPOWER": "Quality Power Electrical Equipments",
    "NAVINFLUOR": "Navin Fluorine International",
    "WABAG": "VA Tech Wabag",
    "ELECON": "Elecon Engineering",
    "FEDERALBNK": "Federal Bank",
    "LAURUSLABS": "Laurus Labs",
    "LT": "Larsen & Toubro",
    "NESTLEIND": "Nestlé India",
    "NH": "Narayana Hrudayalaya",
    "TVSMOTOR": "TVS Motor Company",
    "KPIL": "Kalpataru Projects International",
    "NETWEB": "Netweb Technologies India",
    "ENRIN": "Siemens Energy India",
    "VBL": "Varun Beverages",
    "SCHAEFFLER": "Schaeffler India",
    "MINDSPACE": "Mindspace Business Parks REIT",
    "SETL": "Standard Engineering Technology",
    "SHAKTIPUMP":   "Shakti Pumps",
    "POLYCAB":      "Polycab India",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_returns(symbols: list[str]) -> pd.DataFrame:
    """Download price history and compute multi-period returns."""
    today = datetime.today().date()
    start = today - timedelta(days=365 * 6)   # 6 yr buffer

    periods = {
        "1W":  today - timedelta(weeks=1),
        "1M":  today - timedelta(days=30),
        "3M":  today - timedelta(days=91),
        "6M":  today - timedelta(days=182),
        "1Y":  today - timedelta(days=365),
        "3Y":  today - timedelta(days=365 * 3),
        "5Y":  today - timedelta(days=365 * 5),
    }

    tickers = [f"{s}.NS" for s in symbols]
    raw = yf.download(tickers, start=str(start), end=str(today),
                      auto_adjust=True, progress=False)["Close"]

    # Handle single ticker (yfinance returns Series)
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    rows = []
    for sym, ticker in zip(symbols, tickers):
        if ticker not in raw.columns:
            continue
        prices = raw[ticker].dropna()
        if prices.empty:
            continue
        current = prices.iloc[-1]
        row = {"Symbol": sym}
        for label, past_date in periods.items():
            try:
                past_prices = prices[prices.index.date <= past_date]
                if past_prices.empty:
                    row[label] = None
                else:
                    past_price = past_prices.iloc[-1]
                    row[label] = (current - past_price) / past_price * 100
            except Exception:
                row[label] = None
        rows.append(row)

    return pd.DataFrame(rows).set_index("Symbol")


def color_column(col):
    """
    Column-wise percentile coloring.
    The best value in the column gets the darkest green,
    the worst gets the darkest red — regardless of absolute value.
    """
    styles = []
    valid = col.dropna()
    if valid.empty:
        return ["background-color: #1a1d27; color: #4b5563;"] * len(col)

    col_min = valid.min()
    col_max = valid.max()

    for val in col:
        if pd.isna(val):
            styles.append("background-color: #1a1d27; color: #4b5563;")
            continue

        # Normalise to 0-1 within this column
        if col_max == col_min:
            rank = 0.5
        else:
            rank = (val - col_min) / (col_max - col_min)

        # Map rank to colour
        if rank >= 0.85:
            bg, fg = "#14532d", "#86efac"
        elif rank >= 0.70:
            bg, fg = "#166534", "#bbf7d0"
        elif rank >= 0.55:
            bg, fg = "#1a4731", "#d1fae5"
        elif rank >= 0.45:
            bg, fg = "#1e2d1e", "#e8e8e8"
        elif rank >= 0.30:
            bg, fg = "#2d1e1e", "#e8e8e8"
        elif rank >= 0.15:
            bg, fg = "#3d1e1e", "#fca5a5"
        else:
            bg, fg = "#5c1a1a", "#fca5a5"

        styles.append(f"background-color: {bg}; color: {fg}; font-weight: 500;")

    return styles


def fmt(val):
    if pd.isna(val):
        return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Stocks")
    st.markdown("Enter NSE symbols (one per line):")

    default_text = "\n".join(
        [f"{sym}  # {name}" for sym, name in DEFAULT_STOCKS.items()]
    )
    raw_input = st.text_area("", value=default_text, height=420,
                              label_visibility="collapsed")

    refresh = st.button("🔄  Refresh Data")
    if refresh:
        st.cache_data.clear()

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.72rem; color:#4b5563; font-family: monospace;'>
Format: <code>SYMBOL  # Optional Name</code><br>
Uses NSE symbols via Yahoo Finance.<br>
Data refreshes every hour.
</div>
""", unsafe_allow_html=True)

# ── Parse sidebar input ───────────────────────────────────────────────────────
stock_map = {}
for line in raw_input.strip().splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split("#")
    sym = parts[0].strip().upper()
    name = parts[1].strip() if len(parts) > 1 else sym
    if sym:
        stock_map[sym] = name

symbols = list(stock_map.keys())

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="dashboard-title">📊 Indian Stock Returns</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">NSE · Multi-period return heatmap · Powered by Yahoo Finance</div>', unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
if not symbols:
    st.warning("Add at least one stock symbol in the sidebar.")
    st.stop()

with st.spinner("Fetching price data..."):
    df = fetch_returns(symbols)

if df.empty:
    st.error("No data returned. Check that your NSE symbols are correct.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
cols = st.columns(3)
period_labels = ["1W", "1M", "1Y"]
period_names  = ["1 Week", "1 Month", "1 Year"]

for col, period, label in zip(cols, period_labels, period_names):
    if period in df.columns:
        median_ret = df[period].median()
        green = int((df[period] > 0).sum())
        total = int(df[period].notna().sum())
        sign  = "positive" if median_ret >= 0 else "negative"
        sign_str = "+" if median_ret >= 0 else ""
        col.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value {sign}">{sign_str}{median_ret:.1f}%</div>
  <div style="font-size:0.7rem; color:#6b7280; margin-top:0.3rem;">
    {green}/{total} stocks positive
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main heatmap table ────────────────────────────────────────────────────────
display_df = df.copy()

# Add company name column
display_df.insert(0, "Company", [stock_map.get(s, s) for s in display_df.index])

# Style
styled = (
    display_df.style
    .apply(color_column, axis=0, subset=["1W", "1M", "3M", "6M", "1Y", "3Y", "5Y"])
    .format(fmt, subset=["1W", "1M", "3M", "6M", "1Y", "3Y", "5Y"])
    .set_properties(**{
        "font-family": "'IBM Plex Mono', monospace",
        "font-size":   "13px",
        "text-align":  "right",
        "padding":     "6px 14px",
    })
    .set_properties(subset=["Company"], **{
        "text-align":  "left",
        "color":       "#d1d5db",
        "font-family": "'IBM Plex Sans', sans-serif",
        "font-weight": "400",
    })
    .set_table_styles([
        {"selector": "thead th", "props": [
            ("background-color", "#13151f"),
            ("color", "#9ca3af"),
            ("font-family", "'IBM Plex Mono', monospace"),
            ("font-size", "11px"),
            ("letter-spacing", "0.06em"),
            ("text-transform", "uppercase"),
            ("padding", "8px 14px"),
            ("border-bottom", "1px solid #2a2d3a"),
            ("text-align", "right"),
        ]},
        {"selector": "tbody tr:nth-child(even)", "props": [
            ("background-color", "#13151f"),
        ]},
        {"selector": "tbody tr:nth-child(odd)", "props": [
            ("background-color", "#0f1117"),
        ]},
        {"selector": "tbody tr:hover", "props": [
            ("background-color", "#1e2030 !important"),
        ]},
        {"selector": "td, th", "props": [
            ("border", "none"),
        ]},
        {"selector": "table", "props": [
            ("border-collapse", "collapse"),
            ("width", "100%"),
        ]},
    ])
)

st.dataframe(styled, use_container_width=True, height=500)

# ── Legend ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; gap:1rem; margin-top:0.5rem; flex-wrap:wrap; align-items:center;'>
  <span style='font-size:0.72rem; font-family:monospace; color:#6b7280;'>Color scale (column-wise rank):</span>
  <span style='font-size:0.72rem; font-family:monospace; background:#5c1a1a; color:#fca5a5; padding:1px 8px; border-radius:3px;'>Worst</span>
  <span style='font-size:0.72rem; font-family:monospace; background:#3d1e1e; color:#fca5a5; padding:1px 8px; border-radius:3px;'>Low</span>
  <span style='font-size:0.72rem; font-family:monospace; background:#2d1e1e; color:#e8e8e8; padding:1px 8px; border-radius:3px;'>Below mid</span>
  <span style='font-size:0.72rem; font-family:monospace; background:#1e2d1e; color:#e8e8e8; padding:1px 8px; border-radius:3px;'>Above mid</span>
  <span style='font-size:0.72rem; font-family:monospace; background:#166534; color:#bbf7d0; padding:1px 8px; border-radius:3px;'>High</span>
  <span style='font-size:0.72rem; font-family:monospace; background:#14532d; color:#86efac; padding:1px 8px; border-radius:3px;'>Best</span>
  <span style='font-size:0.72rem; font-family:monospace; color:#4b5563; margin-left:auto;'>Last updated: {}</span>
</div>
""".format(datetime.now().strftime("%d %b %Y, %H:%M")), unsafe_allow_html=True)

# ── Best / Worst ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏆 Top Performers (1Y)")
    if "1Y" in df.columns:
        top = df["1Y"].dropna().nlargest(5)
        for sym, ret in top.items():
            name = stock_map.get(sym, sym)
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid #1e2030;'>"
                f"<span style='color:#d1d5db;font-size:0.82rem;'>{name}</span>"
                f"<span style='color:#22c55e;font-family:monospace;font-size:0.82rem;"
                f"font-weight:600;'>+{ret:.2f}%</span></div>",
                unsafe_allow_html=True
            )

with col2:
    st.markdown("#### 📉 Worst Performers (1Y)")
    if "1Y" in df.columns:
        bot = df["1Y"].dropna().nsmallest(5)
        for sym, ret in bot.items():
            name = stock_map.get(sym, sym)
            sign = "+" if ret > 0 else ""
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid #1e2030;'>"
                f"<span style='color:#d1d5db;font-size:0.82rem;'>{name}</span>"
                f"<span style='color:#ef4444;font-family:monospace;font-size:0.82rem;"
                f"font-weight:600;'>{sign}{ret:.2f}%</span></div>",
                unsafe_allow_html=True
            )
