"""MC daily tracker dashboard — model vs market time series from Supabase.

Deploy on Streamlit Community Cloud pointing at this file; set in app secrets:
    SUPABASE_URL = "https://<project>.supabase.co"
    SUPABASE_ANON_KEY = "<anon key>"
Local: streamlit run dashboard/mc_tracker.py (reads the same two env vars).
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="MC vs Market — WC2026", layout="wide")


def _cred(name):
    return st.secrets.get(name, os.environ.get(name, ""))


LOCAL_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "mc_simu", "daily_log.csv")


@st.cache_data(ttl=600)
def load_log() -> pd.DataFrame:
    url, key = _cred("SUPABASE_URL"), _cred("SUPABASE_ANON_KEY")
    if url and key:
        r = requests.get(
            f"{url.rstrip('/')}/rest/v1/daily_log?select=*&order=date.asc&limit=100000",
            headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
    elif os.path.exists(LOCAL_CSV):
        st.info("No Supabase credentials — local preview from data/mc_simu/daily_log.csv")
        df = pd.read_csv(LOCAL_CSV)
    else:
        st.error("Set SUPABASE_URL + SUPABASE_ANON_KEY in secrets, or generate "
                 "data/mc_simu/daily_log.csv with `python -m mc_simu.run_daily`")
        st.stop()
    if df.empty:
        st.warning("daily_log is empty — first scheduled run hasn't landed yet")
        st.stop()
    df["date"] = pd.to_datetime(df["date"])
    for c in ("model_pct", "pm_pct", "kalshi_pct", "consensus_pct", "abs_pp", "rel_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


df = load_log()
dates = sorted(df["date"].unique())
latest = df[df["date"] == dates[-1]].sort_values("model_pct", ascending=False)

st.title("MC simulator vs market — WC2026 daily tracker")
st.caption(f"{len(dates)} snapshot days · last: {pd.Timestamp(dates[-1]).date()} · "
           "model = ELO+MV+star, static, re-conditioned on played results daily")

c1, c2, c3 = st.columns(3)
l1 = (latest["model_pct"] - latest["consensus_pct"]).abs().sum()
c1.metric("Teams quoted", len(latest))
c2.metric("L1 distance (latest)", f"{l1:.1f} pp")
c3.metric("Max |edge| (latest)", f"{latest['abs_pp'].abs().max():.2f} pp")

st.subheader("Champion probability through time — model (line) vs market (dots)")
default_teams = list(latest.head(6)["team"])
teams = st.multiselect("Teams", sorted(df["team"].unique()), default=default_teams)
fig = go.Figure()
palette = px.colors.qualitative.Plotly
for i, t in enumerate(teams):
    sub = df[df["team"] == t]
    c = palette[i % len(palette)]
    fig.add_scatter(x=sub["date"], y=sub["model_pct"], name=f"{t} model",
                    mode="lines+markers", line=dict(color=c, width=2))
    fig.add_scatter(x=sub["date"], y=sub["consensus_pct"], name=f"{t} market",
                    mode="markers", marker=dict(color=c, symbol="x", size=9),
                    showlegend=False)
fig.update_layout(height=480, yaxis_title="champion prob (%)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Today's edge board (model − market)")
board = latest[["team", "model_pct", "consensus_pct", "abs_pp", "rel_pct"]].copy()
board = board.reindex(board["abs_pp"].abs().sort_values(ascending=False).index)
st.dataframe(board, use_container_width=True, hide_index=True, height=420)

st.subheader("Model–market distance per day (is our bias stable?)")
daily = df.groupby("date").apply(
    lambda g: (g["model_pct"] - g["consensus_pct"]).abs().sum(), include_groups=False
).rename("L1 (pp)").reset_index()
st.plotly_chart(px.line(daily, x="date", y="L1 (pp)", markers=True, height=300),
                use_container_width=True)
