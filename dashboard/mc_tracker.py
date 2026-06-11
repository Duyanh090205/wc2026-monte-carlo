"""MC daily tracker dashboard — model vs market time series from Supabase.

Deploy on Streamlit Community Cloud pointing at this file; set in app secrets:
    SUPABASE_URL = "https://<project>.supabase.co"
    SUPABASE_ANON_KEY = "<anon key>"
Local: streamlit run dashboard/mc_tracker.py
(falls back to data/mc_simu/daily_log.csv when no credentials are set).
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="MC vs Market — WC2026", layout="wide", page_icon="📈")

MODEL_C, MARKET_C = "#1f77b4", "#444444"
UP_C, DOWN_C = "#2a8a2a", "#cc3b2f"
TPL = "plotly_white"


def _cred(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets.toml at all -> fall through to env
        pass
    return os.environ.get(name, "")


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


def spread_labels(xn: np.ndarray, yn: np.ndarray, labeled: list) -> list:
    """Assign per-point textposition so labels of nearby points don't overlap."""
    cycle = ["top center", "bottom center", "middle right", "middle left"]
    pos = ["top center"] * len(xn)
    placed = []
    for i in sorted(range(len(xn)), key=lambda k: -(xn[k] + yn[k])):
        if not labeled[i]:
            continue
        taken = {pos[j] for j in placed
                 if abs(xn[i] - xn[j]) < 0.09 and abs(yn[i] - yn[j]) < 0.07}
        pos[i] = next((p for p in cycle if p not in taken), "top center")
        placed.append(i)
    return pos


def jsd_pct(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """Base-2 JSD, mirrors mc_simu.tune_to_market.jsd so numbers match run logs."""
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


df = load_log()
dates = sorted(df["date"].unique())

st.title("MC simulator vs market — WC2026 daily tracker")
st.caption("Model: ELO + squad-MV + star (static, locked pre-tournament) — re-conditioned daily "
           "on played results only. Goal: track the market with a stable, understood bias; "
           "an unusual divergence from that bias is the signal.")

day = st.sidebar.selectbox("Snapshot day", [pd.Timestamp(d).date() for d in reversed(dates)])
top_n = st.sidebar.slider("Teams shown in charts", 10, 48, 20)
snap = df[df["date"] == pd.Timestamp(day)].sort_values("model_pct", ascending=False)

l1 = (snap["model_pct"] - snap["consensus_pct"]).abs().sum()
j = jsd_pct(snap["model_pct"].to_numpy(), snap["consensus_pct"].to_numpy())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Days tracked", len(dates),
          help="Number of daily snapshots in the log (one per cron run).")
m2.metric("JSD vs market", f"{j:.4f}",
          help="Jensen-Shannon divergence (base 2) between the model's 48-team champion "
               "distribution and the market consensus distribution. 0 = identical, "
               "1 = no overlap. Our single closeness number — same figure the daily run logs.")
m3.metric("L1 distance", f"{l1:.1f} pp",
          help="Sum of |model − market| over all teams, in percentage points. "
               "It double-counts: 23 pp means ~11.5 pp of probability mass sits on "
               "different teams than the market puts it on.")
m4.metric("Max |edge|", f"{snap['abs_pp'].abs().max():.2f} pp",
          help="Largest single-team gap |model − market| in today's snapshot.")

with st.expander("How to read these numbers"):
    st.markdown(
        "- **JSD / L1** — distribution-level closeness between model and market; expect them "
        "to be roughly stable day to day. A sudden jump = model and market disagree about "
        "newly revealed information (a result, an injury) — that day is worth a look.\n"
        "- **Absolute edge (abs pp)** — `model% − market%` per team, in percentage points. "
        "Positive (green): model rates the team higher than the market.\n"
        "- **Relative edge (rel %)** — `(market − model) / model`. +100% = the market prices "
        "the team at double the model's probability. Large positive values on small teams "
        "are the classic favorite-longshot premium, not necessarily mispricing.\n"
        "- The model is **static** (locked pre-tournament): day-to-day movement comes only "
        "from re-conditioning on played results, never from re-fitting. So a *stable* bias "
        "vs the market is expected and fine — the signal is a *change* in that bias.")

tab_today, tab_scatter, tab_traj, tab_stab, tab_data = st.tabs(
    ["Today's edge", "Model vs market", "Trajectories", "Bias stability", "Data"])


with tab_today:
    st.caption("Where model and market disagree today. Left: the raw gap per team. "
               "Right: the same gap relative to the model's own number — this is where the "
               "longshot premium becomes visible.")
    c_abs, c_rel = st.columns(2)
    sub = snap.reindex(snap["abs_pp"].abs().sort_values(ascending=False).index).head(top_n)
    sub = sub.sort_values("abs_pp")
    fig = go.Figure(go.Bar(
        x=sub["abs_pp"], y=sub["team"], orientation="h",
        marker_color=[UP_C if v > 0 else DOWN_C for v in sub["abs_pp"]],
        text=[f"{v:+.2f}" for v in sub["abs_pp"]], textposition="outside",
        cliponaxis=False))
    fig.update_layout(template=TPL, height=26 * len(sub) + 120,
                      title="Absolute edge — model − market (pp)<br>"
                            "<sup>green: model above market · red: model below</sup>",
                      xaxis_title="model − market (pp)", margin=dict(l=10, r=40))
    fig.add_vline(x=0, line_color="black", line_width=1)
    c_abs.plotly_chart(fig, width="stretch")

    mc_floor = 0.02
    rel = snap.dropna(subset=["rel_pct"])
    rel = rel[rel["model_pct"] >= mc_floor]
    n_noise = len(snap) - len(rel)
    if n_noise:
        c_rel.caption(f"{n_noise} longshots hidden (model < {mc_floor}% = under ~10 of 50k MC hits): "
                      "their relative edge is sampling noise, not signal — read them on the absolute chart")
    rel = rel.reindex(rel["rel_pct"].abs().sort_values(ascending=False).index).head(top_n)
    rel = rel.sort_values("rel_pct")
    fig = go.Figure(go.Bar(
        x=rel["rel_pct"], y=rel["team"], orientation="h",
        marker_color=[DOWN_C if v > 0 else UP_C for v in rel["rel_pct"]],
        text=[f"{v:+.0f}%" for v in rel["rel_pct"]], textposition="outside",
        cliponaxis=False))
    fig.update_layout(template=TPL, height=26 * len(rel) + 120,
                      title="Relative edge — (market − model) / model (%)<br>"
                            "<sup>red: market prices the team RICHER than model (longshot premium)</sup>",
                      xaxis_title="(market − model) / model (%)", margin=dict(l=10, r=50))
    fig.add_vline(x=0, line_color="black", line_width=1)
    c_rel.plotly_chart(fig, width="stretch")


with tab_scatter:
    st.caption("One dot per team: x = what the market says, y = what the model says. "
               "On the dashed diagonal the two agree; vertical distance from it is the edge "
               "(dot colour). Log scale spreads out the longshots in the bottom-left.")
    log_axes = st.toggle("Log scale (see the longshot tail)", value=True)
    s = snap[(snap["model_pct"] > 0) & (snap["consensus_pct"] > 0)]
    n_zero = len(snap) - len(s)
    if n_zero:
        st.caption(f"{n_zero} teams hidden: model gives them 0% (log axes cannot show 0)")
    lim_lo = min(s["model_pct"].min(), s["consensus_pct"].min()) * 0.7
    lim_hi = max(s["model_pct"].max(), s["consensus_pct"].max()) * 1.3
    fig = go.Figure()
    fig.add_shape(type="line", x0=lim_lo, y0=lim_lo, x1=lim_hi, y1=lim_hi,
                  line=dict(color="gray", width=1.5, dash="dash"))
    labels = [t if (r["model_pct"] > 2 or r["consensus_pct"] > 2) else ""
              for t, (_, r) in zip(s["team"], s.iterrows())]
    ft = np.log10 if log_axes else np.asarray
    span = float(ft(lim_hi) - ft(lim_lo))
    xn = (ft(s["consensus_pct"].to_numpy(dtype=float)) - ft(lim_lo)) / span
    yn = (ft(s["model_pct"].to_numpy(dtype=float)) - ft(lim_lo)) / span
    fig.add_scatter(
        x=s["consensus_pct"], y=s["model_pct"], mode="markers+text",
        text=labels,
        textposition=spread_labels(xn, yn, [bool(t) for t in labels]),
        textfont_size=10,
        marker=dict(size=9, color=s["abs_pp"], colorscale="RdYlGn", cmid=0,
                    colorbar=dict(title="edge pp")),
        customdata=s["team"], name="teams",
        hovertemplate="%{customdata}<br>market %{x:.2f}% · model %{y:.2f}%<extra></extra>")
    ax = dict(type="log" if log_axes else "linear", range=None)
    if log_axes:
        ax["range"] = [np.log10(lim_lo), np.log10(lim_hi)]
    fig.update_layout(template=TPL, height=620, showlegend=False,
                      title="Model vs market — points above the dashed line = model higher than market<br>"
                            "<sup>distance from the diagonal IS the edge; the tail shows the favorite-longshot pattern</sup>",
                      xaxis={**ax, "title": "market consensus (%)"},
                      yaxis={**ax, "title": "model (%)"})
    st.plotly_chart(fig, width="stretch")


with tab_traj:
    st.caption("Champion probability through the tournament: solid line = model "
               "(re-conditioned daily on results), ✕ = market consensus the same day. "
               "Diverging line and ✕ = model and market reading the bracket differently.")
    if len(dates) == 1:
        st.info("Trajectories build up as daily snapshots accumulate — come back after a few days. "
                "Backtest preview of what this becomes: WC2022 replay in mc_simu/audits/.")
    default_teams = list(snap.head(6)["team"])
    teams = st.multiselect("Teams", sorted(df["team"].unique()), default=default_teams)
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, t in enumerate(teams):
        subt = df[df["team"] == t]
        c = palette[i % len(palette)]
        fig.add_scatter(x=subt["date"], y=subt["model_pct"], name=t,
                        mode="lines+markers", line=dict(color=c, width=2))
        fig.add_scatter(x=subt["date"], y=subt["consensus_pct"], name=f"{t} market",
                        mode="markers", marker=dict(color=c, symbol="x", size=10),
                        showlegend=False)
    x0 = pd.Timestamp(min(dates)) - pd.Timedelta(hours=12)
    x1 = pd.Timestamp(max(dates)) + pd.Timedelta(hours=12)
    fig.update_layout(template=TPL, height=520, hovermode="x unified",
                      title="Champion probability — model (line) vs market (✕)",
                      yaxis_title="champion prob (%)",
                      xaxis=dict(range=[x0, x1], tickformat="%b %d", dtick=86_400_000))
    st.plotly_chart(fig, width="stretch")


with tab_stab:
    st.caption("Is the model's bias vs the market stable? Left: total distance per day "
               "(JSD + L1) — flat is good. Right: per-team edge heatmap — a row keeping its "
               "colour is an understood bias; a row flipping colour is the anomaly to investigate.")
    c_l1, c_hm = st.columns([1, 2])
    daily = df.groupby("date").apply(
        lambda g: pd.Series({
            "L1 (pp)": (g["model_pct"] - g["consensus_pct"]).abs().sum(),
            "JSD": jsd_pct(g["model_pct"].to_numpy(), g["consensus_pct"].to_numpy()),
        }), include_groups=False).reset_index()
    fig = px.line(daily, x="date", y="L1 (pp)", markers=True, template=TPL, height=300,
                  title="Total model–market distance per day")
    c_l1.plotly_chart(fig, width="stretch")
    fig = px.line(daily, x="date", y="JSD", markers=True, template=TPL, height=300,
                  title="JSD per day")
    c_l1.plotly_chart(fig, width="stretch")

    top_teams = (df.groupby("team")["consensus_pct"].max()
                 .sort_values(ascending=False).head(top_n).index)
    piv = (df[df["team"].isin(top_teams)]
           .pivot_table(index="team", columns="date", values="abs_pp")
           .reindex(top_teams))
    piv.columns = [pd.Timestamp(c).strftime("%m-%d") for c in piv.columns]
    fig = px.imshow(piv, color_continuous_scale="RdYlGn", zmin=-3, zmax=3, aspect="auto",
                    template=TPL, height=26 * len(piv) + 140,
                    title="Per-team edge through time (pp) — a STABLE row colour = understood bias;<br>"
                          "<sup>a row that flips colour is the anomaly worth investigating</sup>")
    fig.update_layout(coloraxis_colorbar_title="pp")
    c_hm.plotly_chart(fig, width="stretch")


with tab_data:
    st.caption("Raw snapshot for the selected day. pm/kalshi = platform mid prices; "
               "consensus = their normalized median; blank cells = platform doesn't quote "
               "that team. Download gives the full multi-day log.")
    show = snap[["team", "model_pct", "pm_pct", "kalshi_pct", "consensus_pct", "abs_pp", "rel_pct"]]
    st.dataframe(
        show.style.background_gradient(subset=["abs_pp"], cmap="RdYlGn", vmin=-3, vmax=3)
            .format({"model_pct": "{:.2f}", "pm_pct": "{:.2f}", "kalshi_pct": "{:.2f}",
                     "consensus_pct": "{:.2f}", "abs_pp": "{:+.2f}", "rel_pct": "{:+.0f}%"},
                    na_rep="—"),
        width="stretch", hide_index=True, height=620)
    st.download_button("Download full log (CSV)", df.to_csv(index=False),
                       file_name="daily_log.csv", mime="text/csv")
