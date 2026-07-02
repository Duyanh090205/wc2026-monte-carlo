"""MC daily tracker dashboard — model vs market time series from Supabase.

Deploy on Streamlit Community Cloud pointing at this file; set in app secrets:
    SUPABASE_URL = "https://<project>.supabase.co"
    SUPABASE_ANON_KEY = "<anon key>"
Local: streamlit run dashboard/mc_tracker.py
(falls back to data/mc_simu/daily_log.csv when no credentials are set).
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mc_simu.tournaments.wc2026 import (  # bracket topology — single source of truth
    LATER_ROUNDS as KO_ROUNDS, R32_BRACKET as KO_R32,
)

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
    for c in ("model_pct", "pm_pct", "kalshi_pct", "consensus_pct", "abs_pp", "rel_pct",
              "mle_pct", "pool_pct", "pm_bid", "pm_ask", "kalshi_bid", "kalshi_ask"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


GW_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "mc_simu", "group_winner_log.csv")


@st.cache_data(ttl=600)
def load_gw() -> pd.DataFrame:
    """Group-winner series (model vs Polymarket). Supabase table group_winner_log
    when present, else the local backfill CSV. Empty frame when neither exists —
    the tab degrades to an info box rather than erroring."""
    url, key = _cred("SUPABASE_URL"), _cred("SUPABASE_ANON_KEY")
    df = pd.DataFrame()
    if url and key:
        r = requests.get(
            f"{url.rstrip('/')}/rest/v1/group_winner_log?select=*&order=date.asc&limit=100000",
            headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=30)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
    if df.empty and os.path.exists(GW_CSV):
        df = pd.read_csv(GW_CSV)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in ("model_pct", "pm_raw_pct", "pm_devig_pct", "model_state_matches"):
        if c in df.columns:
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


KO_STAGE_X = {**{m: 0 for m in range(73, 89)}, **{m: 1 for m in range(89, 97)},
              **{m: 2 for m in range(97, 101)}, 101: 3, 102: 3, 104: 4}


def prob_bar(p_home: float, p_draw: float, p_away: float) -> str:
    return (f'<div style="display:flex;width:150px;height:7px;border-radius:3px;'
            f'overflow:hidden;margin:2px 0 7px 0">'
            f'<div style="width:{p_home * 100:.0f}%;background:{UP_C}"></div>'
            f'<div style="width:{p_draw * 100:.0f}%;background:#9a9a9a"></div>'
            f'<div style="width:{p_away * 100:.0f}%;background:{DOWN_C}"></div></div>')


def bracket_figure(ko: pd.DataFrame, score_col: str = "score_pred") -> go.Figure:
    """Fixed FIFA bracket tree; boxes fill in from the predictions CSV as the
    tournament resolves. Drawn from the same constants the simulator uses."""
    children = {mid: (left, right) for mid, left, right in KO_ROUNDS}

    def slot(src: tuple) -> str:
        kind, key = src
        if kind == "W":
            return f"Group {key} winner"
        if kind == "RU":
            return f"Group {key} runner-up"
        return "best 3rd place"

    slots = {mid: (slot(left), slot(right)) for mid, left, right in KO_R32}
    leaves: list = []

    def walk(mid: int) -> None:
        if mid in children:
            walk(children[mid][0])
            walk(children[mid][1])
        else:
            leaves.append(mid)

    walk(104)
    ypos = {mid: float(i) for i, mid in enumerate(leaves)}

    def ynode(mid: int) -> float:
        if mid not in ypos:
            left, right = children[mid]
            ypos[mid] = (ynode(left) + ynode(right)) / 2
        return ypos[mid]

    ynode(104)
    by_mid = ({int(r["match_id"]): r for _, r in ko.iterrows()} if len(ko) else {})

    def short(t: str) -> str:
        return t if len(t) <= 14 else t[:13] + "…"

    fig = go.Figure()
    hover_x, hover_y, hover_t = [], [], []
    for mid, yv in ypos.items():
        x = KO_STAGE_X[mid]
        row = by_mid.get(mid)
        if row is not None:
            a, b, w = row["home_team"], row["away_team"], row["winner"]
            w = w if isinstance(w, str) and w else None
            la = f"{short(a)} {row['p_home'] * 100:.0f}%"
            lb = f"{short(b)} {row['p_away'] * 100:.0f}%"
            if w == a:
                la = f"<b><span style='color:{UP_C}'>{la} ✓</span></b>"
            elif w == b:
                lb = f"<b><span style='color:{UP_C}'>{lb} ✓</span></b>"
            sp = row.get(score_col)
            sp = sp if isinstance(sp, str) and sp else None
            sp_prob = row.get("score_prob") if score_col == "score_pred" else None
            sp_txt = ""
            if sp:
                sp_txt = f" · likely {sp} (90')"
                if sp_prob is not None and pd.notna(sp_prob):
                    sp_txt = f" · likely {sp} ({float(sp_prob) * 100:.0f}%, 90')"
            text, border = la + "<br>" + lb, "rgba(127,127,127,0.55)"
            if sp:
                text += (f"<br><i><span style='color:rgba(127,127,127,0.9)'>"
                         f"likely {sp}</span></i>")
            hover_t.append(f"M{mid}: {a} {row['p_home'] * 100:.0f}% — "
                           f"{b} {row['p_away'] * 100:.0f}%{sp_txt}"
                           + (f" · advanced: {w}" if w else ""))
        else:
            if mid in slots:
                pa, pb = slots[mid]
            else:
                pa, pb = f"M{children[mid][0]} winner", f"M{children[mid][1]} winner"
            text, border = f"<i>{pa}</i><br><i>{pb}</i>", "rgba(127,127,127,0.25)"
            hover_t.append(f"M{mid}: waiting for {pa} vs {pb}")
        fig.add_annotation(x=x, y=yv, text=text, showarrow=False, align="left",
                           font=dict(size=10), bordercolor=border, borderwidth=1,
                           borderpad=4, bgcolor="rgba(127,127,127,0.08)")
        hover_x.append(x)
        hover_y.append(yv)
        if mid in children:
            for c in children[mid]:
                fig.add_shape(type="line", x0=KO_STAGE_X[c] + 0.34, y0=ypos[c],
                              x1=x - 0.34, y1=yv,
                              line=dict(color="rgba(127,127,127,0.35)", width=1))
    fig.add_scatter(x=hover_x, y=hover_y, mode="markers",
                    marker=dict(size=26, opacity=0), hovertext=hover_t,
                    hoverinfo="text", showlegend=False)
    final_row = by_mid.get(104)
    if final_row is not None and isinstance(final_row["winner"], str) and final_row["winner"]:
        fig.add_annotation(x=4, y=ynode(104) - 1.4, showarrow=False, font=dict(size=13),
                           text=f"🏆 <b>{final_row['winner']}</b>")
    fig.update_layout(template=TPL, height=860, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(tickvals=[0, 1, 2, 3, 4],
                                 ticktext=["Round of 32", "Round of 16", "Quarter-finals",
                                           "Semi-finals", "Final"],
                                 range=[-0.55, 4.8], showgrid=False, zeroline=False,
                                 side="top"),
                      yaxis=dict(visible=False, autorange="reversed"))
    return fig


def apply_market_source(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Set mkt_pct to the chosen source; for a single platform, recompute edges
    per day over the teams that platform quotes, with both sides renormalized."""
    if col == "consensus_pct":
        out = df.copy()
        out["mkt_pct"] = out["consensus_pct"]
        return out
    out = df.dropna(subset=[col]).copy()
    out["model_pct"] = out["model_pct"] * 100 / out.groupby("date")["model_pct"].transform("sum")
    out["mkt_pct"] = out[col] * 100 / out.groupby("date")[col].transform("sum")
    out["abs_pp"] = out["model_pct"] - out["mkt_pct"]
    out["rel_pct"] = np.where(out["model_pct"] > 0,
                              (out["mkt_pct"] - out["model_pct"]) / out["model_pct"] * 100,
                              np.nan)
    return out


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
           "an unusual divergence from that bias is the signal. "
           "New snapshot daily at 08:30 UTC (04:30 ET), after the previous matchday settles.")

day = st.sidebar.selectbox("Snapshot day", [pd.Timestamp(d).date() for d in reversed(dates)])
top_n = st.sidebar.slider("Teams shown in charts", 10, 48, 20)
market_src = st.sidebar.radio(
    "Market reference", ["Polymarket", "Consensus", "Kalshi"],
    help="Consensus = normalized median of platform mids, exactly as logged by the daily run. "
         "Picking one platform recomputes every edge, JSD and L1 against that platform's mid "
         "prices, renormalized over only the teams it quotes.")
SRC_COL = {"Consensus": "consensus_pct", "Polymarket": "pm_pct", "Kalshi": "kalshi_pct"}
MODEL_COL = {"Production (ELO+MV+star)": "model_pct",
             "MLE strength": "mle_pct",
             "Pool 50/50": "pool_pct"}
model_choices = [k for k, v in MODEL_COL.items()
                 if v == "model_pct" or (v in df.columns and df[v].notna().any())]
model_src = st.sidebar.radio(
    "Model source", model_choices,
    help="Production = locked ELO+MV+star. MLE strength = parallel weighted-MLE "
         "rating source (results-only, frozen artifact). Pool = 50/50 log-opinion "
         "pool of the two — the consensus prior. Every edge, JSD and chart "
         "recomputes against the picked source. Sources other than Production "
         "exist from the date the parallel tracking went live.")
if MODEL_COL[model_src] != "model_pct":
    df = df[df[MODEL_COL[model_src]].notna()].copy()
    df["model_pct"] = df[MODEL_COL[model_src]]
df = apply_market_source(df, SRC_COL[market_src])
snap = df[df["date"] == pd.Timestamp(day)].sort_values("model_pct", ascending=False)
if snap.empty:
    st.warning(f"No {model_src} rows on {day} — parallel sources start from the day "
               "the dual-model tracking went live. Pick a later snapshot day.")
    st.stop()
st.sidebar.caption(f"{market_src}: {len(snap)} teams quoted on {day}")
with st.sidebar.expander("Data sources"):
    st.markdown(
        "- **Team strength** — Elo ratings, [eloratings.net](https://www.eloratings.net) "
        "(pre-tournament snapshot, frozen)\n"
        "- **Squad value** — [Transfermarkt](https://www.transfermarkt.com) squad market "
        "values, blended into Elo (alpha 0.5)\n"
        "- **Star bonus** — hand-curated list (Transfermarkt MV thresholds + "
        "Ballon d'Or top-30), +15 Elo per star\n"
        "- **Market prices** — Polymarket + Kalshi mid prices via the FairLine API\n"
        "- **Match results** — [football-data.org](https://www.football-data.org), "
        "auto-fetched daily\n"
        "- **Bracket & fixtures** — FIFA final draw, Wikipedia snapshots")

l1 = (snap["model_pct"] - snap["mkt_pct"]).abs().sum()
j = jsd_pct(snap["model_pct"].to_numpy(), snap["mkt_pct"].to_numpy())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Days tracked", len(dates),
          help="Number of daily snapshots in the log (one per cron run).")
m2.metric(f"JSD vs {market_src}", f"{j:.4f}",
          help="Jensen-Shannon divergence (base 2) between the model's champion "
               "distribution and the selected market's distribution. 0 = identical, "
               "1 = no overlap. Our single closeness number — the daily run logs the "
               "consensus variant of this figure.")
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

(tab_today, tab_scatter, tab_traj, tab_gw, tab_bracket, tab_score, tab_stab,
 tab_data) = st.tabs(
    ["Today's edge", "Model vs market", "Trajectories", "Group winner", "Bracket",
     "Scorecard", "Bias stability", "Data"])


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
                      title=f"Absolute edge — model − {market_src} (pp)<br>"
                            "<sup>green: model above market · red: model below</sup>",
                      xaxis_title=f"model − {market_src} (pp)", margin=dict(l=10, r=40))
    fig.add_vline(x=0, line_color="black", line_width=1)
    c_abs.plotly_chart(fig, width="stretch")

    mc_floor = 0.005
    rel = snap.dropna(subset=["rel_pct"])
    rel = rel[rel["model_pct"] >= mc_floor]
    n_noise = len(snap) - len(rel)
    if n_noise:
        c_rel.caption(f"{n_noise} longshots hidden (model < {mc_floor}% = under ~50 of 1M MC hits): "
                      "their relative edge is sampling noise, not signal — read them on the absolute chart")
    rel = rel.reindex(rel["rel_pct"].abs().sort_values(ascending=False).index).head(top_n)
    rel = rel.sort_values("rel_pct")
    fig = go.Figure(go.Bar(
        x=rel["rel_pct"], y=rel["team"], orientation="h",
        marker_color=[DOWN_C if v > 0 else UP_C for v in rel["rel_pct"]],
        text=[f"{v:+.0f}%" for v in rel["rel_pct"]], textposition="outside",
        cliponaxis=False))
    fig.update_layout(template=TPL, height=26 * len(rel) + 120,
                      title=f"Relative edge — ({market_src} − model) / model (%)<br>"
                            "<sup>red: market prices the team RICHER than model (longshot premium)</sup>",
                      xaxis_title=f"({market_src} − model) / model (%)", margin=dict(l=10, r=50))
    fig.add_vline(x=0, line_color="black", line_width=1)
    c_rel.plotly_chart(fig, width="stretch")


with tab_scatter:
    st.caption("One dot per team: x = what the market says, y = what the model says. "
               "On the dashed diagonal the two agree; vertical distance from it is the edge "
               "(dot colour). Log scale spreads out the longshots in the bottom-left.")
    log_axes = st.toggle("Log scale (see the longshot tail)", value=True)
    s = snap[(snap["model_pct"] > 0) & (snap["mkt_pct"] > 0)]
    n_zero = len(snap) - len(s)
    if n_zero:
        st.caption(f"{n_zero} teams hidden: model gives them 0% (log axes cannot show 0)")
    lim_lo = min(s["model_pct"].min(), s["mkt_pct"].min()) * 0.7
    lim_hi = max(s["model_pct"].max(), s["mkt_pct"].max()) * 1.3
    fig = go.Figure()
    fig.add_shape(type="line", x0=lim_lo, y0=lim_lo, x1=lim_hi, y1=lim_hi,
                  line=dict(color="gray", width=1.5, dash="dash"))
    labels = [t if (r["model_pct"] > 2 or r["mkt_pct"] > 2) else ""
              for t, (_, r) in zip(s["team"], s.iterrows())]
    ft = np.log10 if log_axes else np.asarray
    span = float(ft(lim_hi) - ft(lim_lo))
    xn = (ft(s["mkt_pct"].to_numpy(dtype=float)) - ft(lim_lo)) / span
    yn = (ft(s["model_pct"].to_numpy(dtype=float)) - ft(lim_lo)) / span
    fig.add_scatter(
        x=s["mkt_pct"], y=s["model_pct"], mode="markers+text",
        text=labels,
        textposition=spread_labels(xn, yn, [bool(t) for t in labels]),
        textfont_size=10,
        marker=dict(size=9, color=s["abs_pp"], colorscale="RdYlGn", cmid=0,
                    colorbar=dict(title="edge pp")),
        customdata=s["team"], name="teams",
        hovertemplate="%{customdata}<br>" + market_src + " %{x:.2f}% · model %{y:.2f}%<extra></extra>")
    ax = dict(type="log" if log_axes else "linear", range=None)
    if log_axes:
        ax["range"] = [np.log10(lim_lo), np.log10(lim_hi)]
    fig.update_layout(template=TPL, height=620, showlegend=False,
                      title="Model vs market — points above the dashed line = model higher than market<br>"
                            "<sup>distance from the diagonal IS the edge; the tail shows the favorite-longshot pattern</sup>",
                      xaxis={**ax, "title": f"{market_src} (%)"},
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
        fig.add_scatter(x=subt["date"], y=subt["mkt_pct"], name=f"{t} {market_src}",
                        mode="markers", marker=dict(color=c, symbol="x", size=10),
                        showlegend=False)
    x0 = pd.Timestamp(min(dates)) - pd.Timedelta(hours=12)
    x1 = pd.Timestamp(max(dates)) + pd.Timedelta(hours=12)
    fig.update_layout(template=TPL, height=520, hovermode="x unified",
                      title="Champion probability — model (line) vs market (✕)",
                      yaxis_title="champion prob (%)",
                      xaxis=dict(range=[x0, x1], tickformat="%b %d", dtick=86_400_000))
    st.plotly_chart(fig, width="stretch")


with tab_gw:
    st.caption("Who wins each group — model (solid line) vs Polymarket (✕), devigged "
               "within the group so both sum to 100%. This market is only live during the "
               "group stage; after it resolves the series is a closed calibration record. "
               "Kalshi lists no group-winner market and FairLine is snapshot-only, so "
               "Polymarket is the only market line here.")
    gw = load_gw()
    if gw.empty:
        st.info("No group-winner log yet. Generate it with "
                "`python -m mc_simu.backfill_group_winner` (writes "
                "data/mc_simu/group_winner_log.csv).")
    else:
        groups = sorted(gw["group"].unique())
        g = st.selectbox("Group", groups, format_func=lambda x: f"Group {x}")
        sub = gw[gw["group"] == g]
        carried = sub.dropna(subset=["model_pct"])
        carried = carried[carried["model_state_matches"] ==
                          carried["model_state_matches"].max()]
        fig = go.Figure()
        palette = px.colors.qualitative.Plotly
        for i, t in enumerate(sorted(sub["team"].unique())):
            subt = sub[sub["team"] == t].sort_values("date")
            c = palette[i % len(palette)]
            fig.add_scatter(x=subt["date"], y=subt["model_pct"], name=t,
                            mode="lines+markers", line=dict(color=c, width=2),
                            connectgaps=False)
            fig.add_scatter(x=subt["date"], y=subt["pm_devig_pct"], name=f"{t} Poly",
                            mode="markers", marker=dict(color=c, symbol="x", size=10),
                            showlegend=False)
        x0 = pd.Timestamp(gw["date"].min()) - pd.Timedelta(hours=12)
        x1 = pd.Timestamp(gw["date"].max()) + pd.Timedelta(hours=12)
        fig.update_layout(template=TPL, height=520, hovermode="x unified",
                          title=f"Group {g} winner — model (line) vs Polymarket (✕)",
                          yaxis_title="P(win group) (%)",
                          xaxis=dict(range=[x0, x1], tickformat="%b %d", dtick=86_400_000))
        st.plotly_chart(fig, width="stretch")
        # Carry-forward only when the latest state repeats across several trailing
        # days (git-history backfill); the football-data backfill gives each day its
        # own state, so suppress the warning there.
        per_day = sub.dropna(subset=["model_state_matches"]).groupby("date")[
            "model_state_matches"].max()
        stale = int(per_day.max()) if len(per_day) else 0
        carried = per_day[per_day == stale]
        if len(carried) > 1:
            st.caption(f"Model line is exact through the last distinct result state "
                       f"({stale} group matches) and **carried forward** from "
                       f"{pd.Timestamp(carried.index.min()).strftime('%b %d')} — that day's "
                       f"state repeats because no fresh results were available locally. "
                       f"Run `backfill_group_winner --from-api` (results token) to fill precisely.")
        else:
            st.caption("Model re-conditioned per day on real match dates "
                       "(football-data.org); ✕ = Polymarket the same day. A team the model "
                       "has eliminated (0%) while the market still prices it is a live gap.")


with tab_bracket:
    pred_csv = os.path.join(os.path.dirname(__file__), "..", "data", "mc_simu",
                            "wc2026_match_predictions.csv")
    if not os.path.exists(pred_csv):
        st.info("Per-match predictions land with the next daily run.")
    else:
        mp = pd.read_csv(pred_csv)
        has_mle = "p_home_mle" in mp.columns and mp["p_home_mle"].notna().any()
        if MODEL_COL[model_src] != "model_pct" and not has_mle:
            st.caption("Bracket shows the Production view — per-match mle columns "
                       "land with the next daily run.")
        elif model_src == "MLE strength":
            mp[["p_home", "p_draw", "p_away"]] = \
                mp[["p_home_mle", "p_draw_mle", "p_away_mle"]].to_numpy()
        elif model_src == "Pool 50/50":
            gh = np.sqrt(mp["p_home"].fillna(0) * mp["p_home_mle"].fillna(0))
            gd = np.sqrt(mp["p_draw"].fillna(0) * mp["p_draw_mle"].fillna(0))
            ga = np.sqrt(mp["p_away"].fillna(0) * mp["p_away_mle"].fillna(0))
            tot = gh + gd + ga
            mp["p_home"] = gh / tot
            mp["p_draw"] = np.where(mp["p_draw"].notna(), gd / tot, np.nan)
            mp["p_away"] = ga / tot
        with st.expander("How to read this tab"):
            st.markdown(
                "- **Knockout tree** — FIFA's fixed bracket; it fills in automatically each "
                "morning as results land. An undecided box names who feeds into it — "
                "*Group E winner*, *Group A runner-up*, *best 3rd place*, *M73 winner* — and "
                "shows the real teams once they're known.\n"
                "- **Advance %** — the model's chance to win that tie by any route (normal "
                "time, extra time or penalties). A green name with ✓ actually went through. "
                "Under each decided pairing: the model's most likely **90-minute score** "
                "(hover shows its probability) — extra time and penalties are excluded, so "
                "a 1-1 there means the tie most likely goes past 90 minutes.\n"
                "- **Model lean** — on each group match, the most likely of win / draw / win, "
                "with its probability; the bar shows the full split (🟩 left team · ⬜ draw · "
                "🟥 right team). Locked pre-tournament, never changes.\n"
                "- **verdict** — after a match: *✓ right* (the leaned result happened), "
                "*✗ held to a draw* (model leaned a winner but it ended level), or *✗ upset* "
                "(the other side won). The model talks in probabilities, not tips: about a "
                "quarter of its 75% leans *should* miss. A run of misses is the signal, not one. "
                "Note the lean (most likely *result*) and the likely *score* can disagree — a "
                "team can be favoured to win yet 1-1 still be the most common single score.\n"
                "- **likely 2-0 / nailed 2-0 🎯** — the single most likely of the 81 scorelines "
                "the model prices. Even the top pick rarely beats ~13%, so most miss by "
                "design; 🎯 marks an exact hit.")

        score_col = ("score_pred_mle"
                     if model_src == "MLE strength" and "score_pred_mle" in mp.columns
                     and mp["score_pred_mle"].notna().any()
                     else "score_pred")
        st.subheader("Knockout bracket")
        st.caption("Each box: the two sides with the model's % chance to advance and its most "
                   "likely 90-minute score; ✓ marks who went through. Hover for full team "
                   "names and the score's probability. The 3rd-place play-off is not "
                   "simulated (v1 scope).")
        st.plotly_chart(bracket_figure(mp[mp["stage"] != "group"], score_col),
                        width="stretch")

        st.subheader("Group stage")
        st.caption("Each match shows the model's lean — the most likely of win / draw / win "
                   "(the bar: 🟩 left team · ⬜ draw · 🟥 right team) and its single most likely "
                   "score. Once played: **bold score**, then the verdict — ✓ right, "
                   "✗ held to a draw (favourite only drew), or ✗ upset (the other side won) "
                   "— and 🎯 if it nailed the exact score.")
        groups = mp[mp["stage"] == "group"]
        grid = st.columns(3)
        for i, g in enumerate(sorted(groups["group"].unique())):
            col = grid[i % 3]
            col.markdown(f"**Group {g}**")
            sub = groups[groups["group"] == g]
            pts, gd = {}, {}
            for _, m in sub.iterrows():
                h, a = m["home_team"], m["away_team"]
                bar = prob_bar(m["p_home"], m["p_draw"], m["p_away"])
                lean, leanp = max([(h, m["p_home"]), ("Draw", m["p_draw"]), (a, m["p_away"])],
                                  key=lambda x: x[1])
                lean_txt = f"leans <b>{lean}</b> {leanp * 100:.0f}%"
                sp = m.get(score_col) if score_col in sub.columns else None
                sp = sp if isinstance(sp, str) and sp else None
                score_txt = f", likely {sp}" if sp else ""
                if pd.notna(m["home_goals"]) and str(m["home_goals"]) != "":
                    hg, ag = int(m["home_goals"]), int(m["away_goals"])
                    actual = h if hg > ag else (a if ag > hg else "Draw")
                    if lean == actual:
                        word, colour = "✓ right", "#2a8a2a"
                    elif actual == "Draw":
                        word, colour = "✗ held to a draw", "#c98a00"   # leaned a winner, drew
                    elif lean == "Draw":
                        word, colour = "✗ a winner emerged", "#c98a00"  # leaned draw, decided
                    else:
                        word, colour = "✗ upset", "#cc3b2f"            # the other side won
                    verdict = f"<span style='color:{colour}'>{word}</span>"
                    if sp:
                        score_played = (f" · nailed {sp} 🎯" if sp == f"{hg}-{ag}"
                                        else f" · predicted {sp}")
                    else:
                        score_played = ""
                    col.markdown(f"<small>{h} <b>{hg}–{ag}</b> {a} — model {lean_txt} "
                                 f"{verdict}{score_played}</small>{bar}",
                                 unsafe_allow_html=True)
                    pts[h] = pts.get(h, 0) + (3 if hg > ag else (1 if hg == ag else 0))
                    pts[a] = pts.get(a, 0) + (3 if ag > hg else (1 if hg == ag else 0))
                    gd[h] = gd.get(h, 0) + hg - ag
                    gd[a] = gd.get(a, 0) + ag - hg
                else:
                    col.markdown(f"<small>{h} – {a} — model {lean_txt}{score_txt}</small>{bar}",
                                 unsafe_allow_html=True)
            if pts:
                table = sorted(pts, key=lambda t: (-pts[t], -gd[t]))
                col.caption("Table: " + " · ".join(f"{t} {pts[t]}pt" for t in table))
            col.markdown("")


with tab_score:
    st.caption("How the model's per-match calls have actually done, scored on played "
               "group matches only. The model targets the market distribution, not "
               "match tips — read this as a sanity check, not the headline metric.")
    pred_csv = os.path.join(os.path.dirname(__file__), "..", "data", "mc_simu",
                            "wc2026_match_predictions.csv")
    if not os.path.exists(pred_csv):
        st.info("Scorecard fills in once matches are played.")
    else:
        mp = pd.read_csv(pred_csv)
        has_mle = "p_home_mle" in mp.columns and mp["p_home_mle"].notna().any()
        scored_src = model_src if (model_src == "Production (ELO+MV+star)" or has_mle) \
            else "Production (ELO+MV+star)"
        played_g = mp[(mp["stage"] == "group") & mp["home_goals"].notna()
                      & (mp["home_goals"].astype(str) != "")].copy()
        if played_g.empty:
            st.info("No group matches scored yet — come back after the first matchday.")
        else:
            played_g["hg"] = played_g["home_goals"].astype(int)
            played_g["ag"] = played_g["away_goals"].astype(int)
            played_g["actual"] = np.select(
                [played_g["hg"] > played_g["ag"], played_g["hg"] < played_g["ag"]],
                ["h", "a"], default="d")
            prod = played_g[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
            if scored_src == "MLE strength":
                P = played_g[["p_home_mle", "p_draw_mle", "p_away_mle"]].to_numpy(dtype=float)
            elif scored_src == "Pool 50/50":
                mle = played_g[["p_home_mle", "p_draw_mle", "p_away_mle"]].to_numpy(dtype=float)
                g = np.sqrt(prod * mle)
                P = g / g.sum(axis=1, keepdims=True)
            else:
                P = prod
            pick_idx = P.argmax(axis=1)
            pick = np.array(["h", "d", "a"])[pick_idx]
            onehot = np.array([{"h": [1, 0, 0], "d": [0, 1, 0], "a": [0, 0, 1]}[x]
                               for x in played_g["actual"]])
            brier = float(((P - onehot) ** 2).sum(axis=1).mean())
            uniform = np.full_like(P, 1 / 3)
            brier_base = float(((uniform - onehot) ** 2).sum(axis=1).mean())
            n = len(played_g)
            hit = float((pick == played_g["actual"].to_numpy()).mean())
            score_c = ("score_pred_mle" if scored_src == "MLE strength"
                       and "score_pred_mle" in played_g.columns else "score_pred")
            exact = int((played_g[score_c].astype(str)
                         == played_g["hg"].astype(str) + "-" + played_g["ag"].astype(str)).sum()
                        ) if score_c in played_g.columns else 0

            a, b, c, d_ = st.columns(4)
            a.metric("Matches scored", n)
            b.metric("Outcome hit rate", f"{hit * 100:.0f}%",
                     help="Share where the model's most likely W/D/L outcome matched the "
                          "result. A coin-three-ways baseline sits near 33-40%.")
            c.metric("Brier vs ignorance", f"{brier:.3f}",
                     delta=f"{brier - brier_base:+.3f} vs 1/3-each",
                     delta_color="inverse",
                     help="Multiclass Brier (0 best, 2 worst) of the model's W/D/L "
                          "probabilities. Lower than the uniform 1/3 baseline = the model "
                          "adds information. Green delta = beating ignorance.")
            d_.metric("Exact scoreline 🎯", f"{exact}/{n}",
                      help="Times the single most likely scoreline was exactly right. "
                           "Low by nature — best-pick scorelines rarely clear ~13%.")
            st.caption(f"Scored on {scored_src}. Brier baseline = uniform 1/3-1/3-1/3 "
                       f"({brier_base:.3f}); n={n} is small early on, so read trends, "
                       "not single-day swings.")


with tab_stab:
    st.caption("Is the model's bias vs the market stable? Left: total distance per day "
               "(JSD + L1) — flat is good. Right: per-team edge heatmap — a row keeping its "
               "colour is an understood bias; a row flipping colour is the anomaly to investigate.")
    c_l1, c_hm = st.columns([1, 2])
    daily = df.groupby("date").apply(
        lambda g: pd.Series({
            "L1 (pp)": (g["model_pct"] - g["mkt_pct"]).abs().sum(),
            "JSD": jsd_pct(g["model_pct"].to_numpy(), g["mkt_pct"].to_numpy()),
        }), include_groups=False).reset_index()
    fig = px.line(daily, x="date", y="L1 (pp)", markers=True, template=TPL, height=300,
                  title="Total model–market distance per day")
    c_l1.plotly_chart(fig, width="stretch")
    fig = px.line(daily, x="date", y="JSD", markers=True, template=TPL, height=300,
                  title="JSD per day")
    c_l1.plotly_chart(fig, width="stretch")

    top_teams = (df.groupby("team")["mkt_pct"].max()
                 .sort_values(ascending=False).head(top_n).index)
    piv = (df[df["team"].isin(top_teams)]
           .pivot_table(index="team", columns="date", values="abs_pp")
           .reindex(top_teams))
    # Eliminated teams drop out of the market (no pm_pct) so their later cells are
    # NaN and render as black bands. Their edge is genuinely ~0 once out (model 0%
    # vs market 0%), so fill with 0 -> neutral colour instead of a broken-looking gap.
    piv = piv.fillna(0)
    piv.columns = [pd.Timestamp(c).strftime("%m-%d") for c in piv.columns]
    fig = px.imshow(piv, color_continuous_scale="RdYlGn", zmin=-3, zmax=3, aspect="auto",
                    template=TPL, height=26 * len(piv) + 140,
                    title="Per-team edge through time (pp) — a STABLE row colour = understood bias;<br>"
                          "<sup>a row that flips colour is the anomaly worth investigating</sup>")
    # Force a categorical x-axis: the "MM-DD" labels otherwise get auto-coerced to a
    # date axis and rendered as the wrong months (e.g. 06-11 -> Nov), not the real June dates.
    fig.update_xaxes(type="category")
    fig.update_layout(coloraxis_colorbar_title="pp")
    c_hm.plotly_chart(fig, width="stretch")


with tab_data:
    st.caption("Raw snapshot for the selected day. pm/kalshi = platform mid prices; "
               "consensus = their normalized median; blank cells = platform doesn't quote "
               "that team. abs_pp / rel_pct follow the Market reference picked in the "
               "sidebar. The CSV download also carries per-platform bid/ask "
               "(executable spread) for backtesting.")
    show = snap[["team", "model_pct", "pm_pct", "kalshi_pct", "consensus_pct", "abs_pp", "rel_pct"]]
    st.dataframe(
        show.style.background_gradient(subset=["abs_pp"], cmap="RdYlGn", vmin=-3, vmax=3)
            .format({"model_pct": "{:.2f}", "pm_pct": "{:.2f}", "kalshi_pct": "{:.2f}",
                     "consensus_pct": "{:.2f}", "abs_pp": "{:+.2f}", "rel_pct": "{:+.0f}%"},
                    na_rep="—"),
        width="stretch", hide_index=True, height=620)
    st.download_button("Download full log (CSV)", df.to_csv(index=False),
                       file_name="daily_log.csv", mime="text/csv")
