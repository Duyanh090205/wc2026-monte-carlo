"""Weighted-MLE Poisson strength source (Ley, Van de Wiele & Van Eetvelde 2019).

Third rating source alongside self-Elo (`elo.py`) and eloratings. One strength
scalar per team fit by weighted maximum likelihood over the full match history:

    lambda_home = exp(c + r_i + h*a - r_j),   lambda_away = exp(c + r_j - r_i - h*a)

with match weight = time-decay (half-period H) x importance. This module is the
rating SOURCE only — the score matrix stays the production Poisson grid
(`single_game.goal_distribution`), so any divergence from production is
attributable to ratings, not score shape.

Phase 1 (this file): data loading + importance weights + match weights.
Phase 2 adds the fitter; Phase 3 the predict_match wrapper.

Naming: ratings here are called "strength" / "mle_rating" — NOT Elo (no
sequential updates, no K-factor). Never tuned against market prices; market is
post-hoc diagnostics only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from mc_simu._common import infer_tournament_type  # noqa: E402

DEFAULT_RESULTS_CSV = PROJECT_ROOT / "data" / "mc_simu" / "results.csv"

WINDOW_START = pd.Timestamp("1995-01-01")

# Half-period: matches H days before as_of weigh 0.5. Ley et al. (2019) Table 2
# optimum for national teams = 3 years.
HALF_PERIOD_DAYS = 1095.0

# Ley et al. (2019) §2.1.2 FIFA-style importance, mapped onto the repo's six
# tournament buckets (infer_tournament_type). nations_league=2.5 mirrors the
# K=40 qualifier parity already adopted in elo.K_FACTORS. other_tournament=1.0:
# sub-confed cups / invitationals / Olympics sit outside Ley's competitive tiers.
IMPORTANCE_WEIGHTS: dict[str, float] = {
    "world_cup_final":   4.0,
    "continental_final": 3.0,
    "qualifier":         2.5,
    "nations_league":    2.5,
    "other_tournament":  1.0,
    "friendly":          1.0,
}

MODELING_COLUMNS = ["date", "home_team", "away_team",
                    "home_score", "away_score", "neutral"]

# Adapter-canonical -> results.csv naming, same two divergences the self-Elo
# source already aliases (run_phase3_baselines.NAME_ALIASES_BY_SOURCE["self"]).
# Fitting keeps csv-native names; resolve at the lookup boundary only.
TEAM_NAME_ALIASES: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Curacao": "Curaçao",
}


def resolve_team_name(adapter_name: str) -> str:
    return TEAM_NAME_ALIASES.get(adapter_name, adapter_name)


def load_matches(
    csv_path: Path | str | None = None,
    *,
    window_start: pd.Timestamp | None = None,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Load played matches from results.csv for strength fitting.

    Filtering (in order):
        - drop rows with NaN scores (unplayed fixtures, e.g. future WC2026 rows)
        - drop exact entry-duplicates: same (date, teams, scores) — upstream
          double-entry (e.g. Gibraltar-Cayman 2026-06-06 listed under two city
          spellings). Same-day double-headers with differing scores are kept.
        - strict less-than as_of cutoff (same no-lookahead convention as
          elo.get_rating_as_of); default = today
        - date >= window_start (default 1995-01-01)

    Returns chronologically sorted DataFrame with added columns:
        neutral_bool (bool), tournament_type (str), importance (float)
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_RESULTS_CSV
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["home_score"].notna() & df["away_score"].notna()]
    df = df.drop_duplicates(
        subset=["date", "home_team", "away_team", "home_score", "away_score"],
        keep="first",
    )

    cutoff = pd.Timestamp.now() if as_of is None else pd.Timestamp(as_of)
    df = df[df["date"] < cutoff]
    start = WINDOW_START if window_start is None else pd.Timestamp(window_start)
    df = df[df["date"] >= start]

    df = df.copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    if df["neutral"].dtype == object:
        df["neutral_bool"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    else:
        df["neutral_bool"] = df["neutral"].astype(bool)

    df["tournament_type"] = df["tournament"].map(infer_tournament_type)
    df["importance"] = df["tournament_type"].map(IMPORTANCE_WEIGHTS)

    return df.sort_values("date", kind="stable").reset_index(drop=True)


def time_decay_weights(
    dates: pd.Series,
    as_of: pd.Timestamp | str,
    half_period_days: float = HALF_PERIOD_DAYS,
) -> np.ndarray:
    days_before = (pd.Timestamp(as_of) - pd.to_datetime(dates)).dt.days.to_numpy()
    return 0.5 ** (days_before / half_period_days)


def match_weights(
    df: pd.DataFrame,
    as_of: pd.Timestamp | str,
    half_period_days: float = HALF_PERIOD_DAYS,
) -> np.ndarray:
    return time_decay_weights(df["date"], as_of, half_period_days) \
        * df["importance"].to_numpy()


@dataclass(frozen=True)
class FitResult:
    strengths: dict[str, float]
    c: float
    h: float
    loglik: float
    n_matches: int
    n_teams: int
    as_of: str
    half_period_days: float
    converged: bool
    optimizer_message: str


def _nll_and_grad(
    x: np.ndarray,
    idx_home: np.ndarray,
    idx_away: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    home_flag: np.ndarray,
    w: np.ndarray,
    n_teams: int,
) -> tuple[float, np.ndarray]:
    r = np.empty(n_teams)
    r[:-1] = x[:n_teams - 1]
    r[-1] = -r[:-1].sum()
    c, hfa = x[-2], x[-1]

    eta_home = c + r[idx_home] + hfa * home_flag - r[idx_away]
    eta_away = c + r[idx_away] - r[idx_home] - hfa * home_flag
    lam_home = np.exp(eta_home)
    lam_away = np.exp(eta_away)

    nll = -float((w * (goals_home * eta_home - lam_home
                       + goals_away * eta_away - lam_away)).sum())

    res_home = w * (goals_home - lam_home)
    res_away = w * (goals_away - lam_away)
    diff = res_home - res_away
    grad_r_full = (np.bincount(idx_home, weights=diff, minlength=n_teams)
                   - np.bincount(idx_away, weights=diff, minlength=n_teams))
    # chain rule through r_T = -sum(r_free)
    grad_r_free = grad_r_full[:-1] - grad_r_full[-1]
    grad_c = float((res_home + res_away).sum())
    grad_h = float((home_flag * diff).sum())

    return nll, -np.concatenate([grad_r_free, [grad_c, grad_h]])


def fit_strengths(
    matches: pd.DataFrame,
    as_of: pd.Timestamp | str,
    *,
    half_period_days: float = HALF_PERIOD_DAYS,
    warm_start: FitResult | None = None,
) -> FitResult:
    """Weighted Poisson MLE: one strength per team + intercept c + home effect h.

    Identification via reparameterization: T-1 free strengths, r_T = -sum(rest).
    L-BFGS-B with analytic gradient. Deterministic for fixed inputs.
    """
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    index = {t: k for k, t in enumerate(teams)}
    n_teams = len(teams)

    idx_home = matches["home_team"].map(index).to_numpy(dtype=np.int64)
    idx_away = matches["away_team"].map(index).to_numpy(dtype=np.int64)
    goals_home = matches["home_score"].to_numpy(dtype=np.float64)
    goals_away = matches["away_score"].to_numpy(dtype=np.float64)
    home_flag = (~matches["neutral_bool"]).to_numpy(dtype=np.float64)
    w = match_weights(matches, as_of, half_period_days)

    x0 = np.zeros(n_teams + 1)
    if warm_start is not None:
        prev = warm_start.strengths
        x0[:n_teams - 1] = [prev.get(t, 0.0) for t in teams[:-1]]
        x0[-2], x0[-1] = warm_start.c, warm_start.h

    res = minimize(
        _nll_and_grad, x0, jac=True, method="L-BFGS-B",
        args=(idx_home, idx_away, goals_home, goals_away, home_flag, w, n_teams),
        options={"maxiter": 5000, "maxfun": 10000},
    )

    r = np.empty(n_teams)
    r[:-1] = res.x[:n_teams - 1]
    r[-1] = -r[:-1].sum()

    return FitResult(
        strengths={t: float(r[index[t]]) for t in teams},
        c=float(res.x[-2]),
        h=float(res.x[-1]),
        loglik=-float(res.fun),
        n_matches=len(matches),
        n_teams=n_teams,
        as_of=str(pd.Timestamp(as_of).date()),
        half_period_days=half_period_days,
        converged=bool(res.success),
        optimizer_message=str(res.message),
    )


def write_strengths_artifact(
    fit: FitResult,
    path: Path | str,
    *,
    data_csv_path: Path | str = DEFAULT_RESULTS_CSV,
    window_start: pd.Timestamp = WINDOW_START,
) -> Path:
    payload = {
        "model": "mle_strength",
        "spec": "Ley-VandeWiele-VanEetvelde-2019 independent Poisson, weighted MLE",
        "as_of": fit.as_of,
        "window_start": str(pd.Timestamp(window_start).date()),
        "half_period_days": fit.half_period_days,
        "importance_weights": IMPORTANCE_WEIGHTS,
        "c": fit.c,
        "h": fit.h,
        "loglik": fit.loglik,
        "n_matches": fit.n_matches,
        "n_teams": fit.n_teams,
        "converged": fit.converged,
        "optimizer_message": fit.optimizer_message,
        "data_sha256": hashlib.sha256(Path(data_csv_path).read_bytes()).hexdigest(),
        "strengths": {t: fit.strengths[t] for t in sorted(fit.strengths)},
    }
    out = Path(path)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    return out


def load_strengths_artifact(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_mle_predictor(
    artifact: dict | Path | str,
    *,
    diagonal_inflation: float = 0.20,
    max_goals: int = 8,
):
    """Build a predict_match-shaped callable from a frozen strengths artifact.

    Elo args are ignored (plug-in contract compliance only); teams resolve via
    ctx.home_country / ctx.away_country. Score matrix mirrors production exactly
    (Poisson grid + same diagonal inflation) so any divergence from the
    production model is attributable to ratings, not score shape.

    Unknown team raises KeyError loudly — no silent fallback.
    """
    from mc_simu.single_game import MatchContext, MatchPrediction, goal_distribution

    art = artifact if isinstance(artifact, dict) else load_strengths_artifact(artifact)
    strengths: dict[str, float] = art["strengths"]
    c, h = float(art["c"]), float(art["h"])

    def predict_match_mle(elo_home: float, elo_away: float,
                          ctx: MatchContext) -> MatchPrediction:
        del elo_home, elo_away
        try:
            r_home = strengths[resolve_team_name(ctx.home_country)]
            r_away = strengths[resolve_team_name(ctx.away_country)]
        except KeyError as exc:
            raise KeyError(f"mle_strength: team {exc.args[0]!r} not in artifact "
                           f"({len(strengths)} teams, as_of {art.get('as_of')})") from None
        a = 0.0 if ctx.is_neutral else 1.0
        lam_home = float(np.exp(c + r_home + h * a - r_away))
        lam_away = float(np.exp(c + r_away - r_home - h * a))
        grid = goal_distribution(lam_home, lam_away, diagonal_inflation, max_goals)
        return MatchPrediction(
            p_home=float(np.sum(np.tril(grid, k=-1))),
            p_draw=float(np.sum(np.diag(grid))),
            p_away=float(np.sum(np.triu(grid, k=1))),
            goal_grid=grid,
        )

    return predict_match_mle


__all__ = [
    "DEFAULT_RESULTS_CSV",
    "WINDOW_START",
    "HALF_PERIOD_DAYS",
    "IMPORTANCE_WEIGHTS",
    "MODELING_COLUMNS",
    "TEAM_NAME_ALIASES",
    "resolve_team_name",
    "load_matches",
    "time_decay_weights",
    "match_weights",
    "FitResult",
    "fit_strengths",
    "write_strengths_artifact",
    "load_strengths_artifact",
    "make_mle_predictor",
]
