"""MC Simulator (Stream 3) — standalone tournament outcome predictor.

v1 = research/validation tool per mc_simu.md spec §0.2. Outputs to CSV.
v2 will add DB integration when team approves productionization.
"""

from mc_simu._common import HARD_STOP_LOG, banner, hard_stop, infer_tournament_type
from mc_simu.confederations import get_confederation
from mc_simu.elo import (
    K_FACTORS,
    build_rating_history,
    elo_expected,
    elo_update,
    get_rating_as_of,
    mov_multiplier,
)
from mc_simu.single_game import (
    MatchContext,
    MatchPrediction,
    ModelParams,
    elo_to_lambda,
    goal_distribution,
    hfa_log_goals,
    predict_match,
)

__all__ = [
    # _common
    "HARD_STOP_LOG", "banner", "hard_stop", "infer_tournament_type",
    # confederations
    "get_confederation",
    # elo
    "K_FACTORS", "elo_expected", "mov_multiplier", "elo_update",
    "build_rating_history", "get_rating_as_of",
    # single_game
    "MatchContext", "ModelParams", "MatchPrediction",
    "elo_to_lambda", "goal_distribution", "hfa_log_goals", "predict_match",
]
