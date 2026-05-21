"""Phase 3 deep-audit: integration-layer wiring bugs that unit tests don't catch.

Three cross-path tests:

1. sample_group_score (scalar, unit-tested) vs sample_scores_batch (vectorized,
   used in MC hot path) — should produce IDENTICAL output given identical RNG
   state. If not, unit tests pass while production sim is biased.

2. ko_advance symmetry — p((A,B)) + p((B,A)) should ≈ 1.0 for all directed
   pairs. If not, the model has internal inconsistency that biases asymmetric
   bracket positions.

3. enrich_group_venues flow — verify enriched venue actually reaches
   predict_match (host α HFA applied for host-team group matches).
"""
from __future__ import annotations

import os
# Deterministic BLAS
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from mc_simu.simulator import (
    GroupSampler,
    HostInfo,
    build_group_samplers,
    build_ko_advance_table,
    sample_group_score,
    sample_scores_batch,
)
from mc_simu.single_game import ModelParams
from mc_simu.tournaments.wc2026 import (
    WC2026_HOST_COUNTRIES,
    WC2026_HOST_CONFEDERATION,
    build_sim_context,
    enrich_group_venues,
    load_wc2026_bundle,
)


def _load_ratings() -> dict[str, float]:
    elo_df = pd.read_csv("data/mc_simu/elo_history.csv", parse_dates=["date"])
    elo_df = elo_df.sort_values("date")
    return elo_df.groupby("team").tail(1).set_index("team")["rating_after"].to_dict()


# ── Audit 1: scalar vs batch sampler equivalence ────────────────────────────

def audit_scalar_vs_batch_sampler() -> tuple[bool, str]:
    """Same seed → both functions should give same (h, a) per fixture.

    Method: build 5 hand-crafted CDFs (degenerate, uniform, skewed, low-prob-tail,
    home-blowout). Sample N=200 times each with each function on parallel RNGs
    initialized from same seed. Compare element-wise.
    """
    cdfs = []
    # CDF 1: uniform over 81 cells
    cdfs.append(np.cumsum(np.full(81, 1 / 81)))
    # CDF 2: degenerate at (0, 0)
    c = np.ones(81); cdfs.append(c)
    # CDF 3: 90% mass at (0,0), 10% spread
    flat = np.full(81, 0.1 / 80); flat[0] = 0.9
    cdfs.append(np.cumsum(flat))
    # CDF 4: skewed home win (lower triangle gets more mass)
    flat = np.full(81, 0)
    for i in range(9):
        for j in range(9):
            flat[i * 9 + j] = max(0, (i - j) + 1)
    flat = flat / flat.sum()
    cdfs.append(np.cumsum(flat))
    # CDF 5: simulated realistic — Poisson 1.5 vs 1.0 product with diagonal boost
    from scipy.stats import poisson
    pmf_h = poisson.pmf(np.arange(9), 1.5)
    pmf_a = poisson.pmf(np.arange(9), 1.0)
    grid = np.outer(pmf_h, pmf_a)
    np.fill_diagonal(grid, grid.diagonal() * 1.2)
    grid /= grid.sum()
    cdfs.append(np.cumsum(grid.flatten()))

    # Force last cell to 1.0 (matches GroupSampler invariant)
    for c in cdfs:
        c[-1] = 1.0

    N = 200
    issues = []

    for k, cdf in enumerate(cdfs):
        sampler = GroupSampler(home_team="A", away_team="B", cdf_flat=cdf)

        # Path A: N scalar calls with rng_a
        rng_a = np.random.default_rng(seed=12345 + k)
        scalar_out = np.array([sample_group_score(sampler, rng_a) for _ in range(N)])

        # Path B: N batched calls of size 1 with rng_b (matched seed)
        # Each batched call consumes 1 random per row; with stack of 1 row, equivalent to scalar.
        rng_b = np.random.default_rng(seed=12345 + k)
        cdfs_2d = np.stack([cdf])
        batch_out = np.array([sample_scores_batch(cdfs_2d, rng_b)[0] for _ in range(N)])

        if not np.array_equal(scalar_out, batch_out):
            n_diff = (scalar_out != batch_out).any(axis=1).sum()
            first_diff = np.where((scalar_out != batch_out).any(axis=1))[0][:5]
            issues.append(
                f"  CDF {k}: {n_diff}/{N} samples differ. "
                f"First diffs at i={first_diff.tolist()}\n"
                f"    scalar[{first_diff[0]}] = {scalar_out[first_diff[0]]}\n"
                f"    batch [{first_diff[0]}] = {batch_out[first_diff[0]]}"
            )

    if not issues:
        return True, f"PASS — scalar and batch samplers IDENTICAL across {len(cdfs)} CDFs × {N} samples"
    return False, "FAIL — scalar vs batch divergence:\n" + "\n".join(issues)


# ── Audit 2: ko_advance symmetry ────────────────────────────────────────────

def audit_ko_advance_symmetry() -> tuple[bool, str]:
    """For every directed pair (A, B), require ko_advance[(A,B)] + ko_advance[(B,A)] ≈ 1.0.

    This must hold because P(A beats B in KO) + P(B beats A in KO) = 1.0 by
    complementarity (someone advances). If asymmetric β breaks this invariant,
    the bracket has order-dependent biases.
    """
    ratings = _load_ratings()
    bundle = load_wc2026_bundle(ratings)
    sim = build_sim_context(bundle)
    table = sim.ko_advance

    violations: list[tuple[str, str, float, float, float]] = []
    teams = sim.teams
    for a in teams:
        for b in teams:
            if a == b:
                continue
            p_ab = table[(a, b)]
            p_ba = table[(b, a)]
            total = p_ab + p_ba
            if abs(total - 1.0) > 1e-9:
                violations.append((a, b, p_ab, p_ba, total))

    if not violations:
        n_pairs = len(teams) * (len(teams) - 1)
        return True, f"PASS — KO advance symmetry holds for all {n_pairs} directed pairs (max |p_ab+p_ba-1| < 1e-9)"

    # Show top 5 worst violations
    violations.sort(key=lambda v: -abs(v[4] - 1.0))
    sample = violations[:5]
    body = "\n".join(
        f"  ({a:24}, {b:24})  p_ab={p_ab:.6f}  p_ba={p_ba:.6f}  sum={total:.6f}  Δ={total-1:+.2e}"
        for a, b, p_ab, p_ba, total in sample
    )
    return False, (
        f"FAIL — {len(violations)} directed pairs violate symmetry. "
        f"Top 5 worst:\n{body}"
    )


# ── Audit 3: enriched venue flow ────────────────────────────────────────────

def audit_enriched_venue_flow() -> tuple[bool, str]:
    """Verify enrich_group_venues output reaches predict_match (host α actually applied).

    Method: build_sim_context with WC2026 bundle. For a host-team group match
    (e.g., Mexico vs anyone in Group A), compare the predicted goal_grid AGAINST
    a manually-built MatchContext using raw TBD venue. They MUST differ — if
    they're equal, enrichment is being dropped somewhere in the pipeline.
    """
    ratings = _load_ratings()
    bundle = load_wc2026_bundle(ratings)
    sim = build_sim_context(bundle)

    # Pick a Mexico match — Group A has Mexico, South Africa, South Korea, Czechia
    mexico_matches_in_A = [
        (i, h, a) for i, (h, a) in enumerate(sim.group_team_pairs["A"])
        if h == "Mexico" or a == "Mexico"
    ]
    if not mexico_matches_in_A:
        return False, "FAIL — no Mexico matches found in Group A pairs (test setup bug)"

    # Take the first such match
    i_match, h_team, a_team = mexico_matches_in_A[0]
    enriched_cdf = sim.group_cdfs_per_group["A"][i_match]

    # Now build the same fixture WITHOUT enrichment (raw TBD venue)
    from mc_simu.simulator import _build_group_match_context
    from mc_simu.single_game import predict_match
    raw_ctx = _build_group_match_context(
        home_team=h_team,
        away_team=a_team,
        venue_country="TBD",  # un-enriched
        tournament_type="world_cup_final",
        host=HostInfo(host_countries=list(WC2026_HOST_COUNTRIES), host_confederation=WC2026_HOST_CONFEDERATION),
    )
    raw_pred = predict_match(bundle.ratings[h_team], bundle.ratings[a_team], raw_ctx, bundle.params)
    raw_cdf = np.cumsum(raw_pred.goal_grid.flatten())
    raw_cdf[-1] = 1.0

    # If enrichment is reaching predict_match, enriched_cdf SHOULD differ from raw_cdf
    # (because α HFA is +0.27 log-goals when venue==Mexico)
    max_diff = float(np.abs(enriched_cdf - raw_cdf).max())

    if max_diff < 1e-6:
        return False, (
            f"FAIL — enriched CDF identical to raw TBD CDF for {h_team} vs {a_team}. "
            f"Max diff: {max_diff:.2e}. enrich_group_venues is being DROPPED somewhere."
        )

    # Sanity: Mexico's marginal P(home_goals) should be higher in enriched case
    enriched_grid = enriched_cdf.reshape(-1)  # 81 cells flat
    # Get expected home goals from each
    # Reconstruct grids from CDFs (approximate)
    enriched_flat = np.diff(np.concatenate([[0], enriched_cdf]))
    raw_flat = np.diff(np.concatenate([[0], raw_cdf]))
    enriched_grid_2d = enriched_flat.reshape(9, 9)
    raw_grid_2d = raw_flat.reshape(9, 9)
    mexico_idx = 0 if h_team == "Mexico" else 1  # 0=home (row sums), 1=away (col sums)
    if mexico_idx == 0:
        e_mexico_mean = (np.arange(9) * enriched_grid_2d.sum(axis=1)).sum()
        r_mexico_mean = (np.arange(9) * raw_grid_2d.sum(axis=1)).sum()
    else:
        e_mexico_mean = (np.arange(9) * enriched_grid_2d.sum(axis=0)).sum()
        r_mexico_mean = (np.arange(9) * raw_grid_2d.sum(axis=0)).sum()
    boost = e_mexico_mean - r_mexico_mean

    if boost <= 0:
        return False, (
            f"FAIL — enriched Mexico expected-goals ({e_mexico_mean:.3f}) NOT higher than raw "
            f"({r_mexico_mean:.3f}). Host α should add ~exp(0.27)=1.31× to Mexico's λ but it didn't."
        )

    return True, (
        f"PASS — {h_team} vs {a_team}: enriched ≠ raw (max diff {max_diff:.4f}). "
        f"Mexico expected-goals enriched={e_mexico_mean:.3f} vs raw={r_mexico_mean:.3f} (Δ=+{boost:.3f})"
    )


# ── Driver ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3 DEEP-AUDIT — integration-layer wiring bugs")
    print("=" * 70)

    audits = [
        ("Audit 1 — scalar vs batch sampler equivalence", audit_scalar_vs_batch_sampler),
        ("Audit 2 — KO advance symmetry (p(A,B)+p(B,A)=1)", audit_ko_advance_symmetry),
        ("Audit 3 — enriched venue flows into predict_match", audit_enriched_venue_flow),
    ]

    results = []
    for name, fn in audits:
        print(f"\n--- {name} ---")
        try:
            ok, msg = fn()
            print(msg)
            results.append((name, ok, msg))
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            results.append((name, False, f"ERROR: {e}"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    pass_count = sum(1 for _, ok, _ in results if ok)
    for name, ok, _ in results:
        marker = "[PASS]" if ok else "[FAIL]"
        print(f"  {marker} {name}")
    print(f"\n{pass_count}/{len(results)} audits passed")

    sys.exit(0 if pass_count == len(results) else 1)
