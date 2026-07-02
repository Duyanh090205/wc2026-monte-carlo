-- Supabase SQL editor (Project > SQL > New query). Fully idempotent —
-- safe to paste and run the WHOLE file on both fresh and existing projects.
create table if not exists daily_log (
    date          date        not null,
    team          text        not null,
    model_pct     double precision not null,
    pm_pct        double precision,
    kalshi_pct    double precision,
    consensus_pct double precision not null,
    abs_pp        double precision not null,
    rel_pct       double precision,
    inserted_at   timestamptz not null default now(),
    primary key (date, team)
);

-- Migration 2026-06-11: parallel mle_strength source + 50/50 pool.
-- Idempotent — safe to run on an existing table.
alter table daily_log add column if not exists mle_pct  double precision;
alter table daily_log add column if not exists pool_pct double precision;

-- Migration 2026-06-13: executable spread per platform (bid/ask), for backtest.
alter table daily_log add column if not exists pm_bid     double precision;
alter table daily_log add column if not exists pm_ask     double precision;
alter table daily_log add column if not exists kalshi_bid double precision;
alter table daily_log add column if not exists kalshi_ask double precision;

alter table daily_log enable row level security;

-- Dashboard reads with the anon key; writes go through the service key (bypasses RLS).
drop policy if exists "anon read" on daily_log;
create policy "anon read" on daily_log for select to anon using (true);

-- Migration 2026-06-22: group-winner series (model vs Polymarket), one row per
-- (date, group, team). Separate table — different shape from daily_log (per-group,
-- group-stage only). "group" is a reserved word, hence quoted. pm_raw vs pm_devig
-- keep the overround visible (raw) alongside the comparable probability (devig).
create table if not exists group_winner_log (
    date                 date    not null,
    "group"              text    not null,
    team                 text    not null,
    model_pct            double precision,
    model_state_matches  smallint,
    pm_raw_pct           double precision,
    pm_devig_pct         double precision,
    inserted_at          timestamptz not null default now(),
    primary key (date, "group", team)
);

alter table group_winner_log enable row level security;
drop policy if exists "anon read" on group_winner_log;
create policy "anon read" on group_winner_log for select to anon using (true);

-- Migration 2026-07-02: knockout reach series (model vs Polymarket), one row per
-- (date, round, team). round in ('r16','qf','sf','final'); reach = wins the
-- feeder tie, matching Polymarket "Nation To Reach <round>" events. pm_devig
-- scales the round's quotes to its slot count (16/8/4/2).
create table if not exists reach_log (
    date                 date    not null,
    round                text    not null,
    team                 text    not null,
    model_pct            double precision,
    model_state_matches  smallint,
    pm_raw_pct           double precision,
    pm_devig_pct         double precision,
    inserted_at          timestamptz not null default now(),
    primary key (date, round, team)
);

alter table reach_log enable row level security;
drop policy if exists "anon read" on reach_log;
create policy "anon read" on reach_log for select to anon using (true);
