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

alter table daily_log enable row level security;

-- Dashboard reads with the anon key; writes go through the service key (bypasses RLS).
drop policy if exists "anon read" on daily_log;
create policy "anon read" on daily_log for select to anon using (true);
