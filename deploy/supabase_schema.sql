-- Run once in Supabase SQL editor (Project > SQL > New query).
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

alter table daily_log enable row level security;

-- Dashboard reads with the anon key; writes go through the service key (bypasses RLS).
create policy "anon read" on daily_log for select to anon using (true);
