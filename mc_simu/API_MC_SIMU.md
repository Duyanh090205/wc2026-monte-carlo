# MC Tracker Public Data API

**Base URL:** `https://sprizfzojukdmitawcoq.supabase.co/rest/v1`

Read-only access to the daily model-vs-market tracking log (Supabase PostgREST).
No registration needed â€” pass the public read key in **both** headers:

```
apikey: sb_publishable_4gNm_63ldh7Z_S2suefz9g_8PR1hObm
Authorization: Bearer sb_publishable_4gNm_63ldh7Z_S2suefz9g_8PR1hObm
```

The key is safe to share: row-level security only allows `SELECT` on the
`daily_log` table. Writes go through a separate private key used by CI.

---

## Table: `daily_log`

One row per **(date, team)** â€” a daily snapshot of the MC structural prior vs
the live prediction-market prices for WC2026 outrights. New snapshot lands at
**08:30 UTC (04:30 ET)** daily via GitHub Actions; reruns the same day upsert
in place (no duplicates).

| Column | Type | Meaning |
|---|---|---|
| `date` | date | Snapshot day (UTC) |
| `team` | text | Canonical team name (48 finalists) |
| `model_pct` | float | MC champion probability (%) â€” ELO + squad-MV + star model, static, re-conditioned daily on played results |
| `pm_pct` | float | Polymarket mid price (%), null if not quoted |
| `kalshi_pct` | float | Kalshi mid price (%), null if not quoted (~34 of 48 teams) |
| `consensus_pct` | float | Normalized median of quoted platforms (%) |
| `abs_pp` | float | `model_pct âˆ’ consensus_pct` (percentage points). Positive = model above market |
| `rel_pct` | float | `(consensus âˆ’ model) / model Ã— 100`. Large positive on longshots = favorite-longshot premium. Null when model gives 0% |
| `inserted_at` | timestamptz | Row write time |

---

## Examples

All examples are plain GET requests â€” paste into a browser won't work (headers
required); use curl / Python / JS.

### Today's full snapshot

```
GET /daily_log?date=eq.2026-06-11&order=model_pct.desc
```

### One team's trajectory through the tournament

```
GET /daily_log?team=eq.Spain&order=date.asc&select=date,model_pct,consensus_pct,abs_pp
```

### Biggest model-vs-market gaps today

```
GET /daily_log?date=eq.2026-06-11&order=abs_pp.desc&limit=10
```

### Several teams at once

```
GET /daily_log?team=in.(Spain,France,England)&order=date.asc
```

### Everything (full log download)

```
GET /daily_log?select=*&order=date.asc,team.asc
```

### curl

```bash
KEY=sb_publishable_4gNm_63ldh7Z_S2suefz9g_8PR1hObm
curl -s "https://sprizfzojukdmitawcoq.supabase.co/rest/v1/daily_log?date=eq.2026-06-11&order=model_pct.desc" \
  -H "apikey: $KEY" -H "Authorization: Bearer $KEY"
```

### Python

```python
import requests, pandas as pd

KEY = "sb_publishable_4gNm_63ldh7Z_S2suefz9g_8PR1hObm"
URL = "https://sprizfzojukdmitawcoq.supabase.co/rest/v1/daily_log"
r = requests.get(URL, params={"order": "date.asc,team.asc"},
                 headers={"apikey": KEY, "Authorization": f"Bearer {KEY}"})
df = pd.DataFrame(r.json())
```

### JavaScript

```js
const KEY = "sb_publishable_4gNm_63ldh7Z_S2suefz9g_8PR1hObm";
const rows = await fetch(
  "https://sprizfzojukdmitawcoq.supabase.co/rest/v1/daily_log?date=eq.2026-06-11",
  { headers: { apikey: KEY, Authorization: `Bearer ${KEY}` } }
).then(r => r.json());
```

---

## Query syntax (PostgREST)

Filters compose with `&`: `eq.` `neq.` `gt.` `gte.` `lt.` `lte.` `in.(a,b)`
`is.null` / `not.is.null`; `select=` picks columns; `order=col.asc|desc`;
`limit=` / `offset=` paginate. Row count: send header `Prefer: count=exact`
and read the `Content-Range` response header. Full reference:
[postgrest.org/en/stable/references/api/tables_views.html](https://docs.postgrest.org/en/stable/references/api/tables_views.html)

---

## Notes

- **Model is static** (locked pre-tournament, LOTO-CV validated): day-to-day
  movement in `model_pct` comes only from re-conditioning on played results,
  never from re-fitting. A stable bias vs the market is expected; the signal
  is a change in that bias.
- **Market sources:** prices come from the FairLine `/prices` endpoint
  (Polymarket WebSocket + Kalshi REST mids). This log stores one snapshot per
  day; for live intraday prices use the FairLine API directly
  (`https://seal-app-yatxw.ondigitalocean.app/api/docs`).
- **Coverage:** consensus/Polymarket quote all 48 finalists; Kalshi lists ~34
  (longshots missing). `model_pct` can be 0.000 for extreme longshots
  (sub-1-in-1,000,000 MC resolution).
- Free-tier Supabase â€” please keep polling to a few requests/minute.
- Dashboard with the same data: Streamlit app (`dashboard/mc_tracker.py`).
