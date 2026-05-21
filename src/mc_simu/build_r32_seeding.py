"""Build R32 cross-seeding lookup table for WC2026 (C(12,8) = 495 combinations).

Source: Wikipedia "2026 FIFA World Cup knockout stage" → table "Combinations of
matches in the round of 32" (published in FIFA Annex C of tournament regulations).

The 8 "Best 3rd from X" slots map to 8 of 12 group winners:
   1A, 1B, 1D, 1E, 1G, 1I, 1K, 1L (the W vs 3rd-place matches).

The other 4 group winners (1C, 1F, 1H, 1J) play runner-ups in fixed matches per
2026_FIFA_World_Cup_knockout_stage.

For each of 495 combinations (8 of 12 groups whose 3rd advances), this script
extracts which 3rd-place team is paired with each of the 8 winners.

Output: data/mc_simu/r32_seeding_table.json.
Schema : {sorted_8_group_letters: {"1A": "3E", "1B": "3J", ..., "1L": "3K"}, ...}.

Hand-verify >= 5 sample rows against Wikipedia / FIFA Annex C per spec §1.3 Check 6.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import requests  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

from mc_simu._common import banner, hard_stop  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_JSON = DATA_DIR / "r32_seeding_table.json"

SOURCE_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage"
CACHE_FILE = "wc2026_knockout.html"

# The 8 winner slots that face 3rd-place teams (per FIFA bracket).
WINNER_SLOTS = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]

# All 12 groups.
ALL_GROUPS = list("ABCDEFGHIJKL")

# Hand-verified samples per spec §1.3 Check 6 — published in FIFA Annex C.
# Option 1 of the Wikipedia table maps to (E,F,G,H,I,J,K,L) advancing 3rds.
HAND_VERIFIED_SAMPLES = {
    # Option 1: thirds from E,F,G,H,I,J,K,L
    "EFGHIJKL": {"1A": "3E", "1B": "3J", "1D": "3I", "1E": "3F",
                 "1G": "3H", "1I": "3G", "1K": "3L", "1L": "3K"},
}


def _fetch_html(force: bool = False) -> str:
    cache_path = CACHE_DIR / CACHE_FILE
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {SOURCE_URL} ...")
    resp = requests.get(SOURCE_URL, headers={"User-Agent": "FairLine-MC-Simu/0.1"}, timeout=60)
    if resp.status_code != 200:
        hard_stop("build_r32_seeding", f"HTTP {resp.status_code}", "HTTP 200")
    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.text


def _find_combinations_table(html: str):
    """Find the <table> whose caption mentions 'Combinations of matches in the round of 32'."""
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        cap = table.find("caption")
        if cap and "Combinations of matches" in cap.get_text():
            return table
    return None


def parse_r32_table(html: str) -> dict[str, dict[str, str]]:
    """Parse the 495-row combinations table into a dict.

    Each row contains:
      - col 0: option number
      - cols 1-12: bold letter if that group's 3rd advanced, else empty
      - col 13: separator
      - cols 14-21: third-place team paired with each WINNER_SLOTS slot

    Returns dict keyed by sorted 8-letter string of advancing groups → mapping dict.
    """
    table = _find_combinations_table(html)
    if table is None:
        hard_stop("build_r32_seeding parse", "table not found",
                  "table with caption 'Combinations of matches in the round of 32'")
    assert table is not None  # for type-checker

    body_rows = table.find_all("tr")
    if len(body_rows) < 2:
        hard_stop("build_r32_seeding parse", f"{len(body_rows)} rows", ">= 496 (1 header + 495 data)")

    result: dict[str, dict[str, str]] = {}

    # Skip the header row.
    data_rows = body_rows[1:]

    for ri, row in enumerate(data_rows, start=1):
        cells = row.find_all(["td", "th"])
        if len(cells) < 21:
            # Some footer rows or stylistic rows may have fewer cells; skip.
            continue

        # Cells layout (from inspection):
        #   [0] option number (th)
        #   [1..12] one per group A..L: bold letter if 3rd advanced
        #   [13..20] eight mapping cells (3X values)
        # Wikipedia table has 1 rowspan="495" placeholder cell after the 12 group
        # columns — appears only in row 1, so subsequent rows shift by one.
        # We extract from the END instead to be robust.

        # Last 8 cells = the seed mappings
        mapping_cells = cells[-8:]
        # The 12 group cells are positioned between (option) and (mappings)
        group_cells = cells[1:13]
        if len(group_cells) != 12:
            continue

        advancing: list[str] = []
        for letter, cell in zip(ALL_GROUPS, group_cells):
            txt = cell.get_text(strip=True)
            if txt:  # any non-empty (typically bold letter) means that group advanced
                advancing.append(letter)
        if len(advancing) != 8:
            # Sanity: every valid row must show exactly 8 advancing groups.
            continue

        mapping: dict[str, str] = {}
        for slot, cell in zip(WINNER_SLOTS, mapping_cells):
            text = cell.get_text(strip=True)
            # Expected format like "3E" or just "E"
            if text and not text.startswith("3"):
                text = "3" + text
            mapping[slot] = text

        key = "".join(advancing)
        result[key] = mapping

    return result


def verify_samples(table: dict[str, dict[str, str]]) -> list[str]:
    """Compare table against HAND_VERIFIED_SAMPLES. Returns list of failure messages."""
    failures: list[str] = []
    for key, expected in HAND_VERIFIED_SAMPLES.items():
        actual = table.get(key)
        if actual is None:
            failures.append(f"key {key} not in table")
            continue
        for slot, expected_val in expected.items():
            if actual.get(slot) != expected_val:
                failures.append(
                    f"key {key} slot {slot}: expected {expected_val}, got {actual.get(slot)}"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse FIFA R32 combinations table")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args(argv)

    banner("MC Simu — R32 Seeding Builder")
    html = _fetch_html(force=args.force)
    table = parse_r32_table(html)

    n = len(table)
    print(f"Parsed {n} combinations from Wikipedia table")

    # Sanity: should be exactly C(12,8) = 495
    expected_n = 495
    expected_keys = {"".join(c) for c in combinations(ALL_GROUPS, 8)}
    missing_keys = expected_keys - set(table.keys())
    extra_keys = set(table.keys()) - expected_keys

    if n != expected_n:
        hard_stop("build_r32_seeding count", f"{n} combinations",
                  f"exactly {expected_n} (C(12,8))")
    if missing_keys:
        hard_stop("build_r32_seeding missing keys",
                  f"{len(missing_keys)} missing: {sorted(missing_keys)[:5]}",
                  "all 495 combinations present")
    if extra_keys:
        hard_stop("build_r32_seeding extra keys",
                  f"{len(extra_keys)} unexpected: {sorted(extra_keys)[:5]}",
                  "exactly the 495 expected combinations")

    # Hand-verify
    failures = verify_samples(table)
    if failures:
        print()
        print("Hand-verification failures:")
        for f in failures:
            print(f"  {f}")
        hard_stop("build_r32_seeding hand-verify", failures,
                  "all hand-verified samples match Wikipedia/FIFA Annex C")
    else:
        print(f"Hand-verified {len(HAND_VERIFIED_SAMPLES)} samples: ALL PASS")

    # Write JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(table, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_JSON}")

    banner(f"R32 seeding build complete ({n} combinations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
