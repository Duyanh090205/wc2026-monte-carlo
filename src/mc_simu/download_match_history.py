"""Download international match history from GitHub mirror of Kaggle dataset.

Dataset: martj42/international-football-results-from-1872-to-2017 (auto-updated to current).
Mirror : https://github.com/martj42/international_results (no auth required).

CLI:
    python src/mc_simu/download_match_history.py            # idempotent (skip if exists)
    python src/mc_simu/download_match_history.py --force    # re-download
    python src/mc_simu/download_match_history.py --skip-download  # validation only

Output:
    data/mc_simu/results.csv
    data/mc_simu/shootouts.csv
    data/mc_simu/goalscorers.csv

Alternative methods (when mirror falls behind or for re-sync) documented in
audit_report.md per spec §1.4.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import requests  # noqa: E402

from mc_simu._common import banner, hard_stop  # noqa: E402

MIRROR_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"

FILES = ["results.csv", "shootouts.csv", "goalscorers.csv"]

# Minimum expected bytes — sanity check vs truncated/error responses.
# results.csv was ~3.7 MB as of 2026-05-17 (verified via HEAD request).
MIN_BYTES = {"results.csv": 1_000_000, "shootouts.csv": 5_000, "goalscorers.csv": 1_000_000}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def download_one(filename: str, *, force: bool) -> tuple[bool, int]:
    """Download one file from the GitHub mirror. Returns (downloaded, bytes).

    Skips if file already exists unless force=True.
    """
    url = f"{MIRROR_BASE}/{filename}"
    dest = DATA_DIR / filename

    if dest.exists() and not force:
        size = dest.stat().st_size
        print(f"  {filename:25s} already present ({size:,} bytes) — skip (use --force to redownload)")
        return False, size

    print(f"  {filename:25s} fetching {url} ...")
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        hard_stop(
            f"download_match_history({filename})",
            f"HTTP {resp.status_code}",
            "HTTP 200 from GitHub mirror",
        )
    size = len(resp.content)
    min_required = MIN_BYTES.get(filename, 100)
    if size < min_required:
        hard_stop(
            f"download_match_history({filename})",
            f"{size:,} bytes (truncated?)",
            f">= {min_required:,} bytes",
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    print(f"  {filename:25s} saved {size:,} bytes  sha256={_sha256(dest)}")
    return True, size


def validate_results_csv() -> None:
    """Quick smoke check on results.csv columns + row count."""
    dest = DATA_DIR / "results.csv"
    if not dest.exists():
        hard_stop("validate_results_csv", "file missing", str(dest))

    with dest.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
        sample = f.readline().strip()
        total_rows = 1 + sum(1 for _ in f)  # include sample line already read

    expected_cols = {"date", "home_team", "away_team", "home_score", "away_score",
                     "tournament", "neutral", "country"}
    actual_cols = set(header.split(","))
    missing = expected_cols - actual_cols
    if missing:
        hard_stop("validate_results_csv header", f"missing {missing}", f"⊇ {expected_cols}")

    print(f"  results.csv validated: header={header!r}")
    print(f"                         sample={sample!r}")
    print(f"                         rows={total_rows:,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download international match history CSV")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Validate existing files without downloading",
    )
    args = parser.parse_args(argv)

    banner("MC Simu — Match History Download")
    print(f"Mirror: {MIRROR_BASE}")
    print(f"Target: {DATA_DIR}/")
    print()

    if not args.skip_download:
        total_downloaded = 0
        total_bytes = 0
        for f in FILES:
            did, b = download_one(f, force=args.force)
            if did:
                total_downloaded += 1
            total_bytes += b
        print()
        print(f"Downloaded {total_downloaded}/{len(FILES)} files, {total_bytes:,} bytes total on disk")
    else:
        print("Skip download (--skip-download), validating existing files only")
    print()

    print("Validation:")
    validate_results_csv()

    banner("Download complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
