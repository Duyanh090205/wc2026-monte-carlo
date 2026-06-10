"""Pytest fixtures + sys.path bootstrap for the mc_simu test suite.

Adds src/ to sys.path so test modules can `from mc_simu.X import ...`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest  # noqa: E402


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def mc_simu_data_dir() -> Path:
    return PROJECT_ROOT / "data" / "mc_simu"


@pytest.fixture
def mc_simu_fixtures_dir() -> Path:
    return PROJECT_ROOT / "tests" / "fixtures" / "mc_simu"
