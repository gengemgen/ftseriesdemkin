"""Filesystem locations used by the replication package."""

import os
from pathlib import Path


def _find_code_root() -> Path:
    """Locate the replication directory in editable and installed use."""

    configured = os.environ.get("WSTS_CYCLE_CODE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    package_candidate = Path(__file__).resolve().parents[2]
    working_directory = Path.cwd().resolve()
    candidates = (
        package_candidate,
        working_directory,
        working_directory / "CODE",
    )
    for candidate in candidates:
        if (candidate / "data").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return package_candidate


CODE_ROOT = _find_code_root()

DATA_DIR = CODE_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = CODE_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

PAPER_DIR = CODE_ROOT / "paper"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
PAPER_TABLES_DIR = PAPER_DIR / "tables"
