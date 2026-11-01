"""Repository locations shared by experiments and figure scripts.

Set ``MEMGP_RESULTS`` to send simulation outputs to another directory (for
example a scratch disk, or a throwaway folder for a quick test run) and
``MEMGP_FIGURES`` to redirect rendered figures.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "memgp" / "configs"
DATA_DIR = ROOT / "data"
RESULTS_DIR = Path(os.environ.get("MEMGP_RESULTS", ROOT / "results")).resolve()
FIGURES_DIR = Path(os.environ.get("MEMGP_FIGURES", ROOT / "figures" / "output")).resolve()


def results_dir(experiment: str) -> Path:
    """``results/<experiment>/``, created on first use."""
    path = RESULTS_DIR / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_path(experiment: str, name: str) -> Path:
    return results_dir(experiment) / name


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)
