"""Shared journal-style plotting configuration."""

from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

INK = "#1a1a2e"
STEEL = "#3a4a63"
ACCENT = "#a33b3b"
BAND = "#b8c4d8"
BAND2 = "#e0c4bd"
NOTE = "#666666"
GRID = "#d0d0d0"

FULL_WIDTH = 6.3


def apply_style() -> None:
    """Apply the common figure style to Matplotlib's runtime settings."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
        }
    )


def save_figure(
    figure: Figure,
    pdf_path: Path,
) -> None:
    """Save a paper figure as PDF.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.
    pdf_path : pathlib.Path
        Destination used by the paper.
    """

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path)
