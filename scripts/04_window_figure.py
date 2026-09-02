"""Plot the matched training windows and forecast horizons."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from wsts_cycle import plotting
from wsts_cycle.data import load_monthly
from wsts_cycle.episodes import (
    AI_DATED_TROUGH,
    AI_ORIGIN,
    FORECAST_ORIGINS,
    HORIZON_MONTHS,
    TRAINING_MONTHS,
)
from wsts_cycle.paths import PAPER_FIGURES_DIR

FIG_PDF = PAPER_FIGURES_DIR / "fig_anchor_windows.pdf"

TRAIN = plotting.BAND
HORIZON = plotting.BAND2
LINE = plotting.INK
INK = plotting.STEEL

EPISODES = [
    ("dot-com", FORECAST_ORIGINS["dotcom"], 100, "top"),
    ("AI era", FORECAST_ORIGINS["ai"], 2.0, "bottom"),
]


def main() -> None:
    """Write the paper window figure as PDF."""

    plotting.apply_style()
    s = load_monthly() / 1e6

    fig, ax = plt.subplots(figsize=(plotting.FULL_WIDTH, 3.6))
    for name, origin, y_lab, va in EPISODES:
        t0 = origin - pd.DateOffset(months=TRAINING_MONTHS - 1)
        h_end = origin + pd.DateOffset(months=HORIZON_MONTHS)
        ax.axvspan(t0, origin, color=TRAIN, alpha=0.5, linewidth=0)
        ax.axvspan(origin, h_end, color=HORIZON, alpha=0.5, linewidth=0)
        ax.axvline(origin, color=INK, linestyle=":", linewidth=0.9)
        ax.annotate(
            f"{name} training\n{t0:%Y-%m} .. {origin:%Y-%m}",
            xy=(t0 + (origin - t0) / 2, y_lab),
            ha="center",
            va=va,
            fontsize=7,
            color=INK,
            linespacing=1.15,
        )
        ax.annotate(
            f"horizon\n{HORIZON_MONTHS} m",
            xy=(origin + (h_end - origin) / 2, y_lab),
            ha="center",
            va=va,
            fontsize=7,
            color=INK,
            linespacing=1.15,
        )

    ax.plot(s.index, s.values, color=LINE, linewidth=1.1)

    ax.set_yscale("log")
    ax.set_ylim(1.5, 160)
    ax.set_yticks([2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(s.index.min(), s.index.max())

    ax.set_ylabel("Billings (billion USD, nominal, log scale)")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)

    fig.text(
        0.01,
        0.01,
        "Source: WSTS Historical Billings Report, May 2026 vintage; raw "
        "monthly (unsmoothed) series. Windows and horizons follow the "
        f"fixed design.\nAI-era origin {AI_ORIGIN:%Y-%m} is design-fixed; "
        f"the dating rule's own candidate trough is {AI_DATED_TROUGH:%Y-%m}.",
        fontsize=6.5,
        color=plotting.NOTE,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    plotting.save_figure(
        fig,
        FIG_PDF,
    )
    print(f"wrote {FIG_PDF}")


if __name__ == "__main__":
    main()
