"""Plot WSTS history with matched boom windows and historical expansions."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from wsts_cycle import plotting
from wsts_cycle.data import load_monthly
from wsts_cycle.episodes import (
    DATED_BOOM_WINDOWS,
    FORECAST_ORIGINS,
    HORIZON_MONTHS,
)
from wsts_cycle.paths import PAPER_FIGURES_DIR

FIG_PDF = PAPER_FIGURES_DIR / "fig_history_booms.pdf"


def main() -> None:
    """Write the paper history figure as PDF."""

    plotting.apply_style()
    s = load_monthly() / 1e6

    fig, ax = plt.subplots(figsize=(plotting.FULL_WIDTH, 3.6))
    selected_labels = {"dot-com\nboom", "AI\nera"}
    for start, end, label in DATED_BOOM_WINDOWS:
        if label in selected_labels:
            continue
        ax.axvspan(start, end, color=plotting.BAND, alpha=0.22, linewidth=0)
        ax.annotate(
            label,
            xy=(start + (end - start) / 2, 2.05),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#3a4a63",
            linespacing=1.1,
        )

    comparison_windows = [
        ("dot-com\nboom", FORECAST_ORIGINS["dotcom"]),
        ("AI era", FORECAST_ORIGINS["ai"]),
    ]
    for label, origin in comparison_windows:
        end = origin + pd.DateOffset(months=HORIZON_MONTHS)
        ax.axvspan(origin, end, color=plotting.BAND, alpha=0.5, linewidth=0)
        ax.axvline(origin, color=plotting.STEEL, linestyle=":", linewidth=0.9)
        ax.annotate(
            label,
            xy=(origin + (end - origin) / 2, 2.05),
            ha="center",
            va="bottom",
            fontsize=7,
            color=plotting.STEEL,
            linespacing=1.1,
        )
        ax.annotate(
            f"origin\n{origin:%b %Y}",
            xy=(origin, 95),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7,
            color=plotting.STEEL,
            linespacing=1.1,
        )

    ax.plot(s.index, s.values, color=plotting.INK, linewidth=1.1)

    ax.set_yscale("log")
    ax.set_ylim(1.5, 160)
    ax.set_yticks([2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(s.index.min(), s.index.max())

    ax.set_ylabel("Billings (billion USD, nominal, log scale)")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)

    fig.text(
        0.01,
        0.01,
        "Source: WSTS Historical Billings Report, May 2026 vintage; raw "
        "monthly (unsmoothed) series.\nDarker shading marks the matched "
        "origin-to-end boom comparison windows; lighter shading marks "
        "other mechanically dated expansions.",
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
