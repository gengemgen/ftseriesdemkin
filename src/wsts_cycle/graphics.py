"""Forecast plots for the anchor model."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from wsts_cycle import plotting
from wsts_cycle.model import AnchorForecastResults


def plot_anchor_forecast(
    forecast: AnchorForecastResults,
    pdf_path: Path,
    disclosure: str | None = None,
) -> None:
    """Plot a fixed-origin forecast against realized log billings.

    Parameters
    ----------
    forecast : AnchorForecastResults
        Forecast results returned by :func:`wsts_cycle.model.forecast_anchor`.
    pdf_path : pathlib.Path
        Paper figure destination.
    disclosure : str, optional
        Additional episode-specific figure note.
    """

    plotting.apply_style()
    fit = forecast.fit
    frame = forecast.frame
    origin = fit.episode.origin

    figure, axis = plt.subplots(figsize=(plotting.FULL_WIDTH, 3.6))
    tail = fit.endog.loc[origin - pd.DateOffset(months=35) :]
    axis.plot(
        tail.index,
        tail.values,
        color=plotting.INK,
        linewidth=1.1,
        label="training (log SA)",
    )
    axis.fill_between(
        frame.index,
        frame["log_sa_lo95"],
        frame["log_sa_hi95"],
        color=plotting.BAND,
        alpha=0.55,
        linewidth=0,
        label="95% band",
    )
    axis.plot(
        frame.index,
        frame["log_sa_median"],
        color=plotting.STEEL,
        linewidth=1.0,
        linestyle="--",
        label="median forecast",
    )
    axis.plot(
        frame.index,
        frame["actual_log_sa"],
        color=plotting.ACCENT,
        linewidth=1.2,
        label="actual (log SA)",
    )
    axis.axvline(origin, color=plotting.STEEL, linewidth=0.8, linestyle=":")
    axis.annotate(
        f"origin {origin:%b %Y}",
        xy=(origin, axis.get_ylim()[0]),
        xytext=(4, 8),
        textcoords="offset points",
        fontsize=7,
        color=plotting.STEEL,
    )
    axis.set_ylabel("log billings (SA, log thousand USD)")
    axis.xaxis.set_major_locator(mdates.YearLocator(1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.legend(loc="upper left")

    note = (
        "Source: WSTS Historical Billings Report, May 2026 vintage; raw "
        "monthly (unsmoothed) worldwide series, seasonally adjusted using "
        "training-window factors."
    )
    if disclosure:
        note += "\n" + disclosure
    figure.text(0.01, 0.01, note, fontsize=6.5, color=plotting.NOTE)
    figure.tight_layout(rect=(0, 0.08, 1, 1))

    plotting.save_figure(figure, pdf_path)
    plt.close(figure)
