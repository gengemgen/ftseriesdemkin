"""Shared output workflow for the two anchor episodes."""

from pathlib import Path

from wsts_cycle.episodes import AI_DATED_TROUGH, AI_ORIGIN
from wsts_cycle.graphics import plot_anchor_forecast
from wsts_cycle.model import (
    AnchorFitResults,
    AnchorForecastResults,
    fit_anchor,
    forecast_anchor,
    residual_diagnostics,
    residual_diagnostics_table,
    seasonal_factors_by_month,
)
from wsts_cycle.paths import PAPER_FIGURES_DIR, TABLES_DIR

ORIGIN_DISCLOSURE = (
    f"AI-era origin {AI_ORIGIN:%Y-%m} is design-fixed; the dating rule's own "
    f"candidate trough is {AI_DATED_TROUGH:%Y-%m}."
)


def _output_paths(episode: str) -> dict[str, Path]:
    """Return standardized output paths for an anchor episode."""

    if episode not in {"dotcom", "ai"}:
        raise ValueError("anchor outputs are defined only for 'dotcom' and 'ai'")
    return {
        "seasonal_csv": TABLES_DIR / f"{episode}_autoarima_seasonal_factors.csv",
        "residual_csv": TABLES_DIR / f"{episode}_autoarima_residual_diagnostics.csv",
        "forecast_csv": TABLES_DIR / f"{episode}_autoarima_forecast.csv",
        "forecast_pdf": PAPER_FIGURES_DIR / f"fig_{episode}_autoarima_forecast.pdf",
    }


def run_anchor_analysis(
    episode: str,
) -> tuple[AnchorFitResults, AnchorForecastResults]:
    """Fit and forecast one episode without refitting for each output.

    Parameters
    ----------
    episode : {"dotcom", "ai"}
        Anchor episode to estimate.

    Returns
    -------
    tuple[AnchorFitResults, AnchorForecastResults]
        The single fitted model and its forecast results.
    """

    paths = _output_paths(episode)
    fit = fit_anchor(episode)
    diagnostics = residual_diagnostics(fit)
    forecast = forecast_anchor(fit)

    paths["seasonal_csv"].parent.mkdir(parents=True, exist_ok=True)
    seasonal_factors_by_month(fit.seasonal_factors).to_csv(
        paths["seasonal_csv"]
    )
    residual_diagnostics_table(diagnostics).to_csv(paths["residual_csv"])
    frame = forecast.frame
    frame.to_csv(paths["forecast_csv"], index_label="date")
    disclosure = ORIGIN_DISCLOSURE if episode == "ai" else None
    plot_anchor_forecast(
        forecast,
        paths["forecast_pdf"],
        disclosure=disclosure,
    )

    print(
        f"selected ARIMA{fit.selection.order} "
        f"{'with' if fit.selection.include_drift else 'without'} drift\n"
        f"wrote {paths['seasonal_csv']}\n"
        f"wrote {paths['residual_csv']}\n"
        f"wrote {paths['forecast_csv']}\n"
        f"wrote {paths['forecast_pdf']}"
    )
    return fit, forecast
