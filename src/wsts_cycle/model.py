"""Automatically selected ARIMA anchor models and forecasts."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from statsforecast.models import AutoARIMA
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

from wsts_cycle.data import load_monthly
from wsts_cycle.episodes import (
    AI_ORIGIN,
    HORIZON_MONTHS,
    TRAINING_MONTHS,
    Episode,
    get_episode,
)

AUTO_MAX_P = 50
AUTO_MAX_Q = 50
AUTO_MAX_ORDER = AUTO_MAX_P + AUTO_MAX_Q
AUTO_NMODELS = 500
ACF_LAGS = 24
COMMON_ORDER = (5, 1, 0)
FORECAST_Z99 = 2.5758293035489004


def effective_auto_bounds(n_obs: int) -> tuple[int, int]:
    """Return StatsForecast's sample-size-adjusted AR and MA bounds."""

    return min(AUTO_MAX_P, n_obs // 3), min(AUTO_MAX_Q, n_obs // 3)


@dataclass(frozen=True)
class ARIMASelection:
    """ARIMA specification and selection AIC for an anchor fit."""

    order: tuple[int, int, int]
    trend: str
    include_drift: bool
    aic: float


@dataclass(frozen=True)
class AnchorFitResults:
    """Data and fitted results for one anchor episode.

    Parameters
    ----------
    episode : Episode
        Fixed episode definition.
    seasonal_factors : pandas.Series
        Multiplicative seasonal factors estimated within the window.
    endog : pandas.Series
        Log seasonally adjusted billings.
    delta_endog : pandas.Series
        First difference of log seasonally adjusted billings.
    selection : ARIMASelection
        Order, drift treatment, and AIC. Primary fits use AutoARIMA;
        fixed-order robustness fits record their refit AIC.
    results : statsmodels.tsa.arima.model.ARIMAResultsWrapper
        Maximum-likelihood model results.
    """

    episode: Episode
    seasonal_factors: pd.Series
    endog: pd.Series
    delta_endog: pd.Series
    selection: ARIMASelection
    results: ARIMAResultsWrapper


@dataclass(frozen=True)
class ResidualDiagnostics:
    """Residual autocorrelation diagnostics for an anchor fit."""

    residuals: pd.Series
    acf: np.ndarray
    bound: float
    exceedances: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class AnchorForecastResults:
    """Forecast table for one fitted episode."""

    fit: AnchorFitResults
    frame: pd.DataFrame


def training_window(
    origin: pd.Timestamp = AI_ORIGIN,
    months: int = TRAINING_MONTHS,
) -> pd.Series:
    """Construct a fixed-length monthly estimation window.

    Parameters
    ----------
    origin : pandas.Timestamp, optional
        Final observation included in the window.
    months : int, optional
        Required number of monthly observations.
    Returns
    -------
    pandas.Series
        Requested estimation window.
    """

    if months < 1:
        raise ValueError("months must be positive")
    endog = load_monthly()
    start = origin - pd.DateOffset(months=months - 1)
    window = endog.loc[start:origin]
    if len(window) != months:
        raise ValueError(
            f"window {start:%Y-%m}..{origin:%Y-%m} has {len(window)} obs"
        )
    return window


def seasonally_adjust(endog: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Apply multiplicative classical seasonal decomposition.

    Parameters
    ----------
    endog : pandas.Series
        Monthly level series with a frequency-aware index.

    Returns
    -------
    adjusted : pandas.Series
        Seasonally adjusted level series.
    factors : pandas.Series
        Multiplicative seasonal factors.
    """

    decomposition = seasonal_decompose(endog, model="multiplicative", period=12)
    return endog / decomposition.seasonal, decomposition.seasonal


def _autoarima_selector() -> AutoARIMA:
    """Construct the single AutoARIMA configuration used by the project."""

    return AutoARIMA(
        d=1,
        seasonal=False,
        ic="aic",
        stepwise=True,
        max_p=AUTO_MAX_P,
        max_q=AUTO_MAX_Q,
        max_order=AUTO_MAX_ORDER,
        nmodels=AUTO_NMODELS,
        approximation=False,
        # A time regressor forces drift; do not add a duplicate drift term.
        allowdrift=False,
        allowmean=False,
    )


def _fit_autoarima(endog: pd.Series) -> AutoARIMA:
    """Fit the configured selector with the design's required drift."""

    selector = _autoarima_selector()
    time = np.arange(len(endog), dtype=float).reshape(-1, 1)
    selector.fit(endog.to_numpy(), X=time)
    return selector


def _selection_from_selector(selector: AutoARIMA) -> ARIMASelection:
    """Extract the selected specification and criteria from a fitted selector."""

    fitted = selector.model_
    p, q, _P, _Q, _period, d, _D = fitted["arma"]
    return ARIMASelection(
        order=(int(p), int(d), int(q)),
        trend="t",
        include_drift=True,
        aic=float(fitted["aic"]),
    )


def select_arima(endog: pd.Series) -> ARIMASelection:
    """Select nonseasonal ``p`` and ``q`` by stepwise AIC.

    The configured AR and MA caps are 50, including ARIMA(0,1,0), but
    StatsForecast reduces each cap to one third of the sample length. It uses
    the Hyndman--Khandakar stepwise procedure rather than fitting the complete
    effective grid. The differencing order is fixed at one by the level-series
    behavior and the separate over-differencing diagnostic used in the paper.
    The design requires drift through a linear time regressor.

    Parameters
    ----------
    endog : pandas.Series
        Log seasonally adjusted billings in one training window.
    Returns
    -------
    ARIMASelection
        Selected order and information criteria from the automatic search.
    """

    selector = _fit_autoarima(endog)
    return _selection_from_selector(selector)


@lru_cache(maxsize=None)
def fit_anchor(episode: str) -> AnchorFitResults:
    """Automatically select, fit, and retain one anchor model.

    Parameters
    ----------
    episode : str
        Key identifying a fixed episode.

    Returns
    -------
    AnchorFitResults
        Training data, automatic selection, and statsmodels results object.
    """

    definition = get_episode(episode)
    raw_endog = training_window(origin=definition.origin)
    adjusted, factors = seasonally_adjust(raw_endog)
    endog = np.log(adjusted).asfreq("MS")
    endog.name = "log_sa_billings"
    delta_endog = endog.diff().dropna().asfreq("MS")
    delta_endog.name = "dlog_sa"

    selection = select_arima(endog)
    model = ARIMA(endog, order=selection.order, trend=selection.trend)
    results = model.fit(method_kwargs={"maxiter": 1000})
    return AnchorFitResults(
        episode=definition,
        seasonal_factors=factors,
        endog=endog,
        delta_endog=delta_endog,
        selection=selection,
        results=results,
    )


@lru_cache(maxsize=None)
def fit_fixed_anchor(
    episode: str, order: tuple[int, int, int] = COMMON_ORDER
) -> AnchorFitResults:
    """Refit one anchor window at a fixed order with required drift.

    This function supports the common-order robustness check. It reuses the
    exact training data and within-window seasonal adjustment of the primary
    automatic fit, but it does not alter or replace that fit.

    Parameters
    ----------
    episode : str
        Key identifying a fixed episode.
    order : tuple[int, int, int], optional
        Fixed ARIMA order. The robustness design requires ``d=1``.

    Returns
    -------
    AnchorFitResults
        Fixed-order maximum-likelihood fit on the fixed training window.
    """

    if order[1] != 1:
        raise ValueError("fixed-order robustness fits require d=1")
    primary = fit_anchor(episode)
    model = ARIMA(primary.endog, order=order, trend="t")
    results = model.fit(method_kwargs={"maxiter": 1000})
    selection = ARIMASelection(
        order=order,
        trend="t",
        include_drift=True,
        aic=float(results.aic),
    )
    return AnchorFitResults(
        episode=primary.episode,
        seasonal_factors=primary.seasonal_factors,
        endog=primary.endog,
        delta_endog=primary.delta_endog,
        selection=selection,
        results=results,
    )


def seasonal_factors_by_month(factors: pd.Series) -> pd.Series:
    """Collapse repeated seasonal factors to one factor per month."""

    return (
        factors.groupby(factors.index.month)
        .first()
        .rename_axis("month")
        .rename("factor")
    )


def residual_diagnostics(
    fit: AnchorFitResults, nlags: int = ACF_LAGS
) -> ResidualDiagnostics:
    """Compute innovation ACF values and conventional approximate bounds."""

    burn = fit.results.loglikelihood_burn
    innovations = fit.results.filter_results.standardized_forecasts_error[0]
    residuals = pd.Series(
        innovations,
        index=fit.endog.index,
        name="standardized_innovation",
    ).iloc[burn:]
    values = acf(residuals, nlags=nlags, fft=False)
    bound = 1.96 / np.sqrt(len(residuals))
    exceedances = tuple(
        (lag, float(values[lag]))
        for lag in range(1, nlags + 1)
        if abs(values[lag]) > bound
    )
    return ResidualDiagnostics(residuals, values, bound, exceedances)


def residual_diagnostics_table(
    diagnostics: ResidualDiagnostics,
) -> pd.DataFrame:
    """Return the numerical residual-ACF audit used by the paper."""

    exceeded_lags = {lag for lag, _ in diagnostics.exceedances}
    lags = np.arange(1, len(diagnostics.acf))
    return pd.DataFrame(
        {
            "acf": diagnostics.acf[1:],
            "bound": diagnostics.bound,
            "exceeds_bound": [lag in exceeded_lags for lag in lags],
        },
        index=pd.Index(lags, name="lag"),
    )


def selection_table(fit: AnchorFitResults) -> pd.DataFrame:
    """Return one row documenting the automatic specification choice."""

    selected = fit.selection
    p, d, q = selected.order
    effective_p, effective_q = effective_auto_bounds(len(fit.endog))
    return pd.DataFrame(
        {
            "p": [p],
            "d": [d],
            "q": [q],
            "drift": [selected.include_drift],
            "search_aic": [selected.aic],
            "max_p": [AUTO_MAX_P],
            "max_q": [AUTO_MAX_Q],
            "effective_max_p": [effective_p],
            "effective_max_q": [effective_q],
            "stepwise": [True],
        },
        index=pd.Index([fit.episode.key], name="episode"),
    )


def _actual_log_sa(fit: AnchorFitResults, index: pd.DatetimeIndex) -> pd.Series:
    """Seasonally adjust realized outcomes using training-window factors."""

    actual = load_monthly().loc[index[0] : index[-1]]
    if len(actual) != len(index):
        raise ValueError(
            f"forecast horizon contains {len(actual)} actuals, not {len(index)}"
        )
    factors = seasonal_factors_by_month(fit.seasonal_factors)
    return pd.Series(
        np.log(actual.to_numpy() / factors.reindex(index.month).to_numpy()),
        index=index,
        name="actual_log_sa",
    )


def _forecast_frame(
    fit: AnchorFitResults,
    index: pd.DatetimeIndex,
    log_median: pd.Series,
    scale: pd.Series,
    lower95: pd.Series,
    upper95: pd.Series,
) -> pd.DataFrame:
    """Build the forecast table used for the reported results."""

    actual_log = _actual_log_sa(fit, index)
    direction = np.where(
        actual_log > upper95,
        "above",
        np.where(actual_log < lower95, "below", "in"),
    )
    standardized_error = (actual_log - log_median) / scale
    return pd.DataFrame(
        {
            "h": np.arange(1, len(index) + 1),
            "log_sa_median": log_median,
            "sd_h": scale,
            "log_sa_lo95": lower95,
            "log_sa_hi95": upper95,
            "actual_log_sa": actual_log,
            "z_dev": standardized_error,
            "band95": direction,
        }
    )


def forecast_anchor(
    fit: AnchorFitResults, steps: int = HORIZON_MONTHS
) -> AnchorForecastResults:
    """Produce a fixed-origin pseudo-out-of-sample forecast.

    Parameters
    ----------
    fit : AnchorFitResults
        Fitted anchor model.
    steps : int, optional
        Number of monthly forecasts.

    Returns
    -------
    AnchorForecastResults
        Forecast table, actual outcomes, and interval flags.
    """

    origin = fit.episode.origin
    if steps < 1:
        raise ValueError("steps must be positive")
    prediction = fit.results.get_forecast(steps=steps)
    summary95 = prediction.summary_frame(alpha=0.05)
    index = pd.date_range(
        origin + pd.DateOffset(months=1), periods=steps, freq="MS"
    )
    if not summary95.index.equals(index):
        raise RuntimeError("forecast index does not match the fixed horizon")

    log_median = summary95["mean"].rename("log_sa_median")
    scale = summary95["mean_se"].rename("sd_h")
    lower95 = summary95["mean_ci_lower"]
    upper95 = summary95["mean_ci_upper"]

    frame = _forecast_frame(
        fit, index, log_median, scale, lower95, upper95
    )
    return AnchorForecastResults(fit, frame)


def forecast_accuracy(
    frame: pd.DataFrame, expected_horizon: int = HORIZON_MONTHS
) -> dict[str, float]:
    """Calculate the paper's log-error and level-percentage metrics.

    MSE and MAE are evaluated on the log seasonally adjusted series, where
    the ARIMA models are estimated. MAPE is evaluated after exponentiating
    both actuals and conditional-median forecasts back to seasonally adjusted
    levels, matching the standard percentage-error definition.

    Parameters
    ----------
    frame : pandas.DataFrame
        Complete forecast table with actual and predicted log levels.
    expected_horizon : int, optional
        Required number of observations; defaults to the matched 33 months.
    """

    required = {"actual_log_sa", "log_sa_median"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"forecast frame is missing columns: {sorted(missing)}")
    if len(frame) != expected_horizon:
        raise ValueError(
            f"forecast frame has {len(frame)} rows, expected {expected_horizon}"
        )
    if not frame.index.is_unique:
        raise ValueError("forecast frame index must not contain duplicates")
    values = frame[["actual_log_sa", "log_sa_median"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("forecast accuracy inputs must all be finite")

    error_log = frame["actual_log_sa"] - frame["log_sa_median"]
    actual_level = np.exp(frame["actual_log_sa"])
    forecast_level = np.exp(frame["log_sa_median"])
    if (
        not np.isfinite(actual_level).all()
        or not np.isfinite(forecast_level).all()
        or (actual_level <= 0).any()
    ):
        raise ValueError("level forecasts must be finite and actuals positive")
    return {
        "mse_log": float(np.mean(error_log**2)),
        "mae_log": float(np.mean(np.abs(error_log))),
        "mape_pct": float(
            100.0
            * np.mean(np.abs((actual_level - forecast_level) / actual_level))
        ),
    }


def forecast_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize the forecast outcomes reported in the paper."""

    derived = np.where(
        frame["actual_log_sa"] > frame["log_sa_hi95"],
        "above",
        np.where(
            frame["actual_log_sa"] < frame["log_sa_lo95"],
            "below",
            "in",
        ),
    )
    if not (derived == frame["band95"].to_numpy()).all():
        raise RuntimeError("forecast band flags disagree with interval columns")

    outside = derived != "in"
    longest = current = 0
    for value in outside:
        current = current + 1 if value else 0
        longest = max(longest, current)

    terminal = 0
    for value in outside[::-1]:
        if not value:
            break
        terminal += 1

    count = int(outside.sum())
    return {
        "n_out": count,
        "above": int((derived == "above").sum()),
        "below": int((derived == "below").sum()),
        "first_h": int(frame.loc[outside, "h"].iloc[0]) if count else None,
        "first_month": (
            frame.index[outside][0].strftime("%Y-%m") if count else "--"
        ),
        "longest_run": longest,
        "terminal_run": terminal,
        "out_at_end": bool(outside[-1]),
        "max_z": float(frame["z_dev"].abs().max()),
        "max_z_month": frame["z_dev"].abs().idxmax().strftime("%Y-%m"),
        "n_out_99": int((frame["z_dev"].abs() > FORECAST_Z99).sum()),
        "z_end": float(frame["z_dev"].iloc[-1]),
        **forecast_accuracy(frame),
    }
