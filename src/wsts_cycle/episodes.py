"""Episode definitions fixed by the research design."""

from dataclasses import dataclass

import pandas as pd

TRAINING_MONTHS = 120
HORIZON_MONTHS = 33
DOTCOM_ORIGIN = pd.Timestamp("1998-05-01")
AI_DATED_TROUGH = pd.Timestamp("2023-03-01")
AI_ORIGIN = pd.Timestamp("2023-08-01")
SAMPLE_END = pd.Timestamp("2026-05-01")


@dataclass(frozen=True)
class Episode:
    """Definition of an estimation episode.

    Parameters
    ----------
    key : str
        Short identifier used in files and function calls.
    label : str
        Human-readable label used in tables and figures.
    origin : pandas.Timestamp
        Final month of the fixed training window.
    """

    key: str
    label: str
    origin: pd.Timestamp

    @property
    def training_start(self) -> pd.Timestamp:
        """First month of the 120-month training window."""

        return self.origin - pd.DateOffset(months=TRAINING_MONTHS - 1)

    @property
    def forecast_start(self) -> pd.Timestamp:
        """First month after the fixed forecast origin."""

        return self.origin + pd.DateOffset(months=1)

    @property
    def forecast_end(self) -> pd.Timestamp:
        """Last month of the matched 33-month forecast horizon."""

        return self.origin + pd.DateOffset(months=HORIZON_MONTHS)


EPISODES = {
    "dotcom": Episode("dotcom", "Dot-com", DOTCOM_ORIGIN),
    "ai": Episode("ai", "AI era", AI_ORIGIN),
}

FORECAST_ORIGINS = {key: episode.origin for key, episode in EPISODES.items()}

DATED_BOOM_WINDOWS = [
    (pd.Timestamp("1991-08-01"), pd.Timestamp("1995-12-01"), "1991-95\nboom"),
    (DOTCOM_ORIGIN, pd.Timestamp("2000-12-01"), "dot-com\nboom"),
    (pd.Timestamp("2016-03-01"), pd.Timestamp("2018-08-01"), "2017\nboom"),
    (pd.Timestamp("2019-06-01"), pd.Timestamp("2022-05-01"), "2021\nboom"),
    (AI_DATED_TROUGH, SAMPLE_END, "AI\nera"),
]


def get_episode(key: str) -> Episode:
    """Return an episode definition.

    Parameters
    ----------
    key : str
        Episode identifier.

    Returns
    -------
    Episode
        Frozen episode definition.

    Raises
    ------
    ValueError
        If `key` is not a defined episode.
    """

    try:
        return EPISODES[key]
    except KeyError as exc:
        choices = ", ".join(EPISODES)
        raise ValueError(f"unknown episode {key!r}; expected one of {choices}") from exc
