"""Compare the matched dot-com and AI-era forecast outcomes."""

from collections.abc import Callable
from typing import Any

import pandas as pd

from wsts_cycle.episodes import HORIZON_MONTHS, get_episode
from wsts_cycle.model import fit_anchor, forecast_anchor, forecast_summary
from wsts_cycle.paths import PAPER_TABLES_DIR, TABLES_DIR

N_H = HORIZON_MONTHS
COVERAGE = 0.95
TEX = PAPER_TABLES_DIR / "tab_verdict_anchor.tex"
CSV = TABLES_DIR / "verdict_anchor.csv"


def summarize(key: str) -> dict[str, Any]:
    """Summarize one recorded forecast table."""

    episode = get_episode(key)
    frame = forecast_anchor(fit_anchor(key)).frame
    if (
        len(frame) != N_H
        or frame.index[0] != episode.forecast_start
        or frame.index[-1] != episode.forecast_end
    ):
        raise RuntimeError(f"{key}: forecast table does not match the design")
    return {"label": episode.label, **forecast_summary(frame)}


def main() -> None:
    """Save the two-episode comparison as CSV and LaTeX tables."""

    dotcom = summarize("dotcom")
    ai = summarize("ai")

    def side(
        name: str, formatter: Callable[[Any], str], key: str
    ) -> tuple[str, str, str]:
        """Format one metric for the two table columns."""

        return name, formatter(dotcom[key]), formatter(ai[key])

    rows = [
        side("Months outside 95\\% band (of 33)", str, "n_out"),
        (
            "\\quad direction (above / below)",
            f"{dotcom['above']} / {dotcom['below']}",
            f"{ai['above']} / {ai['below']}",
        ),
        (
            "Expected months outside under 95\\% coverage",
            f"{N_H * (1 - COVERAGE):.2f}",
            f"{N_H * (1 - COVERAGE):.2f}",
        ),
        (
            "First exceedance",
            f"$h={dotcom['first_h']}$ ({dotcom['first_month']})",
            f"$h={ai['first_h']}$ ({ai['first_month']})",
        ),
        side("Longest consecutive run outside", str, "longest_run"),
        (
            "Outside at sample end",
            "yes" if dotcom["out_at_end"] else "no",
            "yes ($%d$ terminal months)" % ai["terminal_run"]
            if ai["out_at_end"]
            else "no",
        ),
        (
            "Max $|z|$ deviation (month)",
            f"{dotcom['max_z']:.2f} ({dotcom['max_z_month']})",
            f"{ai['max_z']:.2f} ({ai['max_z_month']})",
        ),
        side("MSE, log space", lambda value: f"{value:.3f}", "mse_log"),
        side("MAE, log space", lambda value: f"{value:.3f}", "mae_log"),
        side(
            "MAPE, SA levels (\\%)",
            lambda value: f"{value:.1f}",
            "mape_pct",
        ),
    ]

    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & Dot-com & AI era \\",
        r"\midrule",
    ]
    for name, dotcom_value, ai_value in rows:
        lines.append(f"{name} & {dotcom_value} & {ai_value} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    comparison = pd.DataFrame({"dotcom": dotcom, "ai": ai}).drop(
        index="label"
    )
    CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(CSV, index_label="metric")

    print(f"wrote {TEX}\nwrote {CSV}")


if __name__ == "__main__":
    main()
