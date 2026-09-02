"""Compare primary automatic fits with a common ARIMA(5,1,0) benchmark."""

import pandas as pd

from wsts_cycle.episodes import EPISODES
from wsts_cycle.model import (
    COMMON_ORDER,
    fit_anchor,
    fit_fixed_anchor,
    forecast_anchor,
    forecast_summary,
)
from wsts_cycle.paths import PAPER_TABLES_DIR, TABLES_DIR


CSV = TABLES_DIR / "common_order_robustness.csv"
TEX = PAPER_TABLES_DIR / "tab_common_order_robustness.tex"


def main() -> None:
    """Save the common-order robustness results as CSV and LaTeX tables."""

    records = []
    for episode_key, episode in EPISODES.items():
        fits = (
            ("Automatic", fit_anchor(episode_key)),
            ("Common order", fit_fixed_anchor(episode_key, COMMON_ORDER)),
        )
        for specification, fit in fits:
            forecast = forecast_anchor(fit)
            order = fit.selection.order
            summary = forecast_summary(forecast.frame)
            records.append(
                {
                    "episode": episode_key,
                    "episode_label": episode.label,
                    "specification": specification,
                    "p": order[0],
                    "d": order[1],
                    "q": order[2],
                    "months_above": summary["above"],
                    "months_below": summary["below"],
                    "longest_outside": summary["longest_run"],
                    "max_abs_z": summary["max_z"],
                    "terminal_outside": summary["out_at_end"],
                    "mse_log": summary["mse_log"],
                    "mae_log": summary["mae_log"],
                    "mape_pct": summary["mape_pct"],
                }
            )

    table = pd.DataFrame.from_records(records)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV, index=False)

    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Episode & Model & Above / 33 & Longest & Max $|z|$ & Terminal & "
        r"MAE$_{\log}$ & MAPE (\%) \\",
        r"\midrule",
    ]
    for row in records:
        prefix = "Auto" if row["specification"] == "Automatic" else "Common"
        model = f"{prefix} ({row['p']},{row['d']},{row['q']})"
        terminal = "yes" if row["terminal_outside"] else "no"
        lines.append(
            f"{row['episode_label']} & {model} & "
            f"{row['months_above']} & {row['longest_outside']} & "
            f"{row['max_abs_z']:.2f} & {terminal} & "
            f"{row['mae_log']:.3f} & {row['mape_pct']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {CSV}\nwrote {TEX}")


if __name__ == "__main__":
    main()
