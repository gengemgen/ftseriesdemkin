"""Document automatic ARIMA selection and the differencing check."""

import pandas as pd
from statsmodels.tsa.stattools import acf

from wsts_cycle.model import (
    AUTO_MAX_P,
    AUTO_MAX_Q,
    effective_auto_bounds,
    fit_anchor,
    selection_table,
)
from wsts_cycle.episodes import EPISODES
from wsts_cycle.paths import PAPER_TABLES_DIR, TABLES_DIR


SELECTION_CSV = TABLES_DIR / "autoarima_selection.csv"
DDIAG_CSV = TABLES_DIR / "d_diagnostics.csv"
TEX_SELECTION = PAPER_TABLES_DIR / "tab_autoarima_selection.tex"
TEX_D = PAPER_TABLES_DIR / "tab_d_check.tex"


def d_diagnostics(dy: pd.Series) -> dict[str, float]:
    """Compute the over-differencing diagnostics reported in the paper."""

    d2y = dy.diff().dropna()
    return {
        "acf1_dlog": float(acf(dy, nlags=1, fft=False)[1]),
        "acf1_d2log": float(acf(d2y, nlags=1, fft=False)[1]),
        "var_dlog": float(dy.var()),
        "var_d2log": float(d2y.var()),
        "var_ratio": float(d2y.var() / dy.var()),
    }


def main() -> None:
    """Run automatic model selection and save CSV and LaTeX tables."""

    fits = {key: fit_anchor(key) for key in EPISODES}
    selections = pd.concat([selection_table(fit) for fit in fits.values()])
    diags = {key: d_diagnostics(fit.delta_endog) for key, fit in fits.items()}

    assert (selections["max_p"] == AUTO_MAX_P).all()
    assert (selections["max_q"] == AUTO_MAX_Q).all()
    effective_p, effective_q = effective_auto_bounds(120)
    assert (selections["effective_max_p"] == effective_p).all()
    assert (selections["effective_max_q"] == effective_q).all()
    assert (selections[["p", "q"]] >= 0).all().all()
    assert (selections["p"] <= AUTO_MAX_P).all()
    assert (selections["q"] <= AUTO_MAX_Q).all()
    SELECTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    selections.to_csv(SELECTION_CSV)
    pd.DataFrame(diags).T.rename_axis("episode").to_csv(DDIAG_CSV)

    selection_rows = []
    for key, episode in EPISODES.items():
        fit = fits[key]
        p, d, q = fit.selection.order
        drift = "yes" if fit.selection.include_drift else "no"
        selection_rows.append(
            f"{episode.label} & ARIMA({p},{d},{q}) & {drift} & "
            f"{fit.selection.aic:.2f} \\\\"
        )

    TEX_SELECTION.parent.mkdir(parents=True, exist_ok=True)
    TEX_SELECTION.write_text(
        "\\begin{tabular}{llcr}\n\\toprule\n"
        "Episode & Selected model & Drift & Selection AIC \\\\\n"
        "\\midrule\n"
        + "\n".join(selection_rows)
        + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )

    d_rows = []
    for key, episode in EPISODES.items():
        values = diags[key]
        d_rows.append(
            f"{episode.label} & ${values['acf1_dlog']:+.2f}$ & "
            f"${values['acf1_d2log']:+.2f}$ & {values['var_ratio']:.2f} \\\\"
        )
    TEX_D.write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        " & ACF(1) of $\\Delta\\log$ & ACF(1) of $\\Delta^2\\log$ & "
        "$\\mathrm{var}(\\Delta^2)/\\mathrm{var}(\\Delta)$ \\\\\n"
        "\\midrule\n"
        + "\n".join(d_rows)
        + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )

    print(
        f"wrote {SELECTION_CSV.name}, {DDIAG_CSV.name}, "
        f"{TEX_SELECTION.name}, {TEX_D.name}"
    )


if __name__ == "__main__":
    main()
