"""Apply the Bry-Boschan monthly turning-point rule."""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from wsts_cycle.data import load_monthly
from wsts_cycle.episodes import HORIZON_MONTHS, TRAINING_MONTHS
from wsts_cycle.paths import TABLES_DIR


K = 5
MIN_PHASE = 6
MIN_CYCLE = 15


def candidates(y: pd.Series) -> list[tuple[pd.Timestamp, str]]:
    """Locate strict local extrema using the fixed symmetric window."""

    out = []
    vals, idx = y.to_numpy(), y.index
    for t in range(K, len(y) - K):
        window = vals[t - K : t + K + 1]
        if vals[t] == window.max() and (window < vals[t]).sum() == 2 * K:
            out.append((idx[t], "peak"))
        elif vals[t] == window.min() and (window > vals[t]).sum() == 2 * K:
            out.append((idx[t], "trough"))
    return out


def alternate(
    points: list[tuple[pd.Timestamp, str]], y: pd.Series
) -> list[tuple[pd.Timestamp, str]]:
    """Enforce alternation by retaining the more extreme adjacent point."""

    result = []
    for date, kind in points:
        if result and result[-1][1] == kind:
            prev_date, _ = result[-1]
            better = (
                y[date] > y[prev_date] if kind == "peak" else y[date] < y[prev_date]
            )
            if better:
                result[-1] = (date, kind)
        else:
            result.append((date, kind))
    return result


def censor(
    points: list[tuple[pd.Timestamp, str]], y: pd.Series
) -> list[tuple[pd.Timestamp, str]]:
    """Apply the fixed minimum phase and cycle durations."""

    def months(a: pd.Timestamp, b: pd.Timestamp) -> int:
        """Return the integer number of calendar months between dates."""

        return (b.to_period("M") - a.to_period("M")).n

    changed = True
    while changed:
        changed = False

        for i in range(len(points) - 1):
            if months(points[i][0], points[i + 1][0]) < MIN_PHASE:
                a, b = points[i], points[i + 1]
                drop = a if (
                    (a[1] == "peak" and y[a[0]] < y[b[0]])
                    or (a[1] == "trough" and y[a[0]] > y[b[0]])
                    or (a[1] == b[1])
                ) else b
                points.remove(drop)
                points = alternate(points, y)
                changed = True
                break
        if changed:
            continue

        for i in range(len(points) - 2):
            if points[i][1] == points[i + 2][1] and months(
                points[i][0], points[i + 2][0]
            ) < MIN_CYCLE:
                points.remove(points[i + 1])
                points = alternate(points, y)
                changed = True
                break
    return points


def main() -> None:
    """Date turning points and write the complete chronology."""

    s = load_monthly("Worldwide")
    sa = s / seasonal_decompose(s, model="multiplicative", period=12).seasonal
    y = np.log(sa)

    pts = censor(alternate(candidates(y), y), y)
    tbl = pd.DataFrame(pts, columns=["month", "type"])
    tbl["month"] = tbl["month"].dt.strftime("%Y-%m")

    out = TABLES_DIR / "turning_points.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out, index=False)

    print(
        "Bry-Boschan turning points for log seasonally adjusted worldwide "
        "billings (K=5, minimum phase=6, minimum cycle=15):"
    )
    print(tbl.to_string(index=False))
    print()
    print("Window arithmetic for candidate origins (120m training, 33m horizon):")
    for _, row in tbl[tbl["type"] == "trough"].iterrows():
        t = pd.Timestamp(row["month"] + "-01")
        train_start = t - pd.DateOffset(months=TRAINING_MONTHS - 1)
        h_start = t + pd.DateOffset(months=1)
        h_end = t + pd.DateOffset(months=HORIZON_MONTHS)
        print(
            f"  trough {row['month']}: train {train_start:%Y-%m}..{t:%Y-%m}, "
            f"horizon {h_start:%Y-%m}..{h_end:%Y-%m}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
