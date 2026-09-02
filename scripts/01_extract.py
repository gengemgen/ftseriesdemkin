"""Extract the WSTS source workbook."""

import pandas as pd

from wsts_cycle.data import extract, write_csvs
from wsts_cycle.paths import PROCESSED_DATA_DIR


def main() -> None:
    """Extract all series and check the source data."""

    frames = extract()
    write_csvs(frames)
    monthly, mma = frames["monthly"], frames["3mma"]

    def check(name: str, ok: bool, detail: str) -> None:
        """Stop when a source-data invariant fails."""

        assert ok, f"{name}: {detail}"

    ww = monthly["Worldwide"].dropna()
    full = pd.date_range(ww.index.min(), ww.index.max(), freq="MS")
    check(
        "coverage",
        ww.index.min() == pd.Timestamp("1986-01-01")
        and ww.index.max() == pd.Timestamp("2026-05-01")
        and len(ww) == 485
        and ww.index.equals(full),
        f"{ww.index.min():%Y-%m} .. {ww.index.max():%Y-%m}, "
        f"n={len(ww)}, gap-free",
    )

    regions_sum = monthly[
        ["Americas", "Europe", "Japan", "Asia Pacific"]
    ].sum(axis=1)
    add_dev = (regions_sum - monthly["Worldwide"]).abs().max()
    check(
        "regional additivity",
        add_dev <= 2.0,
        f"max |sum(regions) - Worldwide| = {add_dev:.6f} thousand USD",
    )

    trailing = monthly.rolling(3).mean()
    common = mma.index.intersection(trailing.dropna().index)
    dev = (mma.loc[common] - trailing.loc[common]).abs()
    check(
        "3MMA identity (from 1986-03)",
        float(dev.max().max()) < 1e-6,
        f"n={len(common)} months x {dev.shape[1]} regions, "
        f"max |3MMA - trailing mean| = {float(dev.max().max()):.3e}",
    )

    print(f"wrote extracted WSTS tables to {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
