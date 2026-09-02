"""Select, fit, and forecast the dot-com fixed-origin model."""

from wsts_cycle.workflows import run_anchor_analysis


def main() -> None:
    """Write the dot-com selection, fit, forecast, diagnostics, and figure."""

    run_anchor_analysis("dotcom")


if __name__ == "__main__":
    main()
