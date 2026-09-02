"""Select, fit, and forecast the AI-era fixed-origin model."""

from wsts_cycle.workflows import run_anchor_analysis


def main() -> None:
    """Write the AI-era selection, fit, forecast, diagnostics, and figure."""

    run_anchor_analysis("ai")


if __name__ == "__main__":
    main()
