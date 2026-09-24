"""Backward-compatible entry point for KB3-A fixed Optuna/TPE."""

from ..traditional_hpo.optuna_search import main


if __name__ == "__main__":
    main()

