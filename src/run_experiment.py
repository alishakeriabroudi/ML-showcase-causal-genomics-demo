"""Compatibility entrypoint.

This repository uses `ml_showcase` as the main package. This wrapper exists to
support older commands like: `python -m src.run_experiment`.
"""

from __future__ import annotations

from ml_showcase.run_experiment import main

if __name__ == "__main__":
    main()
