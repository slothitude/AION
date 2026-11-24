#!/usr/bin/env python
"""Entry point script for running experiments."""

import sys
sys.path.insert(0, "src")

from nnfs.experiments.run_comparison import main

if __name__ == "__main__":
    main()