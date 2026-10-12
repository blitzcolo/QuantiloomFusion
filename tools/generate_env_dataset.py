#!/usr/bin/env python3
"""Thin entry point for quantiloom_fusion.datagen.cli; runs on a bare Python >= 3.11."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantiloom_fusion.datagen.cli import main

if __name__ == "__main__":
    sys.exit(main())
