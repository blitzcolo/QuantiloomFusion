"""Cross-check helper: import modules from the sibling paper-exp checkout.

Adds the checkout to sys.path and stubs ``torch`` so
``paper_exp.baselines.adapter`` imports; both are undone on exit. Tests skip
when the checkout is absent — this is a test-time cross-check only, never a
runtime dependency.
"""

import sys
import types
from contextlib import contextmanager
from quantiloom_fusion.core.paths import native_path

PAPER_EXP = native_path("H:/paper-exp")


@contextmanager
def paper_exp_modules():
    """Yield with paper_exp importable; torch stub and sys.path restored after."""
    stub = "torch" not in sys.modules
    if stub:
        sys.modules["torch"] = types.ModuleType("torch")
    sys.path.insert(0, str(PAPER_EXP))
    try:
        yield
    finally:
        sys.path.remove(str(PAPER_EXP))
        if stub:
            sys.modules.pop("torch", None)
