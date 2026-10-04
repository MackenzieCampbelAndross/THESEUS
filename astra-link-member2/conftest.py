"""Pytest path configuration for the Member 2 tracking subsystem.

Makes `pytest` work from this directory (or the repository root) without
requiring `pip install -e .` or a manual PYTHONPATH:
  - src/            -> provides the `astra_link_tracking` package
  - repository root -> provides `astra_link_integration` (used by the tracker tests)
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE / "src", _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
