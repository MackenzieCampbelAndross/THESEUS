"""Pytest path configuration for the Member 3 security subsystem.

Makes `pytest` work from this directory without a manual PYTHONPATH:
  - this directory  -> provides the `src` package (src.identity, src.session, ...)
  - repository root -> provides `astra_link_integration`
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
