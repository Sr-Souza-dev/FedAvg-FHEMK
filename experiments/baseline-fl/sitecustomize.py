import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

try:
    from utils.flwr_inline_backend import register_inline_backend
except Exception:
    pass
else:
    register_inline_backend()

