import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_ROOT = next((parent for parent in _CURRENT_FILE.parents if (parent / 'utils').exists()), None)
if _ROOT and str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

