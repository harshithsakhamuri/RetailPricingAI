import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import lightgbm  # noqa: E402,F401  must load before pandas on Windows (see pricing_profit_optimizer.py)
