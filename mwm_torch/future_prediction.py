from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "surgwmbench_benchmark").is_dir():
        sys.path.insert(0, str(parent))
        break

from surgwmbench_benchmark.future_prediction import main


if __name__ == "__main__":
    raise SystemExit(main("mwm", "MWMFuturePredictionAdapter", "mwm_torch.data.surgwmbench"))
