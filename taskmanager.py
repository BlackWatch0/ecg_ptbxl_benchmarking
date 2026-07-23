#!/usr/bin/env python3

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from task_manager.cli import main


if __name__ == "__main__":
    sys.exit(main())
