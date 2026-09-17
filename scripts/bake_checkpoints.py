"""Record the guided demo's jump points ahead of time.

    python scripts/bake_checkpoints.py

**You usually do not need this.** The server records them itself in the background the
first time it starts, and every beat becomes jumpable as soon as it is recorded. Run this
instead when you want the whole recording finished *before* starting the server -- say,
the night before presenting on a fresh clone -- so Prev/Next can reach every beat from
the first click.

Takes about eight minutes, resumes from wherever an interrupted run got to, and does
nothing if the recording is already current. `--force` discards it and starts again.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twinsync.checkpoints import main                             # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
