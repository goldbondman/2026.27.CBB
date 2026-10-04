from __future__ import annotations

import gate_batch_10 as base
import gate_batch_10_v3 as v3

# v3 import installs hardened team-game/schema helpers and P1 feature construction.
ORIG_RUN_FROZEN = base.run_frozen
EXACT = v3.exact_frozen_continuity()


def run_frozen_exact(g, continuity_ignored, mode="m20", preseason_adjust=None):
    return ORIG_RUN_FROZEN(g, EXACT, mode=mode, preseason_adjust=preseason_adjust)


base.run_frozen = run_frozen_exact
base.main()
