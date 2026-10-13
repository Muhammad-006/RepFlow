"""
feedback.py — decides whether the coach is allowed to speak right now.

Coaches propose; this throttler disposes. At ten frames a second an
un-throttled coach produces ten opinions a second, which on a phone
speaker is just noise.

Three things changed here after testing v2 in a real set:

1. **The gap now depends on how long the last clip actually was.** v2 used
   a flat 2.2 second minimum while several clips ran close to 3 seconds,
   so the coach talked over itself and every cue sounded late.

2. **Stale advice is dropped, not queued.** A cue decided from a frame
   that is already 600 ms old describes a body position the athlete has
   left. Speaking it is worse than silence — that is exactly how "lower
   your chest" arrives while you're already back at the top.

3. **A short cue can interrupt a long one** when it's a safety call.
   Everything else waits its turn.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from cues import CUES, CuePriority, duration_of

# How long the same correction stays muted after it has been given.
PER_CUE_COOLDOWN_S: Dict[str, float] = {
    "calibrating":   6.0,
    "lost_tracking": 7.0,
    "rep_good":      4.5,
    "tempo_cue":     8.0,
    "pu_depth":      5.0,
    "pu_hip_sag":    4.5,
    "pu_hip_pike":   6.0,
    "pu_lockout":    6.0,
    "sq_depth":      5.0,
    "sq_chest":      6.0,
    "sq_knees":      4.5,
    "sq_stand":      6.0,
    "pl_sag":        5.0,
    "pl_pike":       6.0,
    "pl_good":      12.0,
    "pl_broken":     5.0,
}

# Cues that may cut in over a clip that is still playing.
INTERRUPTIBLE_BY = {"pu_hip_sag", "sq_knees", "pl_broken", "lost_tracking"}

# A cue decided from a frame older than this is thrown away.
MAX_DECISION_AGE_S = 0.45


class FeedbackEngine:
    def __init__(self, min_gap_s: float = 0.6, pad_s: float = 0.35,
                 interrupt_gap_s: float = 0.25):
        self.min_gap = min_gap_s
        self.interrupt_gap = interrupt_gap_s
        self.pad = pad_s
        self.last_cue_id: Optional[str] = None
        self.last_cue_at = 0.0
        self.last_cue_ends = 0.0
        self._per_cue: Dict[str, float] = {}
        self.spoken = 0
        self.suppressed = 0

    def reset(self):
        self.last_cue_id = None
        self.last_cue_at = 0.0
        self.last_cue_ends = 0.0
        self._per_cue.clear()
        self.spoken = 0
        self.suppressed = 0

    # ------------------------------------------------------------------
    def select(self, candidates, decision_age_s: float = 0.0,
               force: bool = False, now: Optional[float] = None) -> Optional[str]:
        """Pick at most one cue from a ranked list of candidates."""
        now = now if now is not None else time.time()
        if decision_age_s > MAX_DECISION_AGE_S and not force:
            self.suppressed += len(candidates or ())
            return None
        for cue_id in candidates or ():
            if cue_id not in CUES:
                continue
            if self._allowed(cue_id, now, force):
                self._commit(cue_id, now)
                return cue_id
            self.suppressed += 1
        return None

    # ------------------------------------------------------------------
    def _allowed(self, cue_id: str, now: float, force: bool) -> bool:
        if force:
            return True
        priority = CUES[cue_id][1]
        # A safety call — sagging hips, caving knees, a collapsed plank —
        # is allowed to cut in over a clip that is still playing, because
        # waiting politely for "hold still, I'm calibrating" to finish is
        # how someone hurts their back.
        may_interrupt = priority == CuePriority.HIGH and cue_id in INTERRUPTIBLE_BY
        if now < self.last_cue_ends and not may_interrupt:
            return False
        gap = self.interrupt_gap if may_interrupt else self.min_gap
        if now - self.last_cue_at < gap:
            return False
        if now - self._per_cue.get(cue_id, 0.0) < PER_CUE_COOLDOWN_S.get(cue_id, 3.0):
            return False
        return True

    def _commit(self, cue_id: str, now: float):
        self.last_cue_id = cue_id
        self.last_cue_at = now
        self.last_cue_ends = now + duration_of(cue_id) + self.pad
        self._per_cue[cue_id] = now
        self.spoken += 1

    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict:
        return {"spoken": self.spoken, "suppressed": self.suppressed}
