"""
exercises/plank.py — plank hold coach.

A plank has no reps, so nothing here counts repetitions. What it counts
is *quality seconds*: wall-clock time during which the athlete is visible
and holding a straight line. Time spent sagging still adds to the clock
(you were still planking) but is scored down, and time spent out of frame
or collapsed doesn't count at all.

The set is chopped into ten-second blocks, each scored 0-100. That gives
the report the same rep-by-rep bar chart the other exercises get, and it
makes the shape of a hold visible at a glance — nearly everyone's plank
is a flat green line that turns amber in the last third.

Counting time rather than reps means the state machine is inverted
compared to push-ups and squats: stillness is success, and the motion
gate is used to detect *failure* (shaking) instead of movement.
"""

from __future__ import annotations

import statistics
from typing import List, Optional

from biomech import Keypoints, SideTracker, Stability, body_line, clamp
from exercises.base import ExerciseCoach, RepRecord, Tick, score_parts

SETUP = "SETUP"
HOLDING = "HOLDING"
BROKEN = "BROKEN"

BLOCK_SECONDS = 10.0


class PlankCoach(ExerciseCoach):
    KEY = "plank"
    NAME = "Plank"
    UNIT = "seconds"
    PRIMARY_LABEL = "body line"
    SECONDARY_LABEL = "stability"
    SETUP_HINT = "Set the phone side-on at hip height, far enough back that your head and feet are both in frame."
    CAMERA_VIEW = "side"

    smoothing_alpha = 0.4
    max_jump_deg = 25.0
    still_deg = 3.0
    calibration_seconds = 1.0

    SAG_OK = 12.0             # degrees below the line before it's a fault
    PIKE_OK = 16.0
    BREAK_DEG = 30.0          # past this, the hold has collapsed
    WOBBLE_OK = 4.5           # std-dev of the body line over ~2.5 s
    BREAK_GRACE_S = 1.5       # how long a collapse must last before it counts

    # The hold clock credits the whole real gap between two frames that
    # both show a good position — up to this much. The old cap was 0.5 s,
    # which silently made the timer run SLOW rather than pause: at one
    # frame per second (what a 700 ms-1 s round trip actually achieves)
    # every second of a real plank was credited as half a second, so a
    # 30 s hold displayed as 15. Anything beyond this cap is a gap we
    # genuinely did not observe, and unobserved time is not credited.
    MAX_GAP_CREDIT_S = 2.5

    def _reset_state(self, now):
        self.side_tracker = SideTracker(("shoulder", "hip", "ankle"))
        self.stability = Stability(window=25)
        self.stage = SETUP
        self.hold_s = 0.0
        self.quality_s = 0.0
        self._last_t: Optional[float] = None
        self._broke_at: Optional[float] = None
        self._block_start = 0.0
        self._block_samples: List[float] = []
        self._block_faults: List[str] = []
        self._sag_streak = 0
        self._pike_streak = 0
        self._wobble_streak = 0
        self._milestones: set = set()
        self._baseline = 180.0
        self.target_s = 0

    def on_calibrated(self, baseline: float):
        # Their honest "flat" — nobody is a perfect 180°, and holding them
        # to an ideal they can't hit just produces constant nagging.
        self._baseline = clamp(baseline, 160.0, 185.0)

    def set_target(self, seconds: int):
        self.target_s = max(0, int(seconds or 0))

    # ------------------------------------------------------------------
    def update(self, kp: Optional[Keypoints], now: float, detected: bool) -> Tick:
        side, conf = self.side_tracker.pick(kp if detected else None)
        line = body_line(kp, side, floor_ref="elbow") if side else None
        if line is not None and not line.reliable and side:
            line = body_line(kp, side, floor_ref="wrist")

        raw = line.angle if line else None
        # Only a FRESH reading is allowed to move the hold clock. The
        # smoother returns its previous value whenever a frame fails to
        # produce one, so `angle` on its own cannot tell "still holding"
        # apart from "the model lost you eight seconds ago" — and the
        # second case used to clock up plank seconds for an empty room.
        fresh = raw is not None
        angle = self.smoother.push(raw, now)
        self.motion.push(angle, now)
        wobble = self.stability.push(angle if fresh else None, now)
        deviation = line.deviation if line else None

        faults: List[str] = []
        cues: List[str] = []
        record = None

        if angle is None:
            self._last_t = None
            return Tick(count=int(self.hold_s), stage=self.stage,
                        is_calibrated=self.is_calibrated,
                        calibration_pct=self.calibration_pct,
                        hold_s=self.hold_s,
                        cues=("calibrating",) if not self.is_calibrated else ())

        # ---- calibration --------------------------------------------------
        if not self.is_calibrated:
            if self.motion.is_still and self._calibrate(angle, 150.0, 190.0, now):
                self.stage = HOLDING
                self._last_t = now
                self._block_start = now
            if not self.is_calibrated:
                return Tick(count=0, stage="CALIBRATING", is_calibrated=False,
                            calibration_pct=self.calibration_pct, primary=angle,
                            secondary=wobble, hold_s=0.0, cues=("calibrating",))

        # ---- accumulate the clock -----------------------------------------
        dt = 0.0
        if fresh:
            if self._last_t is not None:
                gap = max(0.0, now - self._last_t)
                # A gap inside the cap is real time the athlete spent in a
                # position we saw them enter and saw them still holding —
                # it counts in full, however sparse the frames were.
                # Beyond the cap we simply did not observe them, so we
                # credit nothing and resync from here.
                dt = gap if gap <= self.MAX_GAP_CREDIT_S else 0.0
            self._last_t = now

        # How far off the athlete's own baseline are they?
        off = (deviation if deviation is not None else abs(self._baseline - angle))
        sag = off if deviation is not None else 0.0
        collapsed = abs(off) > self.BREAK_DEG

        if collapsed:
            if self._broke_at is None:
                self._broke_at = now
            if now - self._broke_at >= self.BREAK_GRACE_S:
                if self.stage != BROKEN:
                    self.stage = BROKEN
                    cues.append("pl_broken")
        else:
            self._broke_at = None
            if self.stage == BROKEN:
                self.stage = HOLDING
            if self.stage == HOLDING:
                self.hold_s += dt
                frame_q = self._frame_quality(off, wobble)
                self.quality_s += dt * frame_q / 100.0
                self._block_samples.append(frame_q)

        # ---- live faults ---------------------------------------------------
        if deviation is not None and self.stage == HOLDING:
            if deviation > self.SAG_OK:
                faults.append("sag")
                self._sag_streak += 1
                self._pike_streak = 0
            elif deviation < -self.PIKE_OK:
                faults.append("pike")
                self._pike_streak += 1
                self._sag_streak = 0
            else:
                self._sag_streak = self._pike_streak = 0
        if wobble is not None and wobble > self.WOBBLE_OK and self.stage == HOLDING:
            faults.append("wobble")
            self._wobble_streak += 1
        else:
            self._wobble_streak = 0

        if self._sag_streak >= 6:
            cues.append("pl_sag")
            self._block_faults.append("sag")
            self._sag_streak = 0
        elif self._pike_streak >= 6:
            cues.append("pl_pike")
            self._block_faults.append("pike")
            self._pike_streak = 0
        elif self._wobble_streak >= 10:
            cues.append("tempo_cue")
            self._block_faults.append("wobble")
            self._wobble_streak = 0

        # ---- ten-second blocks ---------------------------------------------
        if self.stage == HOLDING and now - self._block_start >= BLOCK_SECONDS:
            record = self._close_block(now)

        # ---- milestones ------------------------------------------------------
        cues.extend(self._milestone_cues())

        return Tick(
            count=int(self.hold_s), stage=self.stage, is_calibrated=True,
            calibration_pct=1.0, primary=angle, secondary=wobble,
            faults=tuple(faults), cues=tuple(cues), record=record,
            quality=int(round(self._frame_quality(off, wobble))),
            hold_s=round(self.hold_s, 1),
            extra={"hip_deviation": deviation} if deviation is not None else {},
        )

    # ------------------------------------------------------------------
    def _frame_quality(self, off: float, wobble: Optional[float]) -> float:
        q = 100.0
        q -= max(0.0, abs(off) - 6.0) * 2.8
        if wobble is not None:
            q -= max(0.0, wobble - self.WOBBLE_OK) * 4.0
        return clamp(q, 0.0, 100.0)

    def _close_block(self, now: float) -> RepRecord:
        samples = self._block_samples or [0.0]
        score = int(round(statistics.mean(samples)))
        fault = None
        if self._block_faults:
            fault = max(set(self._block_faults), key=self._block_faults.count)
        self.count += 1
        rec = RepRecord(
            index=self.count, t=now - self.set_start,
            duration=round(now - self._block_start, 2),
            score=score, clean=fault is None and score >= 85, fault=fault,
            metrics={"worst": min(samples), "mean_line": statistics.mean(samples)},
        )
        self.records.append(rec)
        self._block_start = now
        self._block_samples = []
        self._block_faults = []
        return rec

    def _milestone_cues(self) -> List[str]:
        out = []
        held = int(self.hold_s)
        if self.target_s:
            half = self.target_s // 2
            if held >= half > 0 and "half" not in self._milestones:
                self._milestones.add("half")
                out.append("pl_half")
            if held >= self.target_s and "done" not in self._milestones:
                self._milestones.add("done")
                out.append("set_complete")
                return out
        for mark, cue in ((30, "pl_30"), (60, "pl_60")):
            if held >= mark and mark not in self._milestones:
                self._milestones.add(mark)
                out.append(cue)
        return out

    # ------------------------------------------------------------------
    def finalize(self, now: float):
        """Close the part-finished block so the last seconds still count."""
        if self.stage == HOLDING and self._block_samples and now > self._block_start + 2.0:
            self._close_block(now)

    def insights(self, recs: List[RepRecord], elapsed: float) -> List[str]:
        out = []
        held = self.hold_s
        quality_pct = round(100 * self.quality_s / held) if held > 0 else 0
        out.append(f"You held for {held:.0f} seconds, {quality_pct}% of it in a genuinely straight line.")
        faults = {}
        for r in recs:
            if r.fault:
                faults[r.fault] = faults.get(r.fault, 0) + 1
        if faults:
            top = max(faults, key=faults.get)
            out.append({
                "sag": "Your hips drifted toward the floor. That shifts the work off your abs and into your lower back — squeeze the glutes and tuck the pelvis slightly.",
                "pike": "Your hips rode up into a shallow pike, which makes the hold easier than it looks. Drop them until shoulders, hips and ankles line up.",
                "wobble": "There was a lot of shake in the hold. Shorter, steadier sets build more than long shaky ones.",
            }[top])
        scores = [r.score for r in recs]
        if len(scores) >= 3:
            a, b = statistics.mean(scores[:len(scores) // 2]), statistics.mean(scores[len(scores) // 2:])
            if a - b >= 8:
                fade = next((r.t for r in recs if r.score < a - 10), None)
                when = f" around the {fade:.0f} second mark" if fade else ""
                out.append(f"Quality started dropping{when} ({a:.0f} → {b:.0f} points). That's your real plank time — everything after it is practising a bad position.")
            else:
                out.append("Quality held steady from the first block to the last. Add ten seconds next time.")
        if held >= 60:
            out.append("Past a minute, a straight plank stops being an ab exercise and starts being an endurance one. Harder variations beat longer holds.")
        return out

    def summary(self, elapsed: float):
        s = super().summary(elapsed)
        s["count"] = int(self.hold_s)
        s["hold_s"] = round(self.hold_s, 1)
        s["quality_pct"] = round(100 * self.quality_s / self.hold_s) if self.hold_s > 0 else 0
        s["est_kcal"] = round(self.hold_s * 0.05, 1)
        if self.records:
            s["avg_tempo_s"] = None
            s["time_under_tension_s"] = round(self.hold_s, 1)
        return s

    def empty_insight(self) -> str:
        return "No hold was recorded. " + self.SETUP_HINT
