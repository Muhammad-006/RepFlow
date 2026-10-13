"""
exercises/squat.py — squat depth coach.

Primary signal is knee flexion (hip–knee–ankle). Supporting signal is
*forward fold*: how much further the hip has closed than the knee has.
Both are measured purely between the athlete's own segments, so nothing
here depends on which way is up in the image — a phone propped sideways
gives exactly the same numbers.

Forward fold, rather than "torso angle", is what separates a squat from a
good-morning. Someone with good ankle mobility squats more upright than
someone without, and both are correct; what is not correct is the hips
shooting back while the chest dives, and that is precisely a hip closing
faster than a knee.

Depth language follows how lifters actually talk:

    knee angle   ≈ 130°   quarter squat
    knee angle   ≈ 100°   thighs near parallel  ← what we ask for
    knee angle   ≤  85°   below parallel

Knee cave (valgus) is only judged when the camera can actually see it —
from a side-on view both knees project onto nearly the same point and any
"cave" measurement is noise, so from the side the coach simply stays
quiet about knees rather than inventing a fault.
"""

from __future__ import annotations

import statistics
from typing import List, Optional

from biomech import Keypoints, SideTracker, clamp, forward_fold, joint_angle, knee_valgus
from exercises.base import ExerciseCoach, RepRecord, Tick, score_parts

TOP = "TOP"
DESCENDING = "DESCENDING"
BOTTOM = "BOTTOM"
ASCENDING = "ASCENDING"


class SquatCoach(ExerciseCoach):
    KEY = "squat"
    NAME = "Squat"
    UNIT = "reps"
    PRIMARY_LABEL = "knee"
    SECONDARY_LABEL = "fold"
    SETUP_HINT = "Stand side-on or at 45° to the phone, about 2 m back, with your whole body in frame."
    CAMERA_VIEW = "either"

    smoothing_alpha = 0.5
    still_deg = 6.0
    min_partial_rom = 25.0

    DEPTH_OK = 100.0          # knee angle at the bottom
    DEPTH_GREAT = 85.0
    STAND_OK = 160.0          # knee angle at the top
    LEAN_OK = 22.0            # degrees of fold beyond the knee bend
    VALGUS_OK = 0.22          # knee width lost, as a fraction
    MIN_ROM = 40.0
    REVERSAL_DEG = 9.0       # rise off the bottom that counts as standing back up
    MIN_DURATION = 1.0

    def _reset_state(self, now):
        self.side_tracker = SideTracker(("hip", "knee", "ankle"))
        self.stage = TOP
        self.stand_angle = 172.0
        self.bottom_angle = 100.0
        self.descend_at = 155.0
        self.ascend_at = 115.0
        self.lockout_at = 162.0
        self._rep_min = None
        self._rep_max = None
        self._rep_start: Optional[float] = None
        self._max_lean = 0.0
        self._max_valgus = 0.0
        self._lean_streak = 0
        self._valgus_streak = 0
        self._pending_cue: Optional[str] = None

    def on_calibrated(self, baseline: float):
        self.stand_angle = clamp(baseline, 155.0, 180.0)
        self.lockout_at = self.stand_angle - 12.0
        self.descend_at = self.stand_angle - 20.0
        self.bottom_angle = self.DEPTH_OK
        self.ascend_at = self.bottom_angle + 18.0

    # ------------------------------------------------------------------
    def update(self, kp: Optional[Keypoints], now: float, detected: bool) -> Tick:
        side, conf = self.side_tracker.pick(kp if detected else None)
        knee = joint_angle(kp, side, "hip", "knee", "ankle") if side else None
        angle = self.smoother.push(knee, now)
        self.motion.push(angle, now)

        lean = forward_fold(kp, side) if side else None
        valgus = knee_valgus(kp) if kp else None

        faults: List[str] = []
        cues: List[str] = []
        record = None

        if angle is None:
            return Tick(count=self.count, stage=self.stage,
                        is_calibrated=self.is_calibrated,
                        calibration_pct=self.calibration_pct,
                        cues=("calibrating",) if not self.is_calibrated else ())

        # ---- calibration: a still, standing frame ------------------------
        if not self.is_calibrated:
            if self.motion.is_still and self._calibrate(angle, 155.0, 180.0, now):
                pass
            elif self.motion.is_moving and angle <= 140.0:
                self.on_calibrated(172.0)
                self.is_calibrated = True
            if not self.is_calibrated:
                return Tick(count=self.count, stage="CALIBRATING", is_calibrated=False,
                            calibration_pct=self.calibration_pct, primary=angle,
                            secondary=lean, cues=("calibrating",))

        # ---- live faults --------------------------------------------------
        if lean is not None:
            self._max_lean = max(self._max_lean, lean)
            if lean > self.LEAN_OK and self.stage in (DESCENDING, BOTTOM, ASCENDING):
                faults.append("torso")
                self._lean_streak += 1
            else:
                self._lean_streak = 0
        if valgus is not None:
            self._max_valgus = max(self._max_valgus, valgus)
            if valgus > self.VALGUS_OK and self.stage in (DESCENDING, BOTTOM, ASCENDING):
                faults.append("valgus")
                self._valgus_streak += 1
            else:
                self._valgus_streak = 0

        # ---- state machine ------------------------------------------------
        moving = self.motion.is_moving

        if self.stage == TOP:
            if moving and angle <= self.descend_at:
                self.stage = DESCENDING
                self._rep_start = now
                self._rep_min = angle
                # The rep's top is where they were a moment ago, not where
                # they are when the descent threshold trips — seeding from
                # the threshold under-measures every rep's range of motion.
                self._rep_max = max(self.motion.samples or [angle] + [self.stand_angle])
                self._max_lean = lean or 0.0
                self._max_valgus = valgus or 0.0
        elif self.stage == DESCENDING:
            self._rep_min = min(self._rep_min, angle)
            travelled = (self._rep_max or angle) - self._rep_min
            if angle <= self.bottom_angle:
                self.stage = BOTTOM
            elif angle >= self._rep_min + self.REVERSAL_DEG and travelled >= self.min_partial_rom:
                self.stage = ASCENDING
            elif angle >= self.lockout_at:
                travelled = (self._rep_max or angle) - (self._rep_min or angle)
                self.stage = TOP
                if travelled >= self.min_partial_rom:
                    self._pending_cue = "sq_depth"
                self._clear_rep()
        elif self.stage == BOTTOM:
            self._rep_min = min(self._rep_min, angle)
            if angle >= self.ascend_at:
                self.stage = ASCENDING
        elif self.stage == ASCENDING:
            self._rep_max = max(self._rep_max or angle, angle)
            if angle >= self.lockout_at:
                rom = (self._rep_max or 0.0) - (self._rep_min or 0.0)
                self.stage = TOP
                if rom >= self.MIN_ROM:
                    record = self._finish_rep(now)
                else:
                    if rom >= self.min_partial_rom:
                        self._pending_cue = "sq_depth"
                    self._clear_rep()
            elif angle <= self.bottom_angle:
                self.stage = BOTTOM
            elif (angle <= self._rep_max - self.REVERSAL_DEG
                  and (self._rep_max - (self._rep_min or self._rep_max)) >= self.min_partial_rom):
                # Symmetric with the DESCENDING->ASCENDING reversal below: at
                # fast cadence the smoothed knee angle can turn back down
                # before ever reaching `lockout_at`. Without this branch
                # neither ASCENDING exit condition ever fires again (it
                # doesn't have the ROM to bail to TOP, and doesn't drop back
                # to `bottom_angle` either), so the state machine parks here
                # and stops counting for the rest of the set. Close out the
                # rep on the ROM it actually reached and start the next
                # descent from here.
                rom = (self._rep_max or 0.0) - (self._rep_min or 0.0)
                if rom >= self.MIN_ROM:
                    record = self._finish_rep(now)
                else:
                    if rom >= self.min_partial_rom:
                        self._pending_cue = "sq_depth"
                    self._clear_rep()
                self.stage = DESCENDING
                self._rep_start = now
                self._rep_min = angle
                self._rep_max = angle
                self._max_lean = lean or 0.0
                self._max_valgus = valgus or 0.0

        # ---- cue selection -------------------------------------------------
        if record is not None:
            if record.clean:
                cues.append("rep_good")
            else:
                cues.append({
                    "depth": "sq_depth", "torso": "sq_chest", "valgus": "sq_knees",
                    "stand": "sq_stand", "tempo": "tempo_cue",
                }.get(record.fault, "sq_depth"))
        elif self.stage == TOP and self._pending_cue:
            cues.append(self._pending_cue)
            self._pending_cue = None

        # Knees caving is the one thing worth interrupting a rep for.
        if self._valgus_streak >= 4:
            cues.insert(0, "sq_knees")
            self._valgus_streak = 0
        elif self._lean_streak >= 6:
            cues.insert(0, "sq_chest")
            self._lean_streak = 0

        live_q = None
        if self.stage in (DESCENDING, BOTTOM):
            live_q = int(round(clamp(100 - max(0.0, angle - self.DEPTH_GREAT) * 1.6, 0, 100)))

        return Tick(
            count=self.count, stage=self.stage, is_calibrated=True, calibration_pct=1.0,
            primary=angle, secondary=lean, faults=tuple(faults), cues=tuple(cues),
            record=record, quality=live_q,
            extra={k: v for k, v in (("forward_fold", lean), ("valgus", valgus)) if v is not None},
        )

    # ------------------------------------------------------------------
    def _clear_rep(self):
        self._rep_min = self._rep_max = None
        self._rep_start = None
        self._max_lean = self._max_valgus = 0.0

    def _finish_rep(self, now: float) -> RepRecord:
        min_k = self._rep_min if self._rep_min is not None else 110.0
        max_k = self._rep_max if self._rep_max is not None else self.stand_angle
        duration = (now - self._rep_start) if self._rep_start else 0.0
        lean, valgus = self._max_lean, self._max_valgus

        parts = {
            "depth": clamp(100 - (min_k - self.DEPTH_GREAT) * 2.2, 30, 100),
            "torso": clamp(100 - max(0.0, lean - 10.0) * 3.4, 30, 100),
            "knees": clamp(100 - max(0.0, valgus - 0.10) * 260.0, 30, 100),
            "stand": clamp(100 - (self.STAND_OK - max_k) * 3.0, 40, 100),
            "tempo": 65.0 if duration < self.MIN_DURATION else (85.0 if duration > 7.0 else 100.0),
        }
        score = score_parts(parts, {"depth": 0.40, "torso": 0.22, "knees": 0.18,
                                    "stand": 0.12, "tempo": 0.08})

        fault = None
        if min_k > self.DEPTH_OK:
            fault = "depth"
        elif valgus > self.VALGUS_OK:
            fault = "valgus"
        elif lean > self.LEAN_OK:
            fault = "torso"
        elif max_k < self.STAND_OK:
            fault = "stand"
        elif duration < self.MIN_DURATION:
            fault = "tempo"

        self.count += 1
        rec = RepRecord(
            index=self.count, t=now - self.set_start, duration=duration,
            score=score, clean=fault is None, fault=fault,
            metrics={"min_knee": min_k, "max_knee": max_k, "rom": max_k - min_k,
                     "max_fold": lean, "max_valgus": valgus * 100.0},
        )
        self.records.append(rec)
        self._clear_rep()
        return rec

    # ------------------------------------------------------------------
    def insights(self, recs: List[RepRecord], elapsed: float) -> List[str]:
        out = []
        n = len(recs)
        depths = [r.metrics["min_knee"] for r in recs]
        avg_depth = statistics.mean(depths)
        clean = sum(1 for r in recs if r.clean)
        faults = {}
        for r in recs:
            if r.fault:
                faults[r.fault] = faults.get(r.fault, 0) + 1

        below = sum(1 for d in depths if d <= self.DEPTH_GREAT)
        out.append(f"Average depth {avg_depth:.0f}° at the knee — {below} of {n} rep(s) went below parallel.")
        if clean == n:
            out.append("Depth, chest position and lockout all held up across the whole set. Add load or reps.")
        elif faults:
            top = max(faults, key=faults.get)
            cnt = faults[top]
            out.append({
                "depth": f"{cnt} rep(s) stopped above parallel. Sit back and down until the thigh is level — ankle mobility usually limits this before strength does.",
                "valgus": f"Your knees tracked inward on {cnt} rep(s). Drive them out over your toes on the way up; it's the single biggest knee-health cue in the squat.",
                "torso": f"Your chest dropped forward on {cnt} rep(s). Keep the torso angle matching your shin angle so the bar path stays over midfoot.",
                "stand": f"{cnt} rep(s) didn't finish standing. Lock the hips out at the top before starting the next one.",
                "tempo": f"{cnt} rep(s) were dropped rather than lowered. Take a full second on the way down.",
            }[top])
        if n >= 4:
            half = n // 2
            a, b = statistics.mean(depths[:half]), statistics.mean(depths[half:])
            if b - a >= 8:
                out.append(f"Depth got shallower as you fatigued ({a:.0f}° → {b:.0f}°). That's the honest place to end the set.")
        durs = [r.duration for r in recs if r.duration > 0]
        if durs:
            out.append(f"Average tempo {statistics.mean(durs):.1f}s per rep.")
        best = max(recs, key=lambda r: r.score)
        out.append(f"Best rep: #{best.index} at {best.score} points, knee reached {best.metrics['min_knee']:.0f}°.")
        return out

    def summary(self, elapsed: float):
        s = super().summary(elapsed)
        if self.records:
            s["avg_depth_deg"] = round(statistics.mean(r.metrics["min_knee"] for r in self.records), 1)
            durs = [r.duration for r in self.records if r.duration > 0]
            s["avg_tempo_s"] = round(statistics.mean(durs), 2) if durs else None
            s["time_under_tension_s"] = round(sum(durs), 1)
            s["est_kcal"] = round(len(self.records) * 0.6, 1)
        return s
