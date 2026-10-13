"""
exercises/pushup.py — side-view push-up coach.

Tracks elbow flexion through TOP → DESCENDING → BOTTOM → ASCENDING → TOP,
and judges each completed rep on four things: depth, body line, lockout
and tempo.

Notes on what changed from v2, because it matters for how it behaves:

* Depth, lockout and hip line are judged **per rep, from the whole rep**,
  never from a single frame. Judging single frames is why the old build
  said "lower your chest" at the top of every rep — the elbow is supposed
  to be straight up there.

* A rep that is abandoned near the top no longer produces a correction
  unless it actually travelled `min_partial_rom` degrees. Combined with
  the motion gate in base.py, holding a plank position now reliably
  produces silence rather than advice.

* Corrections are proposed for the *next* rep, not shouted about the last
  one, and they are only offered while the athlete is back at the top.
  Talking to someone mid-descent is both useless and late.
"""

from __future__ import annotations

import statistics
from typing import List, Optional

from biomech import Keypoints, SideTracker, body_line, clamp, joint_angle
from exercises.base import ExerciseCoach, RepRecord, Tick, score_parts

TOP = "TOP"
DESCENDING = "DESCENDING"
BOTTOM = "BOTTOM"
ASCENDING = "ASCENDING"


class PushUpCoach(ExerciseCoach):
    KEY = "pushup"
    NAME = "Push-Up"
    UNIT = "reps"
    PRIMARY_LABEL = "elbow"
    SECONDARY_LABEL = "body line"
    SETUP_HINT = "Set the phone side-on at hip height so your whole body is in frame."
    CAMERA_VIEW = "side"

    smoothing_alpha = 0.55
    still_deg = 5.0
    min_partial_rom = 25.0

    # form tolerances
    DEPTH_OK = 95.0          # elbow angle at the bottom
    LOCKOUT_OK = 148.0       # elbow angle at the top
    SAG_OK = 16.0            # degrees of hip drop
    PIKE_OK = 22.0
    MIN_ROM = 35.0
    REVERSAL_DEG = 8.0       # rise off the bottom that counts as turning around
    MIN_DURATION = 0.8

    def _reset_state(self, now):
        self.side_tracker = SideTracker(("shoulder", "elbow", "wrist", "hip"))
        self.stage = TOP
        self.top_angle = 165.0
        self.bottom_angle = 95.0
        self.descend_at = 150.0
        self.ascend_at = 107.0
        self.lockout_at = 158.0
        self._rep_min = None
        self._rep_max = None
        self._rep_start: Optional[float] = None
        self._max_sag = 0.0
        self._max_pike = 0.0
        self._sag_streak = 0
        self._pending_cue: Optional[str] = None

    def on_calibrated(self, baseline: float):
        """Derive every threshold from the athlete's own locked-out plank.

        People's 'straight arm' lands anywhere between about 150° and 178°
        depending on build and camera angle, so fixed thresholds either
        miss reps for one person or double-count for another.
        """
        self.top_angle = clamp(baseline, 145.0, 178.0)
        self.bottom_angle = clamp(self.top_angle - 62.0, 70.0, 102.0)
        self.lockout_at = self.top_angle - 8.0
        self.descend_at = self.top_angle - 18.0
        self.ascend_at = self.bottom_angle + 14.0

    # ------------------------------------------------------------------
    def update(self, kp: Optional[Keypoints], now: float, detected: bool) -> Tick:
        side, conf = self.side_tracker.pick(kp if detected else None)
        elbow = joint_angle(kp, side, "shoulder", "elbow", "wrist") if side else None
        angle = self.smoother.push(elbow, now)
        self.motion.push(angle, now)

        line = body_line(kp, side, floor_ref="wrist") if side else None
        deviation = line.deviation if line else None

        faults: List[str] = []
        cues: List[str] = []
        record = None

        if angle is None:
            return Tick(count=self.count, stage=self.stage,
                        is_calibrated=self.is_calibrated,
                        calibration_pct=self.calibration_pct,
                        secondary=line.angle if line else None,
                        cues=("calibrating",) if not self.is_calibrated else ())

        # ---- calibration ------------------------------------------------
        if not self.is_calibrated:
            steady = self.motion.is_still
            if steady and self._calibrate(angle, 140.0, 180.0, now):
                pass
            elif not steady and angle <= 135.0:
                # They started repping before we finished watching. Take the
                # defaults rather than making them hold a plank first.
                self.on_calibrated(165.0)
                self.is_calibrated = True
            if not self.is_calibrated:
                return Tick(count=self.count, stage="CALIBRATING", is_calibrated=False,
                            calibration_pct=self.calibration_pct, primary=angle,
                            secondary=line.angle if line else None, cues=("calibrating",))

        # ---- live hip line ----------------------------------------------
        if deviation is not None:
            self._max_sag = max(self._max_sag, deviation)
            self._max_pike = max(self._max_pike, -deviation)
            if deviation > self.SAG_OK:
                faults.append("hips_sag")
                self._sag_streak += 1
            elif deviation < -self.PIKE_OK:
                faults.append("hips_pike")
                self._sag_streak = 0
            else:
                self._sag_streak = 0

        # ---- state machine ----------------------------------------------
        # Nothing advances while the athlete is holding still: that is a
        # plank, not a rep, however much the pose estimate jitters.
        moving = self.motion.is_moving
        prev_stage = self.stage

        if self.stage == TOP:
            if moving and angle <= self.descend_at:
                self.stage = DESCENDING
                self._rep_start = now
                self._rep_min = angle
                # The rep's top is where they were a moment ago, not where
                # they are when the descent threshold trips — seeding from
                # the threshold under-measures every rep's range of motion.
                self._rep_max = max(self.motion.samples or [angle] + [self.top_angle])
                self._max_sag = max(0.0, deviation or 0.0)
                self._max_pike = max(0.0, -(deviation or 0.0))
        elif self.stage == DESCENDING:
            self._rep_min = min(self._rep_min, angle)
            travelled = (self._rep_max or angle) - self._rep_min
            if angle <= self.bottom_angle:
                self.stage = BOTTOM
            elif angle >= self._rep_min + self.REVERSAL_DEG and travelled >= self.min_partial_rom:
                # They turned around before reaching depth. That is still a
                # rep — a shallow one — and saying "that was shallow" is far
                # more useful than silently refusing to count it.
                self.stage = ASCENDING
            elif angle >= self.lockout_at:
                # came back up without reaching depth
                travelled = (self._rep_max or angle) - (self._rep_min or angle)
                self.stage = TOP
                if travelled >= self.min_partial_rom:
                    self._pending_cue = "pu_depth"
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
                        self._pending_cue = "pu_depth"
                    self._clear_rep()
            elif angle <= self.bottom_angle:
                self.stage = BOTTOM
            elif (angle <= self._rep_max - self.REVERSAL_DEG
                  and (self._rep_max - (self._rep_min or self._rep_max)) >= self.min_partial_rom):
                # Turned back down before ever reaching lockout — at fast
                # cadence (metcon pace) the smoothed signal can lag the real
                # motion enough that it never crosses `lockout_at` between
                # reps. Without this, the state machine parks in ASCENDING
                # forever: it doesn't have the ROM to bail out to TOP and
                # doesn't drop far enough to re-enter BOTTOM, so every rep
                # for the rest of the set silently stops counting. Close out
                # the rep in progress on whatever ROM it actually got, then
                # start the next descent from right here instead of losing
                # it.
                rom = (self._rep_max or 0.0) - (self._rep_min or 0.0)
                if rom >= self.MIN_ROM:
                    record = self._finish_rep(now)
                else:
                    if rom >= self.min_partial_rom:
                        self._pending_cue = "pu_depth"
                    self._clear_rep()
                self.stage = DESCENDING
                self._rep_start = now
                self._rep_min = angle
                self._rep_max = angle
                self._max_sag = max(0.0, deviation or 0.0)
                self._max_pike = max(0.0, -(deviation or 0.0))

        # ---- what is worth saying right now ------------------------------
        if record is not None:
            if record.clean:
                cues.append("rep_good")
            else:
                cues.append({
                    "depth": "pu_depth", "hips_sag": "pu_hip_sag",
                    "hips_pike": "pu_hip_pike", "lockout": "pu_lockout",
                    "tempo": "tempo_cue",
                }.get(record.fault, "pu_depth"))
        elif self.stage == TOP and self._pending_cue:
            cues.append(self._pending_cue)
            self._pending_cue = None

        # A sustained sag is a safety issue — worth interrupting for, but
        # only once it has persisted, not on one noisy frame.
        if self._sag_streak >= 5:
            cues.insert(0, "pu_hip_sag")
            self._sag_streak = 0

        live_q = None
        if self.is_calibrated and deviation is not None:
            live_q = int(round(clamp(100 - max(0.0, abs(deviation) - 8.0) * 3.0, 0, 100)))

        return Tick(
            count=self.count, stage=self.stage, is_calibrated=True,
            calibration_pct=1.0, primary=angle,
            secondary=line.angle if line else None,
            faults=tuple(faults), cues=tuple(cues), record=record, quality=live_q,
            extra={"hip_deviation": deviation} if deviation is not None else {},
        )

    # ------------------------------------------------------------------
    def _clear_rep(self):
        self._rep_min = self._rep_max = None
        self._rep_start = None
        self._max_sag = self._max_pike = 0.0

    def _finish_rep(self, now: float) -> RepRecord:
        min_e = self._rep_min if self._rep_min is not None else 90.0
        max_e = self._rep_max if self._rep_max is not None else self.top_angle
        duration = (now - self._rep_start) if self._rep_start else 0.0
        sag, pike = self._max_sag, self._max_pike

        parts = {
            "depth": clamp(100 - (min_e - 90.0) * 3.0, 35, 100),
            "align": clamp(100 - max(0.0, sag - 8.0) * 3.2 - max(0.0, pike - 12.0) * 2.2, 25, 100),
            "lockout": clamp(100 - (158.0 - max_e) * 3.0, 40, 100),
            "tempo": 60.0 if duration < self.MIN_DURATION else (80.0 if duration > 6.0 else 100.0),
        }
        score = score_parts(parts, {"depth": 0.40, "align": 0.35, "lockout": 0.15, "tempo": 0.10})

        fault = None
        if sag > self.SAG_OK:
            fault = "hips_sag"
        elif min_e > self.DEPTH_OK:
            fault = "depth"
        elif pike > self.PIKE_OK:
            fault = "hips_pike"
        elif max_e < self.LOCKOUT_OK:
            fault = "lockout"
        elif duration < self.MIN_DURATION:
            fault = "tempo"

        self.count += 1
        rec = RepRecord(
            index=self.count, t=now - self.set_start, duration=duration,
            score=score, clean=fault is None, fault=fault,
            metrics={"min_elbow": min_e, "max_elbow": max_e, "rom": max_e - min_e,
                     "max_sag": sag, "max_pike": pike},
        )
        self.records.append(rec)
        self._clear_rep()
        return rec

    # ------------------------------------------------------------------
    def insights(self, recs: List[RepRecord], elapsed: float) -> List[str]:
        out = []
        n = len(recs)
        scores = [r.score for r in recs]
        clean = sum(1 for r in recs if r.clean)
        faults = {}
        for r in recs:
            if r.fault:
                faults[r.fault] = faults.get(r.fault, 0) + 1
        if clean == n:
            out.append("Every rep was clean — that's a textbook set. Put the goal up next time.")
        elif faults:
            top = max(faults, key=faults.get)
            cnt = faults[top]
            out.append({
                "hips_sag": f"Your hips dropped on {cnt} rep(s). Brace your core and squeeze your glutes to hold one straight line from shoulder to ankle.",
                "depth": f"{cnt} rep(s) stopped short. Aim to bring your elbows to about 90° before you push back up.",
                "hips_pike": f"Your hips rode high on {cnt} rep(s). Think shoulders, hips and ankles all on one line.",
                "lockout": f"{cnt} rep(s) finished without a full lockout. Straighten the arms at the top — it's part of the rep.",
                "tempo": f"{cnt} rep(s) were rushed. A controlled one-to-two second descent is where the strength comes from.",
            }[top])
        if n >= 4:
            half = n // 2
            a, b = statistics.mean(scores[:half]), statistics.mean(scores[half:])
            if a - b >= 6:
                out.append(f"Form faded through the set ({a:.0f} → {b:.0f} points). That's fatigue — stopping a rep or two earlier keeps quality high.")
            elif b - a >= 4:
                out.append(f"You got sharper as you went ({a:.0f} → {b:.0f} points). Worth a longer warm-up next time.")
        durs = [r.duration for r in recs if r.duration > 0]
        if durs:
            avg = statistics.mean(durs)
            if avg < 1.0:
                out.append(f"Average rep took {avg:.1f}s — slow it down for more time under tension.")
            else:
                out.append(f"Average tempo {avg:.1f}s per rep, {sum(durs):.0f}s under tension in total.")
        best = max(recs, key=lambda r: r.score)
        out.append(f"Best rep: #{best.index} at {best.score} points, elbow reached {best.metrics['min_elbow']:.0f}°.")
        return out

    def summary(self, elapsed: float):
        s = super().summary(elapsed)
        if self.records:
            s["avg_depth_deg"] = round(statistics.mean(r.metrics["min_elbow"] for r in self.records), 1)
            durs = [r.duration for r in self.records if r.duration > 0]
            s["avg_tempo_s"] = round(statistics.mean(durs), 2) if durs else None
            s["time_under_tension_s"] = round(sum(durs), 1)
            s["est_kcal"] = round(len(self.records) * 0.45, 1)
        return s
