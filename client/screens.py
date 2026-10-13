"""
screens.py — Setup, Live coaching and Report.
(Boot and Home live in main.py; Privacy lives in privacy.py.)

The live screen is where most of the latency work landed:

* Every frame carries the moment it was captured, inside the encrypted
  payload. The server drops anything that arrives too late to coach from,
  rather than answering about a body position you already left.

* Only one frame is ever in flight. Frames that arrive from the camera
  while one is out are dropped on the phone, not queued — a queue would
  only convert dropped frames into late ones.

* Results carry the age of the frame they came from. Anything older than
  the one already shown is discarded, so the skeleton and the counter
  never jump backwards.

* The coach speaks about the *next* rep. Corrections land at the top of
  the movement, between reps, where they're actually actionable.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import glob
import os
import time

import flet as ft
import flet.canvas as cv

import config
import exercises as ex
from cues import CUES, text_of, tone_of
from spotter_client import SecurityError, SpotterClientError
from ui_kit import (AMBER, BORDER, BORDER_2, CYAN, DIM, GREEN, GREEN_GLOW, H_FONT, INK, M_FONT,
                    MUTED, RED, S_FONT, SURFACE, SURFACE_2, TEXT, VIOLET, H, T, ahex, card, dot,
                    eyebrow, ghost_button, glow, icon_badge, mono, neon_button, pill,
                    section_title, stat_tile)

try:
    import flet_camera as fc
except Exception:  # pragma: no cover
    fc = None
try:
    import flet_permission_handler as fph
except Exception:  # pragma: no cover
    fph = None

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
ARM_EDGES = {(5, 7), (7, 9), (6, 8), (8, 10)}
LEG_EDGES = {(11, 13), (13, 15), (12, 14), (14, 16)}

STAGE_LABELS = {
    "reps": [("TOP", "TOP"), ("DESCENDING", "DOWN"), ("BOTTOM", "BOTTOM"), ("ASCENDING", "UP")],
    "seconds": [("SETUP", "SET"), ("HOLDING", "HOLD"), ("BROKEN", "RESET")],
}

FAULT_LABEL = {
    "hips_sag": "Hip sag", "depth": "Shallow", "hips_pike": "Hips high",
    "lockout": "No lockout", "tempo": "Rushed", "torso": "Chest dropped",
    "valgus": "Knees in", "stand": "No lockout", "sag": "Hips low", "pike": "Hips high",
    "wobble": "Unsteady",
}

# Don't send faster than this, however fast the camera is. Ten frames a
# second is plenty for human movement and leaves the phone's battery and
# the server's CPU alone.
MIN_SEND_INTERVAL_S = 0.09

# What the server's `detect_reason` means to someone holding a phone.
# Every one of these is an instruction the athlete can act on without
# getting up, because the alternative — a frozen screen — is what the
# app used to do.
DETECT_HELP = {
    "nobody_in_frame": "Nobody in frame — move into the camera's view",
    "too_far_or_dark": "Too far away or too dark — move closer or add light",
    "partial_body": "Only part of you is in frame — step back a little",
    "no_pose": "Coach is still warming up…",
}

# If this many frames in a row come back stale, the phone and the laptop
# are not agreeing about time or the link is congested. Either way the
# athlete deserves to be told rather than watch a dead screen.
STALE_WARN_AFTER = 15


def _test_frames() -> list:
    """Desktop only: REPFLOW_TEST_FRAMES=<folder|file.jpg> replaces the camera."""
    src = os.environ.get("REPFLOW_TEST_FRAMES", "")
    if not src:
        return []
    files = sorted(glob.glob(os.path.join(src, "*.jp*g"))) if os.path.isdir(src) else \
        ([src] if os.path.isfile(src) else [])
    frames = []
    for f in files[:600]:
        with open(f, "rb") as fh:
            frames.append(fh.read())
    return frames


def _jpeg_size(data: bytes):
    i = 2
    try:
        while i < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                h = int.from_bytes(data[i + 5:i + 7], "big")
                w = int.from_bytes(data[i + 7:i + 9], "big")
                return w, h
            i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    except Exception:
        pass
    return None


def fmt_time(seconds: float) -> str:
    s = max(0, int(seconds or 0))
    return f"{s // 60:02d}:{s % 60:02d}"


# =========================================================================
def _screens(page, state, client, conn, ui, navigate, view, link_chip, listen, play, haptic,
             services, is_mobile, grade_badge, mini_bars, on_ui):

    def top_bar(title, back_to="/home", trailing=None):
        return ft.Row([
            ft.Container(ft.Icon(ft.Icons.ARROW_BACK_ROUNDED, color=TEXT, size=20), width=40,
                         height=40, border_radius=12, bgcolor=SURFACE_2,
                         border=ft.Border.all(1, BORDER), alignment=ft.Alignment.CENTER,
                         on_click=lambda e: navigate(back_to), ink=True),
            H(title, size=19),
            trailing or ft.Container(width=40),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

    # =====================================================================
    # SETUP
    # =====================================================================
    def build_setup() -> ft.View:
        pres = state.presentation()
        unit = pres.get("unit", "reps")

        # --- exercise switcher -------------------------------------------
        switch_row = ft.Row(spacing=8)

        def render_switch():
            switch_row.controls = []
            for entry in state.catalog:
                sel = entry["key"] == state.exercise
                switch_row.controls.append(ft.Container(
                    ft.Column([ft.Icon(entry.get("icon", ft.Icons.SPORTS_GYMNASTICS),
                                       color=INK if sel else MUTED, size=19),
                               T(entry["name"], size=11, font=S_FONT,
                                 color=INK if sel else MUTED)],
                              spacing=4, tight=True,
                              horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    expand=True, height=62, alignment=ft.Alignment.CENTER, border_radius=14,
                    bgcolor=GREEN if sel else SURFACE_2,
                    border=ft.Border.all(1, GREEN if sel else BORDER_2),
                    shadow=glow(GREEN, 16, opacity="44") if sel else None,
                    on_click=lambda e, k=entry["key"]: pick_exercise(k), ink=True))

        def pick_exercise(key):
            state.exercise = key
            p = ex.presentation(key)
            for entry in state.catalog:
                if entry["key"] == key:
                    p = entry
            state.target = p["goals"][1][0] if len(p["goals"]) > 1 else 0
            navigate("/setup")

        render_switch()

        # --- goal picker ---------------------------------------------------
        goal_row = ft.Row(spacing=8)

        def render_goals():
            goal_row.controls = []
            for val, label in pres["goals"]:
                sel = state.target == val
                goal_row.controls.append(ft.Container(
                    T(label, size=16, font=H_FONT, color=INK if sel else TEXT), expand=True,
                    height=50, alignment=ft.Alignment.CENTER, border_radius=14,
                    bgcolor=GREEN if sel else SURFACE_2,
                    border=ft.Border.all(1, GREEN if sel else BORDER_2),
                    shadow=glow(GREEN, 16, opacity="44") if sel else None,
                    on_click=lambda e, v=val: pick_goal(v), ink=True))

        def pick_goal(v):
            state.target = v
            render_goals()
            page.update()

        render_goals()

        voice_switch = ft.Switch(value=state.voice, active_color=GREEN,
                                 on_change=lambda e: setattr(state, "voice", e.control.value))

        # --- placement checklist ---------------------------------------------
        checklist = ft.Column([
            ft.Row([ft.Icon(icon, color=GREEN, size=17),
                    T(text, size=12.5, color="#C9D6EA", expand=True)], spacing=10)
            for icon, text in pres["setup"]
        ], spacing=10)

        # --- link + safety words (compact) -------------------------------------
        chip, chip_refresh = link_chip()
        words = mono(conn.safety_words or "—", size=12, color=CYAN)

        def refresh():
            chip_refresh()
            words.value = conn.safety_words or "—"

        refresh()
        listen(refresh)

        link_card = card(ft.Row([
            ft.Column([T("Encrypted link", size=13, font=S_FONT),
                       ft.Row([T("verified as", size=11, color=MUTED), words], spacing=6)],
                      spacing=3, tight=True, expand=True),
            chip,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
            padding=14, radius=16, on_click=lambda e: navigate("/privacy"))

        controls = [
            top_bar("Session setup"),
            card(ft.Column([eyebrow("movement"), switch_row], spacing=12), padding=16),
            card(ft.Column([
                ft.Row([eyebrow("camera placement"),
                        T("STEP 1", size=10, color=DIM, font=S_FONT, spacing=1.5)],
                       alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(placement_diagram(pres.get("camera_view", "side")), height=140,
                             border_radius=12, bgcolor="#060A14",
                             border=ft.Border.all(1, BORDER),
                             padding=ft.Padding.symmetric(horizontal=6)),
                checklist,
            ], spacing=12), padding=16),
            card(ft.Column([
                ft.Row([eyebrow(pres["goal_label"]),
                        T("STEP 2", size=10, color=DIM, font=S_FONT, spacing=1.5)],
                       alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                goal_row,
                ft.Divider(height=1, color=BORDER),
                ft.Row([ft.Row([ft.Icon(ft.Icons.RECORD_VOICE_OVER_OUTLINED, color=GREEN, size=18),
                                T("Voice coaching", size=14, font=S_FONT)], spacing=10),
                        voice_switch],
                       alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ], spacing=12), padding=16),
            link_card,
            neon_button("BEGIN", ft.Icons.PLAY_ARROW_ROUNDED, on_click=lambda e: navigate("/live")),
            ft.Container(height=6),
        ]
        return view("/setup", ft.SafeArea(ft.Container(
            ft.Column(controls, spacing=14, scroll=ft.ScrollMode.AUTO, expand=True),
            padding=ft.Padding.only(left=18, right=18, top=10), expand=True), expand=True))

    def placement_diagram(camera_view: str):
        """A drawn 'where to put the phone' sketch, sized to its container."""
        canvas = cv.Canvas(shapes=[], height=140, expand=True)

        def draw(w, h=140):
            floor = h - 20
            sx = w * 0.14
            px0, px1 = w * 0.42, w * 0.92
            thin = ft.Paint(color=BORDER_2, stroke_width=1.5, style=ft.PaintingStyle.STROKE)
            body = ft.Paint(color=CYAN, stroke_width=4, stroke_cap=ft.StrokeCap.ROUND,
                            style=ft.PaintingStyle.STROKE)
            joints = ft.Paint(color=GREEN, style=ft.PaintingStyle.FILL)
            fov = ft.Paint(color=ahex(GREEN, "55"), stroke_width=1.2,
                           stroke_dash_pattern=[5, 4], style=ft.PaintingStyle.STROKE)
            shapes = [
                cv.Line(0, floor + 2, w, floor + 2, thin),
                cv.Rect(sx - 8, floor - 66, 16, 28, border_radius=3,
                        paint=ft.Paint(color=TEXT, stroke_width=2, style=ft.PaintingStyle.STROKE)),
                cv.Circle(sx, floor - 60, 2.2, ft.Paint(color=GREEN)),
                cv.Line(sx, floor - 38, sx, floor + 2, thin),
                cv.Line(sx + 8, floor - 58, px1 + 6, floor - 88, fov),
                cv.Line(sx + 8, floor - 46, px1 + 6, floor + 2, fov),
            ]
            if camera_view == "side":                       # prone: push-up / plank
                head = (px0 - 4, floor - 42)
                sh = (px0 + 16, floor - 34)
                hip = (px0 + (px1 - px0) * 0.52, floor - 24)
                ank = (px1, floor - 8)
                wr, el = (px0 + 18, floor), (px0 + 34, floor - 17)
                shapes += [
                    cv.Circle(head[0], head[1], 8, ft.Paint(color=CYAN, stroke_width=3,
                                                            style=ft.PaintingStyle.STROKE)),
                    cv.Line(*sh, *hip, body), cv.Line(*hip, *ank, body),
                    cv.Line(*sh, *el, body), cv.Line(*el, *wr, body),
                    *[cv.Circle(x, y, 3.6, joints) for (x, y) in (sh, el, wr, hip, ank)],
                ]
            else:                                            # standing: squat
                cx = (px0 + px1) / 2
                head = (cx, floor - 92)
                sh = (cx, floor - 78)
                hip = (cx - 6, floor - 48)
                knee = (cx + 14, floor - 26)
                ank = (cx - 2, floor - 4)
                shapes += [
                    cv.Circle(head[0], head[1], 8, ft.Paint(color=CYAN, stroke_width=3,
                                                            style=ft.PaintingStyle.STROKE)),
                    cv.Line(*sh, *hip, body), cv.Line(*hip, *knee, body),
                    cv.Line(*knee, *ank, body),
                    *[cv.Circle(x, y, 3.6, joints) for (x, y) in (sh, hip, knee, ank)],
                ]
            shapes += [
                cv.Line(sx, floor + 13, px0, floor + 13, ft.Paint(color=MUTED, stroke_width=1)),
                cv.Text(x=(sx + px0) / 2 - 20, y=floor + 3, value="≈ 2 m",
                        style=ft.TextStyle(size=10, color=MUTED, font_family=M_FONT)),
            ]
            canvas.shapes = shapes

        def on_resize(e):
            draw(e.width, e.height or 140)
            canvas.update()

        canvas.on_resize = on_resize
        draw(340)
        return canvas

    # =====================================================================
    # LIVE
    # =====================================================================
    def build_live() -> ft.View:
        pres = state.presentation()
        unit = pres.get("unit", "reps")
        target = state.target
        is_hold = unit == "seconds"

        L = {
            "active": True, "inflight": False, "streaming": False, "sensor": 90,
            "fails": 0, "fps": 0.0, "last_t": None, "last_send": 0.0, "goal_done": False,
            "canvas_w": 360.0, "canvas_h": 480.0, "t_start": time.time(),
            "stale_drops": 0, "sent": 0, "applied": 0, "newest_capture": 0.0,
        }
        ui["live"] = L
        state.live_count = state.live_clean = 0
        state.summary = None
        state.receipt = None
        state.summary_saved = False
        state.set_started = time.time()
        conn.set_live_mode(True)

        # ---------------- skeleton overlay ----------------
        overlay = cv.Canvas(shapes=[], expand=True)

        def on_overlay_resize(e):
            L["canvas_w"], L["canvas_h"] = e.width, e.height

        overlay.on_resize = on_overlay_resize
        overlay.resize_interval = 100

        def draw_skeleton(kps, alert=False):
            if not kps:
                overlay.shapes = []
                return
            w, h = L["canvas_w"], L["canvas_h"]
            pts = [(k[0] * w, k[1] * h, k[2]) for k in kps]
            shapes = []
            line_col = RED if alert else CYAN
            focus = ARM_EDGES if not is_hold and state.exercise == "pushup" else (
                LEG_EDGES if state.exercise == "squat" else set())
            for a, b in SKELETON:
                if pts[a][2] > 0.3 and pts[b][2] > 0.3:
                    col = GREEN if (a, b) in focus else line_col
                    shapes.append(cv.Line(pts[a][0], pts[a][1], pts[b][0], pts[b][1],
                                          ft.Paint(color="#AA000000", stroke_width=7,
                                                   stroke_cap=ft.StrokeCap.ROUND)))
                    shapes.append(cv.Line(pts[a][0], pts[a][1], pts[b][0], pts[b][1],
                                          ft.Paint(color=col, stroke_width=3.5,
                                                   stroke_cap=ft.StrokeCap.ROUND)))
            for x, y, c in pts[5:]:
                if c > 0.3:
                    shapes.append(cv.Circle(x, y, 6, ft.Paint(color="#05070D")))
                    shapes.append(cv.Circle(x, y, 4, ft.Paint(color=GREEN)))
            overlay.shapes = shapes

        # ---------------- camera / preview ----------------
        test_frames = [] if is_mobile else _test_frames()
        test_image = None
        camera = None
        if is_mobile and fc is not None:
            camera = fc.Camera(preview_enabled=True, expand=True, content=overlay)
            preview = ft.Container(camera, expand=True, alignment=ft.Alignment.CENTER)
        elif test_frames:
            fw, fh = _jpeg_size(test_frames[0]) or (4, 3)
            test_image = ft.Image(src=test_frames[0], fit=ft.BoxFit.FILL, gapless_playback=True,
                                  expand=True)
            preview = ft.Container(
                ft.Container(ft.Stack([ft.Container(test_image, expand=True, left=0, top=0,
                                                    right=0, bottom=0), overlay], expand=True),
                             aspect_ratio=fw / fh),
                expand=True, alignment=ft.Alignment.CENTER)
        else:
            preview = ft.Container(ft.Column([
                ft.Icon(ft.Icons.VIDEOCAM_OFF_OUTLINED, color=DIM, size=40),
                T("Camera streaming runs on the phone app.", size=13, color=MUTED,
                  align=ft.TextAlign.CENTER),
                T("Desktop test: set REPFLOW_TEST_FRAMES to a folder of JPEGs.", size=11,
                  color=DIM, align=ft.TextAlign.CENTER),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER, spacing=8),
                expand=True, alignment=ft.Alignment.CENTER)

        # ---------------- HUD ----------------
        timer_text = mono("00:00", size=13, color=TEXT)
        track_dot = dot(AMBER, 8)
        track_text = T("STARTING", size=10.5, font=S_FONT, color=AMBER, spacing=1.4)
        lat_chip = mono("— ms", size=11)

        count_text = H("0", size=50, color=GREEN)
        count_text.animate_scale = ft.Animation(260, ft.AnimationCurve.EASE_OUT_BACK)
        count_ring = ft.ProgressRing(value=0, width=118, height=118, stroke_width=9, color=GREEN,
                                     bgcolor="#14FFFFFF", stroke_cap=ft.StrokeCap.ROUND)
        goal_text = T(goal_caption(unit, target), size=10.5, font=S_FONT, color=MUTED, spacing=1.2)
        ring = ft.Stack([
            count_ring,
            ft.Container(ft.Column([count_text, goal_text], spacing=-4, tight=True,
                                   horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                         width=118, height=118, alignment=ft.Alignment.CENTER),
        ], width=118, height=118)

        stage_boxes = {}
        stage_row = ft.Row(spacing=5)
        for key, lbl in STAGE_LABELS["seconds" if is_hold else "reps"]:
            box = ft.Container(T(lbl, size=9.5, font=M_FONT, color=DIM), expand=True, height=26,
                               alignment=ft.Alignment.CENTER, border_radius=8, bgcolor="#08FFFFFF",
                               border=ft.Border.all(1, "#14FFFFFF"), animate=ft.Animation(150))
            stage_boxes[key] = box
            stage_row.controls.append(box)

        form_icon = ft.Icon(ft.Icons.HOURGLASS_TOP, color=AMBER, size=16)
        form_text = T("Getting ready…", size=13, font=S_FONT, color=AMBER, expand=True, max_lines=1)
        form_badge = ft.Container(ft.Row([form_icon, form_text], spacing=8),
                                  padding=ft.Padding.symmetric(horizontal=10, vertical=8),
                                  border_radius=12, bgcolor="#CC05070D",
                                  border=ft.Border.all(1, ahex(AMBER, "66")))
        score_text = H("—", size=22, color=CYAN)

        toast = ft.Container(
            ft.Row([ft.Icon(ft.Icons.ADD_TASK, color=INK, size=16),
                    T("", size=13, font=H_FONT, color=INK)], spacing=6, tight=True),
            padding=ft.Padding.symmetric(horizontal=12, vertical=7), border_radius=99,
            bgcolor=GREEN, opacity=0, animate_opacity=ft.Animation(250),
            shadow=glow(GREEN, 20, opacity="66"))

        countdown = H("", size=96, color=GREEN)
        countdown.animate_scale = ft.Animation(300, ft.AnimationCurve.EASE_OUT)
        countdown_box = ft.Container(countdown, alignment=ft.Alignment.CENTER, expand=True,
                                     visible=False, bgcolor="#9905070D")

        cue_text = T(pres.get("setup_hint") or "Get into position — the coach will calibrate",
                     size=13.5, font=S_FONT, color=TEXT, expand=True)
        cue_banner = ft.Container(
            ft.Row([ft.Icon(ft.Icons.RECORD_VOICE_OVER, color=GREEN, size=18), cue_text],
                   spacing=10),
            padding=ft.Padding.symmetric(horizontal=14, vertical=12), border_radius=14,
            bgcolor=GREEN_GLOW, border=ft.Border.all(1, ahex(GREEN, "66")),
            animate=ft.Animation(200))

        primary_v = H("—", size=22)
        secondary_v = H("—", size=22)
        clean_v = H("0", size=22)
        metrics = ft.Row([
            stat_tile(pres.get("primary_label", "angle"), primary_v, ft.Icons.CHANGE_HISTORY, CYAN),
            stat_tile(pres.get("secondary_label", "line"), secondary_v, ft.Icons.STRAIGHTEN, GREEN),
            stat_tile("clean" if not is_hold else "quality", clean_v,
                      ft.Icons.VERIFIED_OUTLINED, VIOLET),
        ], spacing=8)

        audio_btn = ft.IconButton(icon=ft.Icons.VOLUME_UP if state.voice else ft.Icons.VOLUME_OFF,
                                  icon_color=GREEN if state.voice else MUTED)

        def toggle_audio(e):
            state.voice = not state.voice
            audio_btn.icon = ft.Icons.VOLUME_UP if state.voice else ft.Icons.VOLUME_OFF
            audio_btn.icon_color = GREEN if state.voice else MUTED
            page.update()

        audio_btn.on_click = toggle_audio

        def chip_box(ctrl):
            return ft.Container(ctrl, padding=ft.Padding.symmetric(horizontal=9, vertical=5),
                                border_radius=9, bgcolor="#B305070D",
                                border=ft.Border.all(1, "#1AFFFFFF"))

        camera_stack = ft.Stack([
            preview,
            ft.Container(ft.Row([chip_box(ft.Row([track_dot, track_text], spacing=6, tight=True)),
                                 chip_box(ft.Row([ft.Icon(ft.Icons.LOCK, size=11, color=GREEN),
                                                  lat_chip], spacing=5, tight=True))],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                         left=10, right=10, top=10),
            ft.Container(toast, right=10, bottom=10),
            countdown_box,
        ], expand=True)

        camera_area = ft.Container(camera_stack, expand=True, border_radius=22,
                                   clip_behavior=ft.ClipBehavior.ANTI_ALIAS, bgcolor="#02040A",
                                   border=ft.Border.all(1.5, ahex(GREEN, "55")),
                                   shadow=glow(GREEN, 24, opacity="22"))

        hud = card(ft.Row([
            ring,
            ft.Column([
                ft.Row([eyebrow("form score", color=MUTED, size=9.5), score_text],
                       alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                stage_row,
                form_badge,
            ], spacing=8, expand=True),
        ], spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER), padding=12, radius=20)

        # ---------------- result handling ----------------
        def set_form(icon, text, color):
            form_icon.icon = icon
            form_icon.color = color
            form_text.value = text
            form_text.color = color
            form_badge.border = ft.Border.all(1, ahex(color, "66"))

        def show_cue(cue_id):
            if cue_id not in CUES:
                return
            cue_text.value = text_of(cue_id)
            tone = tone_of(cue_id)
            col = {"good": GREEN, "tip": AMBER, "warn": RED}.get(tone, GREEN)
            cue_banner.bgcolor = ahex(col, "1F")
            cue_banner.border = ft.Border.all(1, ahex(col, "88"))

        def describe_state(r):
            """The one-line 'what is my body doing' badge."""
            if not r.get("is_detected"):
                # "Step into frame" is useless advice when you are already
                # standing in it. The server says *why* it cannot see you,
                # so say that instead — the fix is different every time.
                return ft.Icons.PERSON_SEARCH, DETECT_HELP.get(
                    r.get("detect_reason"), "Step into frame"), AMBER
            if not r.get("is_calibrated"):
                pct = int(round(r.get("calibration_pct", 0) * 100))
                return ft.Icons.CENTER_FOCUS_STRONG, f"Calibrating… {pct}%", AMBER
            faults = r.get("faults") or []
            if faults:
                return ft.Icons.TRENDING_DOWN, FAULT_LABEL.get(faults[0], faults[0]), (
                    RED if faults[0] in ("hips_sag", "valgus", "sag") else AMBER)
            if is_hold:
                return ft.Icons.CHECK_CIRCLE, "Line locked — hold it", GREEN
            return ft.Icons.CHECK_CIRCLE, "Form looks good", GREEN

        def apply_result(r: dict, captured_at: float):
            # Out-of-order results would make the counter and the skeleton
            # jump backwards. Newest frame wins, always.
            if captured_at < L["newest_capture"]:
                return
            L["newest_capture"] = captured_at
            L["applied"] += 1

            now = time.time()
            if L["last_t"]:
                inst = 1.0 / max(1e-3, now - L["last_t"])
                L["fps"] = inst if not L["fps"] else 0.75 * L["fps"] + 0.25 * inst
            L["last_t"] = now
            lat_chip.value = f"{client.last_rtt_ms or 0:.0f} ms"

            count = r.get("count", 0)
            state.live_count = count
            count_text.value = ex.format_count(unit, count)
            if is_hold:
                q = r.get("quality")
                clean_v.value = f"{q}%" if q is not None else "—"
                count_ring.value = min(1.0, count / target) if target else ((count % 30) / 30)
            else:
                state.live_clean = r.get("clean", 0)
                clean_v.value = str(state.live_clean)
                count_ring.value = min(1.0, count / target) if target else \
                    ((count % 10) / 10 if count else 0)
            if r.get("form_score") is not None:
                score_text.value = str(r["form_score"])

            p, s = r.get("primary"), r.get("secondary")
            primary_v.value = f"{p:.0f}°" if p is not None else "—"
            secondary_v.value = (f"{s:.0f}°" if s is not None else "—") if not is_hold else (
                f"{s:.1f}" if s is not None else "—")

            for key, box in stage_boxes.items():
                on = r.get("stage") == key and r.get("is_calibrated")
                box.bgcolor = GREEN if on else "#08FFFFFF"
                box.border = ft.Border.all(1, GREEN if on else "#14FFFFFF")
                box.content.color = INK if on else DIM

            if r.get("is_detected"):
                track_dot.bgcolor, track_text.value, track_text.color = GREEN, "TRACKING", GREEN
            else:
                track_dot.bgcolor, track_text.value, track_text.color = AMBER, "SEARCHING", AMBER
            set_form(*describe_state(r))

            alert = bool(set(r.get("faults") or []) & {"hips_sag", "sag", "valgus"})
            draw_skeleton(r.get("keypoints"), alert=alert)

            cue = r.get("cue_id")
            if r.get("rep_completed"):
                last = r.get("last_rep") or {}
                label = f"+1 · {last.get('score', '')} PTS" if not is_hold else \
                    f"{last.get('score', '')} PTS"
                toast.content.controls[1].value = label
                toast.opacity = 1
                count_text.scale = 1.22
                haptic("heavy")
                if cue is None:
                    play("tick")

                async def _settle():
                    await asyncio.sleep(0.25)
                    count_text.scale = 1.0
                    page.update()
                    await asyncio.sleep(0.9)
                    toast.opacity = 0
                    page.update()

                page.run_task(_settle)

            if cue:
                show_cue(cue)
                play(cue)

            if target and count >= target and not L["goal_done"]:
                L["goal_done"] = True
                play("goal")

        def rotation_hint() -> int:
            return (L["sensor"] + (360 - state.device_rotation())) % 360

        def show_reconnecting():
            track_dot.bgcolor, track_text.value, track_text.color = RED, "RECONNECTING", RED
            cue_text.value = "Reconnecting securely — hold your position"

        # ---------------- network ----------------
        def send(jpeg: bytes, hint: int, display: int, captured_at: float):
            """Worker thread: network only. UI changes go back through on_ui."""
            try:
                r = client.send_frame(jpeg, rotation_hint=hint, display_rotation=display,
                                      captured_at=captured_at)
                L["fails"] = 0
                conn.frame_ok(client.last_rtt_ms)
                if not L["active"]:
                    return
                if r.get("stale"):
                    # Too late to coach from, so it is skipped — but a
                    # long run of these means the athlete is watching a
                    # screen that has stopped moving, and silence there
                    # is the worst possible answer.
                    L["stale_drops"] += 1
                    if L["stale_drops"] == STALE_WARN_AFTER:
                        on_ui(lambda: setattr(
                            cue_text, "value",
                            "Frames are arriving late — move closer to the router"))
                elif "error" not in r:
                    L["stale_drops"] = 0
                    on_ui(apply_result, r, captured_at)
            except SecurityError:
                L["fails"] += 1
                if L["active"]:
                    on_ui(show_reconnecting)
                    if conn.ensure_session():
                        on_ui(lambda: setattr(cue_text, "value", "Secure link restored"))
            except SpotterClientError:
                L["fails"] += 1
                conn.frame_failed()
                if L["active"] and L["fails"] in (2, 20, 60):
                    on_ui(show_reconnecting)
            finally:
                L["inflight"] = False

        def on_stream_image(e):
            # One frame in flight at a time. A frame that arrives while the
            # last one is still out is dropped here rather than queued —
            # queueing would only turn dropped frames into late ones.
            now = time.time()
            if (not L["streaming"] or L["inflight"] or not L["active"]
                    or now - L["last_send"] < MIN_SEND_INTERVAL_S):
                return
            L["inflight"] = True
            L["last_send"] = now
            L["sent"] += 1
            page.run_thread(send, e.bytes, rotation_hint(), L["sensor"], now)

        if camera is not None:
            camera.on_stream_image = on_stream_image

        # ---------------- lifecycle ----------------
        async def run_countdown():
            countdown_box.visible = True
            for n in ("3", "2", "1", "GO"):
                if not L["active"]:
                    return
                countdown.value = n
                countdown.scale = 1.35
                page.update()
                await asyncio.sleep(0.12)
                countdown.scale = 1.0
                page.update()
                await asyncio.sleep(0.7)
            countdown_box.visible = False
            page.update()

        def start_set():
            try:
                if not conn.ensure_session():
                    raise SpotterClientError("no session")
                client.clone().start_set(state.exercise, target)
            except SpotterClientError:
                on_ui(lambda: setattr(cue_text, "value", "Waiting for a secure link to the coach…"))

        async def start_camera():
            wl = services.get("wakelock")
            if wl:
                try:
                    await wl.enable()
                except Exception:
                    pass
            perm = services.get("perm")
            if perm is not None and fph is not None:
                status = await perm.request(fph.Permission.CAMERA)
                if status != fph.PermissionStatus.GRANTED:
                    cue_text.value = "Camera permission is needed — enable it in Settings"
                    page.update()
                    return
            cams = await camera.get_available_cameras()
            if not cams:
                cue_text.value = "No camera found on this device"
                page.update()
                return
            rear = next((c for c in cams if c.lens_direction == fc.CameraLensDirection.BACK),
                        cams[0])
            L["sensor"] = int(getattr(rear, "sensor_orientation", 90) or 0) % 360
            fmt = (fc.ImageFormatGroup.BGRA8888 if page.platform == ft.PagePlatform.IOS
                   else fc.ImageFormatGroup.YUV420)
            preset = {"low": fc.ResolutionPreset.LOW, "medium": fc.ResolutionPreset.MEDIUM,
                      "high": fc.ResolutionPreset.HIGH}.get(
                getattr(config, "CAMERA_PRESET", "medium"), fc.ResolutionPreset.MEDIUM)
            await camera.initialize(description=rear, resolution_preset=preset,
                                    enable_audio=False, image_format_group=fmt)
            page.run_thread(start_set)
            await run_countdown()
            L["streaming"] = True
            await camera.start_image_stream()

        async def init_camera():
            await asyncio.sleep(0.5)
            try:
                await asyncio.wait_for(start_camera(), timeout=60)
            except asyncio.TimeoutError:
                cue_text.value = "Camera didn't respond — check the permission prompt"
                page.update()
            except Exception as ex_:
                cue_text.value = f"Camera error: {ex_}"
                page.update()

        def test_feed_loop():
            start_set()
            i = 0
            while L["active"]:
                jpg = test_frames[i % len(test_frames)]
                i += 1
                on_ui(lambda j=jpg: setattr(test_image, "src", j))
                if not L["inflight"]:
                    L["inflight"] = True
                    L["sent"] += 1
                    send(jpg, 0, 0, time.time())
                time.sleep(0.08)

        async def test_start():
            await run_countdown()
            L["streaming"] = True
            page.run_thread(test_feed_loop)

        async def clock():
            while L["active"]:
                timer_text.value = fmt_time(time.time() - L["t_start"])
                try:
                    timer_text.update()
                except Exception:
                    pass
                await asyncio.sleep(1)

        if camera is not None:
            page.run_task(init_camera)
        elif test_frames:
            page.run_task(test_start)
        else:
            page.run_thread(start_set)
        page.run_task(clock)

        def end_set(e=None):
            if not L["active"]:
                return
            L["active"] = False
            L["streaming"] = False
            state.set_ended = time.time()
            conn.set_live_mode(False)

            async def _stop():
                if camera is not None:
                    try:
                        await camera.stop_image_stream()
                    except Exception:
                        pass
                wl = services.get("wakelock")
                if wl:
                    try:
                        await wl.disable()
                    except Exception:
                        pass

            page.run_task(_stop)
            ui["live"] = None
            navigate("/summary")

        L["end"] = end_set

        top = ft.Row([
            ft.Row([
                ft.Container(ft.Row([dot(RED, 7), T("LIVE", size=11, font=H_FONT, color=TEXT,
                                                    spacing=1.5)], spacing=6, tight=True),
                             padding=ft.Padding.symmetric(horizontal=10, vertical=6),
                             border_radius=99, bgcolor="#22EF4444",
                             border=ft.Border.all(1, "#66EF4444")),
                timer_text,
            ], spacing=10),
            ft.Row([audio_btn,
                    ft.Container(T("END", size=12, font=H_FONT, color=TEXT, spacing=1.5),
                                 padding=ft.Padding.symmetric(horizontal=14, vertical=8),
                                 border_radius=12, bgcolor=SURFACE_2,
                                 border=ft.Border.all(1, BORDER_2), on_click=end_set, ink=True)],
                   spacing=4),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        body = ft.Column([
            top, camera_area, hud, cue_banner, metrics,
            neon_button("END & ANALYSE", ft.Icons.INSIGHTS, on_click=end_set, height=54),
        ], spacing=10, expand=True)
        return view("/live", ft.SafeArea(
            ft.Container(body, padding=ft.Padding.only(left=14, right=14, top=6, bottom=10),
                         expand=True), expand=True))

    # =====================================================================
    # REPORT
    # =====================================================================
    def build_summary() -> ft.View:
        body = ft.Column(spacing=14, scroll=ft.ScrollMode.AUTO, expand=True)
        body.controls = [ft.Column([
            ft.Container(height=120),
            ft.ProgressRing(color=GREEN, width=46, height=46, stroke_width=4),
            H("Analysing your set…", size=20),
            T("Joint angles, tempo and body-line data", size=12.5, color=MUTED),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=14)]

        def load():
            s, receipt = None, None
            try:
                result = client.clone().finish()
                if result:
                    s = result.get("summary")
                    receipt = result.get("receipt")
            except SpotterClientError:
                s = None
            if not s:
                count = state.live_count
                s = {"count": count, "unit": state.presentation().get("unit", "reps"),
                     "clean": state.live_clean, "clean_pct": 0, "form_score": 0, "grade": "—",
                     "rep_list": [], "duration_s": state.set_ended - state.set_started,
                     "insights": ["Couldn't reach the coach for the full report — "
                                  "showing what the phone counted."]}
            state.summary = s
            state.receipt = receipt
            if s.get("count") and not state.summary_saved:
                state.summary_saved = True
                unit = s.get("unit", "reps")
                label = (f"{s['count']} reps · {s.get('clean_pct', 0)}% clean" if unit == "reps"
                         else f"{fmt_time(s.get('hold_s', s['count']))} hold · "
                              f"{s.get('quality_pct', 0)}% clean line")
                state.history.append({
                    "date": dt.datetime.now().isoformat(timespec="minutes"),
                    "exercise": s.get("exercise", state.exercise),
                    "label": label,
                    "reps": s["count"] if unit == "reps" else 0,
                    "clean_pct": s.get("clean_pct", 0),
                    "score": s.get("form_score", 0), "grade": s.get("grade", "—"),
                    "scores": [r["score"] for r in s.get("rep_list", [])],
                })
                state.save_history(conn)
            on_ui(render, s)

        def render(s):
            unit = s.get("unit", "reps")
            is_hold = unit == "seconds"
            score = s.get("form_score", 0) or 0
            grade = s.get("grade", "—")
            count = s.get("count", 0)
            col = GREEN if grade.startswith("A") else (CYAN if grade.startswith("B") else AMBER)
            if not count:
                col = CYAN

            ring = ft.Stack([
                ft.ProgressRing(value=score / 100 if count else 0, width=150, height=150,
                                stroke_width=12, color=col, bgcolor="#12FFFFFF",
                                stroke_cap=ft.StrokeCap.ROUND),
                ft.Container(ft.Column([H(grade, size=48, color=col),
                                        T(f"{score}/100" if count else "nothing counted",
                                          size=12, font=M_FONT, color=MUTED)],
                                       spacing=-4, tight=True,
                                       horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                             width=150, height=150, alignment=ft.Alignment.CENTER),
            ], width=150, height=150)

            headline = (f"{fmt_time(s.get('hold_s', 0))} hold" if is_hold
                        else ("Set complete" if count else "No reps counted"))
            header = ft.Container(ft.Column([
                ft.Row([pill([ft.Icon(ft.Icons.AUTO_AWESOME, size=13, color=CYAN),
                              T("AI PERFORMANCE REPORT", size=10, font=S_FONT, color=CYAN,
                                spacing=1.5)], color=CYAN)],
                       alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([ring], alignment=ft.MainAxisAlignment.CENTER),
                H(headline, size=26, align=ft.TextAlign.CENTER),
                T(f"{s.get('exercise_name', '')} · {fmt_time(s.get('duration_s', 0))} · "
                  f"{s.get('frames', 0)} frames analysed", size=12, color=MUTED,
                  align=ft.TextAlign.CENTER),
            ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                padding=20, border_radius=24, border=ft.Border.all(1, ahex(col, "44")),
                gradient=ft.LinearGradient(begin=ft.Alignment.TOP_CENTER,
                                           end=ft.Alignment.BOTTOM_CENTER,
                                           colors=[ahex(col, "1A"), SURFACE]))

            def num(v, u=""):
                return ft.Row([H(str(v), size=24), T(u, size=11, color=MUTED)], spacing=3,
                              vertical_alignment=ft.CrossAxisAlignment.END, tight=True)

            tempo = s.get("avg_tempo_s")
            if is_hold:
                grid = ft.Column([
                    ft.Row([stat_tile("hold time", num(fmt_time(s.get("hold_s", 0))),
                                      ft.Icons.TIMER_OUTLINED, GREEN),
                            stat_tile("clean line", num(s.get("quality_pct", 0), "%"),
                                      ft.Icons.VERIFIED_OUTLINED, GREEN)], spacing=10),
                    ft.Row([stat_tile("consistency", num(s.get("consistency", "—"),
                                                         "%" if count else ""),
                                      ft.Icons.INSIGHTS, VIOLET),
                            stat_tile("est. energy", num(s.get("est_kcal", "—"),
                                                         "kcal" if count else ""),
                                      ft.Icons.LOCAL_FIRE_DEPARTMENT_OUTLINED, AMBER)], spacing=10),
                ], spacing=10)
            else:
                grid = ft.Column([
                    ft.Row([stat_tile("total reps", num(count), ft.Icons.REPEAT, GREEN),
                            stat_tile("clean form", num(s.get("clean_pct", 0), "%"),
                                      ft.Icons.VERIFIED_OUTLINED, GREEN)], spacing=10),
                    ft.Row([stat_tile("avg depth",
                                      num(f"{s.get('avg_depth_deg', 0):.0f}" if count else "—", "°"),
                                      ft.Icons.CHANGE_HISTORY, CYAN),
                            stat_tile("tempo", num(f"{tempo:.1f}" if tempo else "—", "s/rep"),
                                      ft.Icons.SPEED, CYAN)], spacing=10),
                    ft.Row([stat_tile("consistency", num(s.get("consistency", "—"),
                                                         "%" if count else ""),
                                      ft.Icons.INSIGHTS, VIOLET),
                            stat_tile("est. energy", num(s.get("est_kcal", "—"),
                                                         "kcal" if count else ""),
                                      ft.Icons.LOCAL_FIRE_DEPARTMENT_OUTLINED, AMBER)], spacing=10),
                ], spacing=10)

            parts = [top_bar("AI Report"), header, grid]

            reps_list = s.get("rep_list", [])
            if reps_list:
                bars = []
                maxh = 110
                for r in reps_list[-24:]:
                    c = GREEN if r["clean"] else (AMBER if r["score"] >= 70 else RED)
                    bars.append(ft.Column([
                        T(str(r["score"]), size=9, font=M_FONT, color=MUTED),
                        ft.Container(width=14, height=max(6, maxh * r["score"] / 100), bgcolor=c,
                                     border_radius=5,
                                     shadow=[ft.BoxShadow(blur_radius=8, color=ahex(c, "55"))]),
                        T(str(r["rep"]), size=9, font=M_FONT, color=DIM),
                    ], spacing=3, horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True))
                parts.append(card(ft.Column([
                    section_title("Ten-second blocks" if is_hold else "Rep-by-rep quality",
                                  T("SCORE", size=10, font=S_FONT, color=DIM, spacing=1.2)),
                    ft.Row(bars, spacing=6, scroll=ft.ScrollMode.AUTO,
                           vertical_alignment=ft.CrossAxisAlignment.END),
                    ft.Row([legend_dot(GREEN, "clean"), legend_dot(AMBER, "minor fault"),
                            legend_dot(RED, "major fault")], spacing=14),
                ], spacing=12), padding=16))

                faults = s.get("faults") or {}
                if faults:
                    parts.append(card(ft.Column([
                        section_title("What went wrong"),
                        ft.Row([pill(T(f"{FAULT_LABEL.get(k, k)} × {v}", size=12, font=S_FONT,
                                       color=AMBER), color=AMBER, padding=(12, 7))
                                for k, v in faults.items()], wrap=True, spacing=8, run_spacing=8),
                    ], spacing=10), padding=16))

            ins = s.get("insights") or []
            if ins:
                rows = []
                for i, text in enumerate(ins):
                    rows.append(ft.Row([
                        ft.Container(T(str(i + 1), size=11, font=M_FONT, color=INK), width=22,
                                     height=22, border_radius=11, bgcolor=CYAN if i else GREEN,
                                     alignment=ft.Alignment.CENTER),
                        T(text, size=13, color="#D5E0F0", expand=True, height=1.35),
                    ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.START))
                parts.append(card(ft.Column([
                    ft.Row([icon_badge(ft.Icons.PSYCHOLOGY_OUTLINED, CYAN, 34, 18),
                            T("Coach insights", size=15, font=S_FONT)], spacing=10),
                    *rows,
                ], spacing=12), padding=16))

            # The privacy receipt, right where the value of the set is shown.
            r = state.receipt or {}
            if r:
                parts.append(card(ft.Row([
                    icon_badge(ft.Icons.SHIELD_OUTLINED, GREEN, 34, 17),
                    ft.Column([
                        T(f"{r.get('frames_received', 0)} frames encrypted · "
                          f"{r.get('frames_stored', 0)} stored", size=13, font=S_FONT),
                        T("Keys destroyed. Nothing was written to disk. Tap for details.",
                          size=11.5, color=MUTED),
                    ], spacing=2, expand=True, tight=True),
                    ft.Icon(ft.Icons.CHEVRON_RIGHT, color=DIM, size=18),
                ], spacing=12), padding=14, radius=16,
                    border=ahex(GREEN, "33"), on_click=lambda e: navigate("/privacy")))

            parts += [
                neon_button("START ANOTHER SET", ft.Icons.REPLAY,
                            on_click=lambda e: navigate("/setup")),
                ghost_button("Back to home", ft.Icons.HOME_OUTLINED,
                             on_click=lambda e: navigate("/home")),
                ft.Container(height=8),
            ]
            body.controls = parts

        page.run_thread(load)
        return view("/summary", ft.SafeArea(
            ft.Container(body, padding=ft.Padding.only(left=18, right=18, top=10), expand=True),
            expand=True))

    def legend_dot(color, label):
        return ft.Row([ft.Container(width=9, height=9, border_radius=3, bgcolor=color),
                       T(label, size=11, color=MUTED)], spacing=6, tight=True)

    return build_setup, build_live, build_summary


def goal_caption(unit: str, target: int) -> str:
    if not target:
        return "OPEN SET"
    return f"/ {target}s" if unit == "seconds" else f"/ {target}"
