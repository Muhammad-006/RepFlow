"""
RepFlow — the AI spotter that never keeps your video.

Screens:  Boot → Home → Setup → Live → Report,  plus Privacy.

The phone captures, encrypts and streams camera frames to a coach server;
everything shown — counts, joint angles, form scores, cues — comes back
from that server's pose model. Frames are sealed with keys agreed per
session (crypto_box.py) and the server is verified against a fingerprint
this phone remembers, so nothing in the middle can read a frame and
nothing at the far end keeps one.

Desktop testing: `python main.py` runs without a camera. Point
REPFLOW_TEST_FRAMES at a folder of JPEGs to drive the whole live pipeline
from disk instead.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import math
import time

import flet as ft

import config
import discovery
import exercises as ex
from connection import ConnectionManager
from cues import CUES, EARCONS
from spotter_client import SpotterClient
from ui_kit import (AMBER, B_FONT, BG, BORDER, BORDER_2, CYAN, DIM, FONTS, GREEN, INK, MUTED, RED, S_FONT,
                    SURFACE, SURFACE_2, SURFACE_3, TEXT, VIOLET, H, T, ahex, background, card,
                    dot, eyebrow, ghost_button, glow, icon_badge, logo_mark, mono, neon_button,
                    pill, section_title, stat_tile)

try:
    import flet_audio as fa
except Exception:  # pragma: no cover
    fa = None
try:
    import flet_permission_handler as fph
except Exception:  # pragma: no cover
    fph = None

APP_VERSION = "3.0"


class AppState:
    def __init__(self):
        self.exercise = "pushup"
        self.target = 10
        self.voice = True
        self.summary: dict | None = None
        self.receipt: dict | None = None
        self.summary_saved = False
        self.live_count = 0
        self.live_clean = 0
        self.set_started = 0.0
        self.set_ended = 0.0
        self.accel = (0.0, 9.8)
        persisted = discovery.load_state()
        self.history: list = persisted.get("history", [])
        self.catalog: list = ex.merge_catalog([])
        self.roadmap: list = []

    def save_history(self, conn: ConnectionManager):
        conn.persist["history"] = self.history[-50:]
        discovery.save_state(conn.persist)

    def presentation(self) -> dict:
        for e in self.catalog:
            if e["key"] == self.exercise:
                return e
        return ex.presentation(self.exercise)

    def device_rotation(self) -> int:
        """How far the phone is turned from upright, read from gravity."""
        x, y = self.accel
        if abs(x) < 3.0 and abs(y) < 3.0:
            return 0
        ang = math.degrees(math.atan2(x, y)) % 360
        return int(round(ang / 90.0)) % 4 * 90


def fmt_time(seconds: float) -> str:
    s = max(0, int(seconds or 0))
    return f"{s // 60:02d}:{s % 60:02d}"


def greeting() -> str:
    h = dt.datetime.now().hour
    return "Good morning" if h < 12 else ("Good afternoon" if h < 18 else "Good evening")


def day_streak(history: list) -> int:
    """Consecutive days with at least one completed set, counting back from
    today. Today doesn't have to be logged yet to keep yesterday's streak
    alive — it only breaks once a full day goes by with nothing."""
    days = {h.get("date", "")[:10] for h in history if h.get("date")}
    days.discard("")
    if not days:
        return 0
    cur = dt.date.today()
    if cur.isoformat() not in days:
        cur -= dt.timedelta(days=1)
        if cur.isoformat() not in days:
            return 0
    streak = 0
    while cur.isoformat() in days:
        streak += 1
        cur -= dt.timedelta(days=1)
    return streak


# =========================================================================
def main(page: ft.Page):
    page.title = "RepFlow"
    page.theme_mode = ft.ThemeMode.DARK
    page.fonts = FONTS
    page.theme = ft.Theme(font_family=B_FONT, color_scheme_seed=GREEN)
    page.dark_theme = ft.Theme(font_family=B_FONT, color_scheme_seed=GREEN)
    page.bgcolor = BG
    page.padding = 0
    try:
        page.window.width, page.window.height = 420, 880
    except Exception:
        pass

    is_mobile = page.platform in (ft.PagePlatform.ANDROID, ft.PagePlatform.IOS)
    state = AppState()
    client = SpotterClient(timeout=config.REQUEST_TIMEOUT)
    conn = ConnectionManager(client)
    ui = {"route": "/", "live": None, "listener": None}

    # ---------------------------------------------------------- services
    services = {}
    audio_clips = {}
    if is_mobile:
        try:
            services["wakelock"] = ft.Wakelock()
            services["haptic"] = ft.HapticFeedback()

            def on_accel(e):
                state.accel = (e.x, e.y)

            services["accel"] = ft.Accelerometer(on_reading=on_accel,
                                                 interval=ft.Duration(milliseconds=250))
        except Exception:
            pass
        if fph is not None:
            services["perm"] = fph.PermissionHandler()
        if fa is not None:
            clips = {cid: path for cid, (_, path, _) in CUES.items() if path}
            clips.update(EARCONS)
            for cid, path in clips.items():
                audio_clips[cid] = fa.Audio(src=path, autoplay=False, volume=1.0)

    def play(cid: str):
        if not state.voice:
            return
        clip = audio_clips.get(cid)
        if clip is None:
            return

        async def _p():
            try:
                await clip.play(0)
            except Exception:
                pass

        page.run_task(_p)

    def haptic(kind="medium"):
        h = services.get("haptic")
        if not h:
            return

        async def _h():
            try:
                await (h.heavy_impact() if kind == "heavy" else h.medium_impact())
            except Exception:
                pass

        page.run_task(_h)

    # ---------------------------------------------------------- navigation
    def navigate(route: str):
        if ui["listener"]:
            conn.unsubscribe(ui["listener"])
            ui["listener"] = None
        ui["route"] = route
        page.route = route
        page.views.clear()
        builders = {"/": build_boot, "/home": build_home, "/setup": build_setup,
                    "/live": build_live, "/summary": build_summary, "/privacy": build_privacy}
        if route not in ("/", "/home"):
            page.views.append(build_home())
            if ui["listener"]:
                conn.unsubscribe(ui["listener"])
                ui["listener"] = None
        page.views.append(builders.get(route, build_home)())
        page.update()

    def on_view_pop(e):
        if ui["route"] == "/live" and ui["live"]:
            ui["live"]["end"]()
        else:
            navigate("/home")

    page.on_view_pop = on_view_pop

    def view(route, content, padding=0) -> ft.View:
        return ft.View(route=route, bgcolor=BG, padding=0,
                       controls=[ft.Container(content=content, gradient=background(),
                                              expand=True, padding=padding)])

    def on_ui(fn, *args):
        """Run `fn(*args)` on Flet's loop. Worker threads must never touch
        controls directly."""
        async def _run():
            try:
                fn(*args)
            finally:
                page.update()
        try:
            page.run_task(_run)
        except Exception:
            pass

    def listen(fn):
        def wrapped():
            on_ui(fn)
        ui["listener"] = wrapped
        conn.subscribe(wrapped)

    # ---------------------------------------------------------- shared bits
    def link_chip() -> tuple:
        """One line that says whether the link is up and encrypted."""
        lock = ft.Icon(ft.Icons.LOCK_OUTLINE, size=13, color=DIM)
        label = T("", size=11.5, font=S_FONT, color=MUTED)
        box = pill([lock, label], color=DIM)

        def refresh():
            if conn.state == "untrusted":
                col, text, icon = RED, "Identity changed — tap Privacy", ft.Icons.GPP_BAD
            elif conn.online:
                col, icon = GREEN, ft.Icons.LOCK
                rtt = f" · {conn.rtt_ms:.0f} ms" if conn.rtt_ms else ""
                text = f"Encrypted{rtt}"
            elif conn.state == "searching":
                col, text, icon = AMBER, "Finding your coach…", ft.Icons.RADAR
            else:
                col, text, icon = RED, "Coach offline", ft.Icons.LOCK_OPEN
            lock.icon, lock.color = icon, col
            label.value, label.color = text, col
            box.bgcolor = ahex(col, "14")
            box.border = ft.Border.all(1, ahex(col, "55"))

        refresh()
        return box, refresh

    # ---------------------------------------------------------- network panel
    def _net_glyph():
        if conn.state == "untrusted":
            return ft.Icons.GPP_BAD, RED
        if conn.online:
            return (ft.Icons.SETTINGS_ETHERNET if conn.manual_locked else ft.Icons.PUBLIC), GREEN
        if conn.state == "searching":
            return ft.Icons.SYNC, AMBER
        return ft.Icons.WIFI_OFF, RED

    def network_icon_button() -> tuple:
        """The dedicated network-status icon: tap it to see (or override)
        which server this phone is routed to."""
        icon = ft.Icon(ft.Icons.PUBLIC, color=DIM, size=19)
        box = ft.Container(icon, width=42, height=42, border_radius=14, bgcolor=SURFACE_2,
                           border=ft.Border.all(1, BORDER_2), alignment=ft.Alignment.CENTER,
                           on_click=lambda e: open_network_panel(), ink=True,
                           tooltip="Network routing")

        def refresh():
            gi, col = _net_glyph()
            icon.icon, icon.color = gi, col
            box.border = ft.Border.all(1, ahex(col, "55"))

        refresh()
        return box, refresh

    def open_network_panel(e=None):
        page.show_dialog(build_network_sheet())

    def build_network_sheet() -> ft.BottomSheet:
        """A clean modal: shows where this phone is currently routed (city
        only — never a raw IP unless the person typed it themselves), and
        an escape hatch to force a direct connection instead of trusting
        auto-discovery."""
        status_col = ft.Column(spacing=10)
        address_field = ft.TextField(
            label="IP address or server URL", hint_text="e.g. 192.168.1.20 or wss://coach.example.com",
            bgcolor=SURFACE_2, border_color=BORDER_2, focused_border_color=GREEN,
            color=TEXT, cursor_color=GREEN, border_radius=12, text_size=13.5)
        error_text = T("", size=12, color=RED)
        busy = ft.ProgressRing(width=15, height=15, stroke_width=2, color=INK, visible=False)
        connect_label = T("CONNECT DIRECTLY", size=13, font=S_FONT, color=INK, spacing=1.0)

        def render_status():
            status_col.controls = []
            if conn.state == "untrusted":
                col, icon, text = RED, ft.Icons.GPP_BAD, "Identity mismatch — see Privacy"
            elif conn.online:
                col = GREEN
                icon = ft.Icons.SETTINGS_ETHERNET if conn.manual_locked else ft.Icons.PUBLIC
                text = f"Connected · {conn.location_label}"
            elif conn.state == "searching":
                col, icon, text = AMBER, ft.Icons.SYNC, conn.status_text or "Searching…"
            else:
                col, icon, text = RED, ft.Icons.WIFI_OFF, conn.status_text or "Not connected"

            mode_pill = pill([T(conn.mode.upper(), size=9.5, font=S_FONT, color=col, spacing=1.2)],
                             color=col, padding=(9, 4))
            trailing = [mode_pill]
            if conn.online and conn.rtt_ms:
                trailing.append(mono(f"{conn.rtt_ms:.0f} ms", size=11, color=MUTED))
            status_col.controls.append(card(ft.Row([
                icon_badge(icon, col, 40, 19),
                ft.Column([T(text, size=13.5, font=S_FONT, max_lines=2),
                          ft.Row(trailing, spacing=8)], spacing=6, expand=True, tight=True),
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER), padding=14, radius=16))

            if conn.manual_locked:
                def use_auto(e):
                    conn.clear_manual_override()
                    render_status()
                    page.update()
                status_col.controls.append(ghost_button(
                    "Switch back to automatic routing", ft.Icons.AUTO_MODE, on_click=use_auto, height=46))

        render_status()

        def modal_listener():
            on_ui(render_status)

        conn.subscribe(modal_listener)

        def on_dismiss(e):
            conn.unsubscribe(modal_listener)

        def do_connect(e):
            addr = (address_field.value or "").strip()
            if not addr:
                error_text.value = "Type an address first"
                page.update()
                return
            error_text.value = ""
            busy.visible = True
            connect_btn.disabled = True
            page.update()
            ok = conn.connect_manual(addr)
            if not ok:
                error_text.value = "That doesn't look like a valid address"
                busy.visible = False
                connect_btn.disabled = False
                page.update()
                return

            async def _poll():
                for _ in range(48):          # ~12s: parse + probe + handshake
                    await asyncio.sleep(0.25)
                    if conn.state in ("online", "untrusted") or (
                            conn.state == "offline" and not conn.searching):
                        break
                busy.visible = False
                connect_btn.disabled = False
                if conn.state != "online":
                    error_text.value = conn.status_text
                render_status()
                page.update()

            page.run_task(_poll)

        connect_btn = ft.Container(
            content=ft.Row([connect_label, busy], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
            height=50, border_radius=14, bgcolor=GREEN, on_click=do_connect, ink=True,
            alignment=ft.Alignment.CENTER)

        content = ft.Container(
            ft.Column([
                ft.Row([ft.Row([ft.Icon(ft.Icons.PUBLIC, color=GREEN, size=18),
                                H("Network", size=18)], spacing=10),
                        ft.Container(ft.Icon(ft.Icons.CLOSE, color=MUTED, size=18), width=34,
                                    height=34, border_radius=10, bgcolor=SURFACE_2,
                                    alignment=ft.Alignment.CENTER, ink=True,
                                    on_click=lambda e: page.pop_dialog())],
                       alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                status_col,
                ft.Divider(height=1, color=BORDER),
                ft.Column([
                    eyebrow("advanced · manual override"),
                    T("Bypass automatic routing and connect straight to a server you trust.",
                      size=11.5, color=MUTED, height=1.3),
                    address_field,
                    connect_btn,
                    error_text,
                ], spacing=10),
                ft.Row([ft.Icon(ft.Icons.SHIELD_OUTLINED, color=DIM, size=13),
                        T("Every connection — automatic or manual — still goes through the "
                          "same identity check and end-to-end encryption.", size=10.5,
                          color=DIM, expand=True, height=1.3)], spacing=8),
            ], spacing=16, tight=True),
            padding=ft.Padding.only(left=20, right=20, top=8, bottom=24))

        return ft.BottomSheet(content=content, bgcolor=SURFACE, show_drag_handle=True,
                              use_safe_area=True, scrollable=True, on_dismiss=on_dismiss)

    def grade_badge(grade, size=64):
        col = GREEN if grade.startswith("A") else (CYAN if grade.startswith("B") else AMBER)
        return ft.Container(H(grade, size=size * 0.42, color=col), width=size, height=size,
                            border_radius=size * 0.3, alignment=ft.Alignment.CENTER,
                            bgcolor=ahex(col, "14"), border=ft.Border.all(1.5, ahex(col, "66")),
                            shadow=glow(col, 18, opacity="33"))

    def mini_bars(scores, height=40):
        if not scores:
            return ft.Container(height=4)
        bars = []
        for s in scores[-20:]:
            col = GREEN if s >= 88 else (AMBER if s >= 70 else RED)
            bars.append(ft.Container(width=7, height=max(4, height * s / 100), bgcolor=col,
                                     border_radius=3))
        return ft.Row(bars, spacing=3, vertical_alignment=ft.CrossAxisAlignment.END, height=height)

    # =====================================================================
    # BOOT
    # =====================================================================
    def build_boot() -> ft.View:
        mark = logo_mark(88)
        mark.scale, mark.opacity = 0.6, 0
        mark.animate_scale = ft.Animation(700, ft.AnimationCurve.EASE_OUT_BACK)
        mark.animate_opacity = ft.Animation(500, ft.AnimationCurve.EASE_OUT)
        progress = ft.ProgressBar(value=0.05, color=GREEN, bgcolor=SURFACE_3, bar_height=4,
                                  border_radius=4)
        status = T("Starting…", size=12.5, color=MUTED, align=ft.TextAlign.CENTER)

        async def sequence():
            await asyncio.sleep(0.15)
            mark.scale, mark.opacity = 1.0, 1.0
            page.update()
            conn.start()
            t0 = time.time()
            while not conn.online and conn.state != "untrusted" and time.time() - t0 < 14:
                status.value = conn.status_text
                progress.value = min(0.9, 0.1 + (time.time() - t0) / 16)
                page.update()
                await asyncio.sleep(0.2)
            if conn.online:
                state.catalog = ex.merge_catalog((conn.info or {}).get("catalog", []))
                state.roadmap = (conn.info or {}).get("roadmap", [])
                status.value = ("Paired — safety words: " + conn.safety_words
                                if (conn.info or {}).get("first_time")
                                else "Secure session ready")
            elif conn.state == "untrusted":
                status.value = "Coach identity doesn't match — check Privacy"
            else:
                status.value = "No coach found yet — still looking"
            progress.value = 1.0
            page.update()
            await asyncio.sleep(0.7)
            if ui["route"] == "/":
                navigate("/home")

        page.run_task(sequence)

        content = ft.Column([
            ft.Container(expand=3),
            ft.Row([mark], alignment=ft.MainAxisAlignment.CENTER),
            ft.Container(height=16),
            ft.Column([H("RepFlow", size=44), eyebrow("your video never leaves the room", size=11)],
                      horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
            ft.Container(expand=2),
            progress,
            ft.Container(height=12),
            status,
            ft.Container(height=24),
        ], horizontal_alignment=ft.CrossAxisAlignment.STRETCH, expand=True, spacing=0)
        return view("/", ft.SafeArea(ft.Container(content, padding=28, expand=True), expand=True))

    # =====================================================================
    # HOME
    # =====================================================================
    def build_home() -> ft.View:
        chip, chip_refresh = link_chip()
        net_icon, net_refresh = network_icon_button()

        def refresh_both():
            chip_refresh()
            net_refresh()

        listen(refresh_both)

        hist = state.history
        total = sum(h.get("reps", 0) for h in hist)
        best = max((h.get("score", 0) for h in hist), default=0)
        streak = day_streak(hist)

        header = ft.Row([
            ft.Column([T(greeting(), size=13, color=MUTED), H("Ready to train?", size=26)],
                      spacing=2, tight=True),
            ft.Row([
                net_icon,
                ft.Container(ft.Icon(ft.Icons.SHIELD_OUTLINED, color=GREEN, size=20), width=42,
                             height=42, border_radius=14, bgcolor=SURFACE_2,
                             border=ft.Border.all(1, BORDER_2), alignment=ft.Alignment.CENTER,
                             on_click=lambda e: navigate("/privacy"), ink=True,
                             tooltip="Privacy & security"),
            ], spacing=8),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        pres = state.presentation()
        hero = ft.Container(
            ft.Column([
                ft.Row([pill([dot(GREEN, 7), T("LIVE AI COACHING", size=10, font=S_FONT,
                                               color=GREEN, spacing=1.5)])], spacing=8),
                H(pres["name"], size=30),
                T(pres.get("tagline", ""), size=13.5, color="#C9D6EA", height=1.35),
                neon_button("START SESSION", ft.Icons.ARROW_FORWARD_ROUNDED,
                            on_click=lambda e: navigate("/setup")),
            ], spacing=14),
            padding=20, border_radius=24,
            gradient=ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
                                       colors=["#0E2A22", "#0B1530", "#0A0F1E"]),
            border=ft.Border.all(1, "#1F4A3A"), shadow=glow(GREEN, blur=40, opacity="22"),
            opacity=0, scale=0.96,
            animate_opacity=ft.Animation(420, ft.AnimationCurve.EASE_OUT),
            animate_scale=ft.Animation(420, ft.AnimationCurve.EASE_OUT_CUBIC))

        stats = ft.Row([
            stat_tile("streak", H(str(streak), size=26), ft.Icons.LOCAL_FIRE_DEPARTMENT, AMBER,
                      sub=T("day" if streak == 1 else "days", size=10.5, color=DIM)),
            stat_tile("reps", H(str(total), size=26), ft.Icons.REPEAT, CYAN,
                      sub=T("all time", size=10.5, color=DIM)),
            stat_tile("best", H(str(best) if best else "—", size=26),
                      ft.Icons.EMOJI_EVENTS_OUTLINED, VIOLET,
                      sub=T("form score", size=10.5, color=DIM)),
        ], spacing=10)
        stats_wrap = ft.Container(stats, opacity=0, offset=ft.Offset(0, 0.08),
                                  animate_opacity=ft.Animation(420, ft.AnimationCurve.EASE_OUT),
                                  animate_offset=ft.Animation(420, ft.AnimationCurve.EASE_OUT_CUBIC))

        async def _entrance():
            await asyncio.sleep(0.05)
            hero.opacity, hero.scale = 1, 1.0
            page.update()
            await asyncio.sleep(0.08)
            stats_wrap.opacity, stats_wrap.offset = 1, ft.Offset(0, 0)
            page.update()

        page.run_task(_entrance)

        controls = [header, ft.Row([chip]), hero, stats_wrap]

        if hist:
            last = hist[-1]
            controls.append(section_title("Last set"))
            controls.append(card(ft.Row([
                grade_badge(last.get("grade", "—"), 58),
                ft.Column([
                    T(f"{last.get('label', '')}", size=15, font=S_FONT),
                    T(f"Form score {last.get('score', 0)} · {last.get('date', '')[11:16]}",
                      size=12, color=MUTED),
                    mini_bars(last.get("scores", []), height=46),
                ], spacing=6, expand=True),
            ], spacing=14), padding=14))

        controls.append(section_title("Choose a movement"))
        for entry in state.catalog:
            controls.append(movement_row(entry))
        for entry in ex.roadmap_catalog(state.roadmap):
            controls.append(locked_movement_row(entry))

        controls.append(ft.Container(height=8))

        return view("/home", ft.SafeArea(
            ft.Container(ft.Column(controls, spacing=16, scroll=ft.ScrollMode.AUTO, expand=True),
                         padding=ft.Padding.only(left=18, right=18, top=12), expand=True),
            expand=True))

    def movement_row(entry: dict):
        key = entry["key"]
        selected = state.exercise == key

        def pick(e):
            state.exercise = key
            state.target = entry["goals"][1][0] if len(entry["goals"]) > 1 else 0
            navigate("/home")

        return card(ft.Row([
            icon_badge(entry.get("icon", ft.Icons.SPORTS_GYMNASTICS),
                       GREEN if selected else CYAN, 42, 20),
            ft.Column([T(entry["name"], size=14.5, font=S_FONT),
                       T(entry.get("tagline", ""), size=11.5, color=MUTED)],
                      spacing=2, expand=True, tight=True),
            ft.Icon(ft.Icons.CHECK_CIRCLE if selected else ft.Icons.CHEVRON_RIGHT,
                    color=GREEN if selected else DIM, size=20),
        ], spacing=12), padding=12, radius=16,
            bgcolor=SURFACE_2 if selected else SURFACE,
            border=GREEN if selected else BORDER, on_click=pick)

    def locked_movement_row(entry: dict):
        """A future-roadmap movement, teased but not selectable: no tap
        handler at all (not even a disabled-look one). The row content is
        dimmed to ~45% so it reads as "not yet" at a glance, but the badge
        is deliberately kept outside that dimming and rendered at full
        strength — a locked card should invite curiosity, not look broken
        or like a rendering glitch, which is what a uniformly faded card
        with a faded label reads as."""
        faded = ft.Container(
            ft.Row([
                icon_badge(entry.get("icon", ft.Icons.SPORTS_GYMNASTICS), MUTED, 42, 20),
                ft.Column([T(entry["name"], size=14.5, font=S_FONT, color=MUTED),
                           T(entry.get("note", ""), size=11.5, color=DIM)],
                          spacing=2, expand=True, tight=True),
                ft.Container(width=104),   # keeps text from running under the pinned badge
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            opacity=0.45)
        return ft.Stack([
            ft.Container(faded, padding=12, border_radius=16, bgcolor=SURFACE,
                        border=ft.Border.all(1, BORDER)),
            ft.Container(
                pill([ft.Icon(ft.Icons.LOCK_OUTLINE, size=11, color=DIM),
                      T("COMING SOON", size=9.5, font=S_FONT, color=DIM, spacing=1.1)],
                     color=DIM, padding=(9, 5)),
                right=12, top=0, bottom=0, alignment=ft.Alignment.CENTER_RIGHT),
        ])

    # =====================================================================
    # PRIVACY
    # =====================================================================
    def build_privacy() -> ft.View:
        return build_privacy_view(page, state, conn, client, navigate, view, on_ui)

    # (Setup / Live / Report live in screens.py; Privacy in privacy.py)
    build_setup, build_live, build_summary = _screens(
        page, state, client, conn, ui, navigate, view, link_chip, listen, play, haptic,
        services, is_mobile, grade_badge, mini_bars, on_ui)

    navigate("/")


# =========================================================================
from privacy import build_privacy_view  # noqa: E402
from screens import _screens            # noqa: E402

if __name__ == "__main__":
    ft.run(main, assets_dir="assets")
