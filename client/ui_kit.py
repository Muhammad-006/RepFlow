"""
ui_kit.py - RepFlow design system: colors, type, and reusable widgets.
"""

from __future__ import annotations

import flet as ft

# ---------------------------------------------------------------- palette
BG = "#05070D"
SURFACE = "#0B1120"
SURFACE_2 = "#0F172A"
SURFACE_3 = "#131D33"
BORDER = "#1B2740"
BORDER_2 = "#243452"
TEXT = "#F1F5F9"
MUTED = "#8A9BB8"
DIM = "#56657F"
GREEN = "#00FF66"
GREEN_DEEP = "#00C853"
GREEN_GLOW = "#0B2A1A"
CYAN = "#22E5FF"
AMBER = "#F59E0B"
RED = "#EF4444"
VIOLET = "#A78BFA"
INK = "#04110A"          # text on neon buttons



def ahex(color: str, alpha: str) -> str:
    """Flet/Flutter 8-digit hex is #AARRGGBB -- put the alpha first."""
    return f"#{alpha}{color.lstrip('#')[-6:]}"


# ---------------------------------------------------------------- type
FONTS = {
    "Grotesk": "fonts/SpaceGrotesk-Bold.ttf",
    "GroteskMed": "fonts/SpaceGrotesk-Medium.ttf",
    "Inter": "fonts/Inter-Regular.ttf",
    "InterSemi": "fonts/Inter-SemiBold.ttf",
    "Mono": "fonts/JetBrainsMono-Bold.ttf",
}

H_FONT = "Grotesk"
B_FONT = "Inter"
S_FONT = "InterSemi"
M_FONT = "Mono"


def T(value="", size=14, color=TEXT, font=B_FONT, weight=None, spacing=None, align=None,
      italic=False, max_lines=None, height=None, **kw) -> ft.Text:
    style = ft.TextStyle(letter_spacing=spacing, height=height) if (spacing or height) else None
    return ft.Text(value, size=size, color=color, font_family=font, weight=weight, style=style,
                   text_align=align, italic=italic, max_lines=max_lines, **kw)


def H(value, size=26, color=TEXT, **kw) -> ft.Text:
    return T(value, size=size, color=color, font=H_FONT, **kw)


def eyebrow(value, color=GREEN, size=10.5) -> ft.Text:
    return T(value.upper(), size=size, color=color, font=S_FONT, spacing=2.2)


def mono(value, size=14, color=TEXT, **kw) -> ft.Text:
    return T(value, size=size, color=color, font=M_FONT, **kw)


# ---------------------------------------------------------------- surfaces
def card(content, padding=16, radius=18, bgcolor=SURFACE, border=BORDER, gradient=None,
         width=None, height=None, on_click=None, expand=None, shadow=None, **kw) -> ft.Container:
    return ft.Container(
        content=content, padding=padding, border_radius=radius,
        bgcolor=None if gradient else bgcolor, gradient=gradient,
        border=ft.Border.all(1, border) if border else None,
        width=width, height=height, on_click=on_click, expand=expand, shadow=shadow,
        ink=bool(on_click), **kw,
    )


def glow(color=GREEN, blur=24, spread=0, opacity="44") -> list:
    return [ft.BoxShadow(blur_radius=blur, spread_radius=spread, color=ahex(color, opacity))]


def pill(controls, color=GREEN, bg=None, border=None, padding=(10, 5)) -> ft.Container:
    return ft.Container(
        content=ft.Row(controls if isinstance(controls, list) else [controls], tight=True, spacing=6,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.Padding.symmetric(horizontal=padding[0], vertical=padding[1]),
        border_radius=99, bgcolor=bg or ahex(color, "14"),
        border=ft.Border.all(1, border or ahex(color, "55")),
    )


def dot(color=GREEN, size=8) -> ft.Container:
    return ft.Container(width=size, height=size, border_radius=size, bgcolor=color,
                        shadow=[ft.BoxShadow(blur_radius=8, color=ahex(color, "AA"))])


def neon_button(label, icon=None, on_click=None, width=None, height=58,
                colors=(GREEN, "#00E0A0"), text_color=INK, expand=None) -> ft.Container:
    row = [T(label, size=15, color=text_color, font=H_FONT, spacing=1.2)]
    if icon:
        row.append(ft.Icon(icon, color=text_color, size=20))
    return ft.Container(
        content=ft.Row(row, alignment=ft.MainAxisAlignment.CENTER, spacing=10),
        height=height, width=width, expand=expand, border_radius=16,
        gradient=ft.LinearGradient(begin=ft.Alignment.CENTER_LEFT, end=ft.Alignment.CENTER_RIGHT,
                                   colors=list(colors)),
        shadow=glow(colors[0], blur=26, opacity="55"),
        on_click=on_click, ink=True,
    )


def ghost_button(label, icon=None, on_click=None, width=None, height=52, color=TEXT,
                 border=BORDER_2, expand=None) -> ft.Container:
    row = []
    if icon:
        row.append(ft.Icon(icon, color=color, size=18))
    row.append(T(label, size=14, color=color, font=S_FONT, spacing=0.6))
    return ft.Container(
        content=ft.Row(row, alignment=ft.MainAxisAlignment.CENTER, spacing=8),
        height=height, width=width, expand=expand, border_radius=14, bgcolor=SURFACE_2,
        border=ft.Border.all(1, border), on_click=on_click, ink=True,
    )


def stat_tile(label, value_ctrl, icon=None, color=GREEN, expand=True, sub=None) -> ft.Container:
    head = [eyebrow(label, color=MUTED, size=9.5)]
    if icon:
        head.insert(0, ft.Icon(icon, size=13, color=color))
    col = [ft.Row(head, spacing=6), value_ctrl]
    if sub is not None:
        col.append(sub)
    return card(ft.Column(col, spacing=6, tight=True), padding=14, radius=16, bgcolor=SURFACE_2, expand=expand)


def icon_badge(icon, color=GREEN, size=40, icon_size=20) -> ft.Container:
    return ft.Container(
        content=ft.Icon(icon, color=color, size=icon_size), width=size, height=size,
        border_radius=size * 0.32, bgcolor=ahex(color, "18"), border=ft.Border.all(1, ahex(color, "40")),
        alignment=ft.Alignment.CENTER,
    )


def logo_mark(size=44) -> ft.Container:
    return ft.Container(
        content=ft.Image(src="logo.png", width=size, height=size, fit=ft.BoxFit.CONTAIN),
        width=size, height=size, border_radius=size * 0.3,
        shadow=glow(GREEN, blur=28, opacity="55"),
    )


def screen(controls, padding=18, scroll=True, spacing=16) -> ft.SafeArea:
    body = ft.Column(controls, spacing=spacing, scroll=ft.ScrollMode.AUTO if scroll else None,
                     expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
    return ft.SafeArea(content=ft.Container(body, padding=padding, expand=True), expand=True)


def background() -> ft.LinearGradient:
    return ft.LinearGradient(begin=ft.Alignment.TOP_CENTER, end=ft.Alignment.BOTTOM_CENTER,
                             colors=["#07110E", BG, BG])


def section_title(title, trailing=None) -> ft.Row:
    row = [T(title, size=16, font=H_FONT)]
    if trailing is not None:
        row.append(trailing)
    return ft.Row(row, alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
