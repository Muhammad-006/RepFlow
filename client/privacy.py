"""
privacy.py — the screen that shows what RepFlow does with your video.

Most fitness apps have a privacy policy. This one has a privacy *screen*:
the same claims, but wired to the live session so the numbers move while
you use it, and so a claim that stopped being true would visibly stop
being true.

Three parts:

  * **The link.** Which coach this phone is paired with, its safety words,
    and — if the identity ever changes — a red banner instead of a silent
    re-pair.
  * **The guarantees.** Plain sentences about what leaves the phone, each
    one paired with the mechanism that enforces it rather than a promise.
  * **The receipt.** After a set, the exact counts: frames encrypted,
    frames stored (zero), bytes written to disk (zero), keys destroyed.
"""

from __future__ import annotations

import flet as ft

from ui_kit import (AMBER, BORDER, BORDER_2, CYAN, DIM, GREEN, H_FONT, MUTED, RED, S_FONT,
                    SURFACE, SURFACE_2, TEXT, VIOLET, H, T, ahex, card, eyebrow, ghost_button,
                    icon_badge, mono, section_title)

GUARANTEES = [
    (ft.Icons.LOCK, "Frames are encrypted on the phone",
     "A fresh key is agreed for every session and never leaves the two devices. "
     "Anyone watching the Wi-Fi sees opaque bytes — not your video, not even which "
     "exercise you picked or how your phone is propped."),
    (ft.Icons.VERIFIED_USER, "The coach is verified, not just trusted",
     "This phone remembers the coach's identity key the first time you pair. If "
     "something else answers later, the session stops instead of quietly connecting."),
    (ft.Icons.DELETE_SWEEP, "Nothing is stored, anywhere",
     "The server decrypts a frame into memory, measures your joints and drops it. "
     "There is no code in it that can write an image to disk — not a cache, not a "
     "log, not a debug folder."),
    (ft.Icons.VIDEOCAM_OFF, "There is no video feed to watch",
     "Earlier versions had a live dashboard you could open in a browser. It's gone. "
     "The server has no web page at all."),
    (ft.Icons.PERSON_OFF, "No account, no name, no device model",
     "Sessions are a random number generated when you open the app. Your phone's "
     "model and IP are never recorded, and nothing links one set to the next."),
    (ft.Icons.KEY_OFF, "Keys are destroyed when the set ends",
     "Each session's keys are overwritten at the end. Even a recording of the whole "
     "network conversation can't be decrypted afterwards — by anyone, including us."),
]


def build_privacy_view(page, state, conn, client, navigate, view, on_ui) -> ft.View:
    def top_bar():
        return ft.Row([
            ft.Container(ft.Icon(ft.Icons.ARROW_BACK_ROUNDED, color=TEXT, size=20), width=40,
                         height=40, border_radius=12, bgcolor=SURFACE_2,
                         border=ft.Border.all(1, BORDER), alignment=ft.Alignment.CENTER,
                         on_click=lambda e: navigate("/home"), ink=True),
            H("Privacy & security", size=19),
            ft.Container(width=40),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

    # ---- the link ----------------------------------------------------
    def link_card():
        untrusted = conn.state == "untrusted"
        col = RED if untrusted else (GREEN if conn.online else AMBER)
        rows = []

        if untrusted:
            rows.append(ft.Container(
                ft.Column([
                    ft.Row([ft.Icon(ft.Icons.GPP_BAD, color=RED, size=18),
                            T("Identity check failed", size=14, font=S_FONT, color=RED)], spacing=8),
                    T(conn.security_warning or "This coach's identity key is not the one this "
                      "phone paired with before.", size=12, color="#F5C2C2", height=1.35),
                    T("That can mean the coach server was reinstalled — or that something else "
                      "is answering in its place. Only re-pair if you know why it changed.",
                      size=11.5, color=MUTED, height=1.35),
                    ghost_button("Re-pair with this coach", ft.Icons.LINK_OFF, height=44,
                                 color=RED, border=ahex(RED, "88"),
                                 on_click=lambda e: (conn.forget_coach(), navigate("/privacy"))),
                ], spacing=10),
                padding=14, border_radius=14, bgcolor=ahex(RED, "12"),
                border=ft.Border.all(1, ahex(RED, "55"))))

        words = conn.safety_words or "—"
        rows += [
            ft.Row([T("Status", size=12, color=MUTED),
                    T("Verified & encrypted" if conn.online else conn.status_text,
                      size=12, font=S_FONT, color=col)],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([T("Round trip", size=12, color=MUTED),
                    mono(f"{conn.rtt_ms:.0f} ms" if conn.rtt_ms else "—", size=12)],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Divider(height=1, color=BORDER),
            T("SAFETY WORDS", size=9.5, font=S_FONT, color=DIM, spacing=1.6),
            ft.Container(mono(words, size=15, color=CYAN), padding=12, border_radius=12,
                         bgcolor="#060A14", border=ft.Border.all(1, ahex(CYAN, "33")),
                         alignment=ft.Alignment.CENTER),
            T("Your coach server prints these same four words when it starts. If they match, "
              "there is nothing sitting in between.", size=11.5, color=MUTED, height=1.35),
        ]
        return card(ft.Column([
            ft.Row([ft.Row([icon_badge(ft.Icons.SHIELD_OUTLINED, col, 34, 17),
                            T("This link", size=15, font=S_FONT)], spacing=10)]),
            *rows,
        ], spacing=10), padding=16, border=ahex(col, "44"))

    # ---- guarantees ---------------------------------------------------
    def guarantee_rows():
        out = []
        for icon, title, body in GUARANTEES:
            out.append(ft.Row([
                ft.Container(ft.Icon(icon, color=GREEN, size=17), width=32, height=32,
                             border_radius=10, bgcolor=ahex(GREEN, "14"),
                             alignment=ft.Alignment.CENTER),
                ft.Column([T(title, size=13.5, font=S_FONT),
                           T(body, size=11.5, color=MUTED, height=1.4)],
                          spacing=3, expand=True, tight=True),
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START))
        return out

    # ---- receipt --------------------------------------------------------
    def receipt_card():
        r = state.receipt or {}
        if not r:
            return card(ft.Column([
                ft.Row([icon_badge(ft.Icons.RECEIPT_LONG, VIOLET, 34, 17),
                        T("Session receipt", size=15, font=S_FONT)], spacing=10),
                T("Finish a set and the exact numbers show up here: how many frames were "
                  "encrypted and sent, how many were kept, and when the keys were destroyed.",
                  size=11.5, color=MUTED, height=1.4),
            ], spacing=10), padding=16)

        def line(label, value, good=True):
            return ft.Row([T(label, size=12, color=MUTED),
                           mono(str(value), size=12.5, color=GREEN if good else AMBER)],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        return card(ft.Column([
            ft.Row([icon_badge(ft.Icons.RECEIPT_LONG, VIOLET, 34, 17),
                    T("Last session receipt", size=15, font=S_FONT)], spacing=10),
            line("Frames encrypted and sent", r.get("frames_received", 0)),
            line("Frames decrypted for analysis", r.get("frames_decrypted", 0)),
            line("Data sent", f"{r.get('mb_received', 0)} MB"),
            line("Rejected (tampered or replayed)", r.get("frames_rejected", 0),
                 good=not r.get("frames_rejected")),
            ft.Divider(height=1, color=BORDER),
            line("Frames stored on the server", r.get("frames_stored", 0)),
            line("Bytes written to disk", r.get("bytes_written_to_disk", 0)),
            line("Average time a frame existed decrypted",
                 f"{r.get('avg_plaintext_lifetime_ms', 0)} ms"),
            line("Session keys destroyed", "yes" if r.get("keys_destroyed") else "on close"),
        ], spacing=8), padding=16, border=ahex(VIOLET, "33"))

    # ---- how it works ----------------------------------------------------
    def mechanism_card():
        steps = [
            ("1", "Handshake", "X25519 key agreement, twice: once against the coach's long-term "
                               "identity key to prove who it is, once against a throw-away key so "
                               "today's session can't be decrypted tomorrow."),
            ("2", "Seal", "Each frame is encrypted with a keyed BLAKE2b keystream and signed with "
                          "HMAC-SHA256 over the ciphertext, the nonce and the request path."),
            ("3", "Verify", "The server checks the signature before it decrypts anything. A frame "
                            "that was tampered with never reaches the decoder."),
            ("4", "Process", "Decrypt into memory, run the pose model, drop the image. Roughly "
                             "seventy milliseconds of existence, then gone."),
            ("5", "Destroy", "End the set and both sides overwrite their keys."),
        ]
        rows = []
        for n, title, body in steps:
            rows.append(ft.Row([
                ft.Container(T(n, size=11, font=S_FONT, color="#04110A"), width=22, height=22,
                             border_radius=11, bgcolor=CYAN, alignment=ft.Alignment.CENTER),
                ft.Column([T(title, size=13, font=S_FONT),
                           T(body, size=11.5, color=MUTED, height=1.4)],
                          spacing=2, expand=True, tight=True),
            ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.START))
        return card(ft.Column([
            ft.Row([icon_badge(ft.Icons.ENHANCED_ENCRYPTION, CYAN, 34, 17),
                    T("How a frame travels", size=15, font=S_FONT)], spacing=10),
            *rows,
        ], spacing=12), padding=16)

    body = ft.Column([
        top_bar(),
        link_card(),
        card(ft.Column([
            ft.Row([icon_badge(ft.Icons.POLICY_OUTLINED, GREEN, 34, 17),
                    T("What we promise", size=15, font=S_FONT)], spacing=10),
            *guarantee_rows(),
        ], spacing=14), padding=16),
        receipt_card(),
        mechanism_card(),
        T("RepFlow keeps no account, no cloud storage and no analytics. Your set history lives "
          "only on this phone and goes away with the app.", size=11, color=DIM, height=1.4),
        ft.Container(height=10),
    ], spacing=14, scroll=ft.ScrollMode.AUTO, expand=True)

    return view("/privacy", ft.SafeArea(
        ft.Container(body, padding=ft.Padding.only(left=18, right=18, top=10), expand=True),
        expand=True))
