"""
config.py — how the app finds and talks to the coach server.

You don't normally touch this. The app discovers the coach by itself
(discovery.py): same Wi-Fi, laptop hotspot, phone hotspot, or USB
(`adb reverse tcp:8765 tcp:8765`). Nobody types an address, and there is
no field in the UI to type one into.

As a head start, `build_apk.bat` / `build_apk.sh` bake the build
machine's current IP addresses into `build_config.py`; those are tried
first, then everything else.

**Edge routing.** If more than one coach server is reachable at once — a
laptop on the local network *and* one or more always-on cloud servers in
`EDGE_SERVERS` — the app doesn't guess which to use. It races GET
`/v1/health` against every one of them at once and ranks every healthy
one by round-trip latency (see `discovery.race_and_rank`), then connects
to the nearest first. If that connection drops or fails to complete its
handshake, it fails over *instantly* to the next-ranked node rather than
starting discovery over — see `EDGE_ROUTING.md` for the full cascade
design. Nobody picks a region for this to work; it stays invisible unless
someone opens the manual-override panel (the network icon on Home), which
is the one place a person can force a specific address instead.

Developer override on desktop:
    REPFLOW_SERVER_HOST=192.168.1.20 python main.py
    REPFLOW_EDGE_SERVERS="us-east@api-us.example.com:443:1:Ashburn,eu@api-eu.example.com:443:1:Frankfurt" python main.py
"""

import os

SERVER_PORT: int = int(os.environ.get("REPFLOW_SERVER_PORT", "8765"))

try:
    from build_config import BAKED_HOSTS  # generated at APK build time
except Exception:
    BAKED_HOSTS = []

try:
    from build_config import EDGE_SERVERS as _BAKED_EDGE_SERVERS  # generated at APK build time
except Exception:
    _BAKED_EDGE_SERVERS = []

PREFERRED_HOSTS = [h for h in [os.environ.get("REPFLOW_SERVER_HOST", "")] + list(BAKED_HOSTS) if h]


def _parse_edge_servers(raw: str) -> list:
    """`name@host:port:https:city` entries, comma-separated. `https` is
    `1`/`0`, defaults to 1 (public internet servers must opt out of TLS
    explicitly, not opt in — see the strict-protocol note in
    discovery.py). `city` is optional and cosmetic only: it's what the
    manual-override modal shows for "connected server location" instead of
    ever printing a raw hostname — it plays no part in routing, which is
    still decided purely by measured round-trip latency."""
    out = []
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            name, rest = entry.split("@", 1)
            parts = rest.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 and parts[1] else 443
            https = bool(int(parts[2])) if len(parts) > 2 and parts[2] != "" else True
            city = parts[3] if len(parts) > 3 and parts[3] != "" else ""
            out.append({"name": name, "host": host, "port": port, "https": https, "city": city})
        except (ValueError, IndexError):
            continue
    return out


# Always-on cloud servers this build knows about, e.g. one per AWS region.
# Baked at build time (`build_config.EDGE_SERVERS`) or set for local dev via
# REPFLOW_EDGE_SERVERS. Empty by default — a build with none of these still
# works exactly as before, discovering a LAN coach only.
EDGE_SERVERS = _parse_edge_servers(os.environ.get("REPFLOW_EDGE_SERVERS", "")) + list(_BAKED_EDGE_SERVERS)

# Public-internet connections must use HTTPS; only hosts on a private
# network (LAN discovery, hotspot, USB) may use plain HTTP. See
# discovery.is_private_host.
REQUIRE_HTTPS_FOR_PUBLIC_HOSTS: bool = True

# Per-request timeout. Kept short on purpose: a frame that fails should
# fail fast, because the next camera frame is 100 ms away and retrying
# with fresh data beats waiting on stale data.
REQUEST_TIMEOUT: float = 4.0

# If frames keep failing for this long, re-run discovery — the coach's IP
# may have changed when someone switched networks.
REDISCOVER_AFTER_S: float = 6.0

# Camera resolution sent to the coach: "low" (~320x240), "medium"
# (~640x480) or "high".
#
# "medium" is the right default. The pose model letterboxes everything to
# 640x640 anyway, so "high" buys no accuracy and costs real milliseconds
# in JPEG encode, encryption and upload. Drop to "low" if you are on a
# congested network or a very old phone; expect the skeleton to get
# noisier on small joints (wrists, ankles) before anything else breaks.
CAMERA_PRESET: str = os.environ.get("REPFLOW_CAMERA_PRESET", "medium")
