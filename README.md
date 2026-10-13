# RepFlow — an AI form coach that keeps nothing

RepFlow watches your push-ups, squats and planks through your phone
camera, counts every rep, scores your form and talks you through the set
in real time. The pose model runs on a machine you control; the phone is
the camera, the screen and the voice.

Every frame is encrypted on the phone before it leaves, decrypted only in
the coach's memory, and dropped about seventy milliseconds later. Nothing
is written to disk. There is no web page to watch you on. When the set
ends, the keys are destroyed and the app shows you the receipt.

```
 ┌───────────── PHONE (APK) ─────────────┐        ┌──────────── COACH SERVER ────────────┐
 │ camera → JPEG                          │        │                                      │
 │   ↓ sealed (X25519 + BLAKE2b + HMAC)  ├── ✱ ──▶│ verify → decrypt in RAM → YOLO11-Pose │
 │ live HUD · skeleton · voice cues       │        │   → per-exercise coach → 0-100 score  │
 │ pinned server identity + safety words  │◀── ✱ ──┤ ← sealed coaching JSON                │
 │ privacy receipt                        │        │ frame dropped · nothing stored        │
 └────────────────────────────────────────┘        └───────────────────────────────────────┘
        finds the coach by itself                   backend only: no dashboard, no video feed
        (UDP discovery + network scan)
```

**Three coaches, live:** Push-Up · Squat · Plank
**Next:** Pull-Up · Dip · Lunge

---

## Quick start

**1. Start the coach server**

| Windows | macOS / Linux |
|---|---|
| double-click **`START_SERVER.bat`** | `./start_server.sh` |

First run installs three small packages (numpy, opencv, onnxruntime) into
`server/.venv`. The pose model ships inside the repo, so nothing is
downloaded at the venue.

Keep the four safety words visible — the app shows the same four, and
matching them is how you know nothing is sitting in between.

On Windows, run **`ALLOW_FIREWALL.bat`** once (or click *Allow* on both
Private **and** Public when Windows asks about Python).

**2. Build the APK (only when the phone app itself changes)**

| Windows | macOS / Linux |
|---|---|
| double-click **`BUILD_APK.bat`** | `./build_apk.sh` |

The first build downloads Flutter and the Android SDK automatically
(10–20 min). The APK lands in `client/build/apk/`.

**3. Put the phone and the coach machine on the same network** — any of:

* the same Wi-Fi, or
* the laptop's hotspot (Windows: *Settings → Network → Mobile hotspot*), or
* the phone's hotspot (connect the laptop to it), or
* a USB cable: `adb reverse tcp:8765 tcp:8765`

Open RepFlow. It finds the coach by itself, runs the key exchange, and
shows *Encrypted*. Pick a movement, tap **Begin**.

---

## What's in the box

```
RepFlow-app/
├── client/                    the phone app (Flet → Android APK)
│   ├── main.py                boot, home, navigation, services
│   ├── screens.py             setup, live coaching, report
│   ├── privacy.py             the privacy screen + session receipt
│   ├── spotter_client.py      encrypted transport
│   ├── crypto_box.py          ← identical copy of the server's
│   ├── connection.py          discovery, pairing, identity pinning
│   ├── exercises.py           how each movement is presented
│   ├── cues.py                cue id → on-screen text + audio clip
│   ├── discovery.py           finds the coach with no IP typed
│   ├── config.py              ports, timeouts, camera preset
│   ├── ui_kit.py              colours, type, components
│   └── assets/audio/          21 pre-rendered voice cues
│
├── server/                    the coach (backend only)
│   ├── server.py              HTTP: handshake, frames, summary, end
│   ├── crypto_box.py          X25519 · HKDF · BLAKE2b-CTR · HMAC
│   ├── session.py             one athlete: keys, replay window, ledger
│   ├── biomech.py             orientation-free joint geometry
│   ├── pose_engine.py         YOLO11-Pose via ONNX
│   ├── feedback.py            when the coach is allowed to speak
│   ├── cues.py                the coach's whole vocabulary
│   ├── models/                the 192 px tracker and 256 px finder
│   └── exercises/
│       ├── base.py            the coach contract + the motion gate
│       ├── pushup.py          depth · hip line · lockout · tempo
│       ├── squat.py           depth · knee tracking · forward fold
│       └── plank.py           body line · stability · honest hold time
│
└── tools/write_build_config.py   bakes this machine's IPs into the APK
```

This tree was cut down to what is needed to run and build the app. The
test suites, the Docker/nginx/AWS deployment kit, the voice-rendering and
tracking-verification tooling, and the long-form design documents
(`SECURITY.md`, `EDGE_ROUTING.md`, `ML_MODEL_COMPARISON.md`,
`TRACKING_FIXES.md`, `AWS_DEPLOYMENT.md`, `DEMO_GUIDE.md`) were removed.
They all still exist in the v3.1.3 zip if any of them is wanted back —
nothing was rewritten to make the removal possible, so they can simply be
copied across.

---

## v3.2.0 — why the counter stayed on zero

Everything in this release is one bug with several faces: **the coaching
layer measured time in frames, and at a slow round trip a frame is no
longer a tenth of a second.**

The phone keeps exactly one frame in flight at a time — it cannot capture
the next one until the reply to the last one arrives. So the frame rate
the coach actually sees is `1 / round-trip-time`. At the 700 ms–1 s round
trips being reported, that is roughly **one frame per second**, ten times
sparser than every constant in the coaching layer was tuned for.

**1. Nothing calibrated below about 1.5 fps, so nothing counted — ever.**
The motion gate decides whether the athlete is holding still, and all
three coaches refuse to finish calibrating until it says yes. It kept a
trailing *time* window but needed three samples to form an opinion, and
trimming the window purely by age left it holding **two** as soon as
frames arrived more than half a window apart. So below ~1.5 fps
`is_still` could not return True no matter how motionless you were. Not
"counted badly" — the set never left calibration, so not a single rep or
second was ever counted. The window is a maximum age now, never a reason
to keep fewer samples than the measurement needs.

**2. The plank clock ran slow, then stopped.** It credited at most 0.5 s
of hold time per frame however much real time had passed, so at one frame
per second a 30-second plank was counted as 15 — and with fault 1 in play
it counted 0. It now credits the real gap between two frames that both
show a good position, up to 2.5 s, and nothing beyond that.

**3. The plank clock could also run while nobody was there.** The angle
smoother returns its previous value when a frame produces none, so a
stale reading looked exactly like a steady hold. Only a fresh reading
moves the clock now. Short tracking dropouts are still bridged; long ones
are not paid for.

**4. The wobble measure described the whole set instead of the moment.**
Its 25-sample window meant ~2.5 s at design cadence and 25 *seconds* at
one frame per second, so a plank that started clean and sagged late still
read as rock solid. It is a real-time window now, and it forms an opinion
after three samples instead of five.

**5. A blank skeleton for up to eight seconds.** Frames that arrived late
were refused so the phone would "send a fresh one" — but the phone cannot
send a fresh one until it gets this reply, so the refusal just cost a
whole round trip and showed nothing. Up to eight in a row were allowed.
The server now recognises when the phone is round-trip-bound and coaches
from the frame it has, never drops during a hold, and caps a run of drops
at two. The cue layer already refused to *speak* from stale data, so this
gives up nothing that was actually protecting anyone.

**6. A latency breakdown, so the next round is data.** During a set the
server prints, every 40 frames, which leg of the round trip owns the
time:

```
latency 910 ms/frame (1.1 fps) = server 11 + upload read 180 + reply write 6
                               + phone/network 713 | frame 79 kB, reply 1 kB
  ^ 713 ms of that is OFF this server (phone capture + JPEG encode + Wi-Fi).
```

Measured on a 2-core CPU with no GPU, the server's own share is **~10 ms
per frame**. If the phone shows 700 ms, roughly 690 of it is happening
somewhere else, and that line says where.

### What this does and does not fix

Reps and plank seconds are now counted correctly at whatever frame rate
the link achieves. That is a real fix and it needs no new APK.

It does **not** make the link faster. If the round trip is still ~1 s,
the skeleton will still be drawn about a second behind you, because it is
drawing a frame that is a second old. Closing that gap needs the client
changes shipped alongside this zip (the camera captures frames nothing
will ever consume, and the live screen repaints the entire page ten times
a second), and those need one APK rebuild.

---

## Options

```bash
python server.py --threads 8        # more CPU threads for inference
python server.py --imgsz 256        # fix the tracking input size instead of auto
python server.py --latency-budget 12   # allow a bigger model for better keypoints
python server.py --no-roi           # always run on the whole frame (slow; for debugging)
python server.py --port 8770
python server.py --verbose          # pose-engine logging, plus dropped-connection counts
python server.py --identity-path ~/repflow_dev_identity   # explicit key location
```

`--imgsz auto` (the default) benchmarks every model in `server/models/`
at startup and keeps the largest that fits `--latency-budget`.

The identity key lives at `~/.repflow/server_identity` by default —
outside this folder, so replacing this folder (a new zip, a fresh clone)
never invalidates a phone that already paired. `--identity-path` is for
running two coach servers on one machine, or for a sandbox where `~`
isn't a sensible place to write.

The camera resolution the phone sends is `REPFLOW_CAMERA_PRESET`
(`low` / `medium` / `high`, default `medium`). Dropping to `low` cuts the
uploaded bytes roughly fourfold, which is the single biggest lever on
round-trip time that does not need a rebuild — at some cost in finding an
athlete standing well back.

Encryption costs about 1 ms per frame and has never been the problem: the
server authenticates and decrypts before the pose model sees anything, so
if the crypto were failing every frame would come back 401 rather than
uncounted.
