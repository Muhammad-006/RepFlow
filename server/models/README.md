# The pose models

Same YOLO11n-Pose weights, exported at several input sizes. The size is
read from the filename, so the engine can benchmark them all on the
machine it is running on and pick — there is no config to keep in sync.

| file | input | what it is for |
|---|---|---|
| `yolo11n-pose-192.onnx` | 192 px | tracking on the athlete crop, on a slow CPU |
| `yolo11n-pose-256.onnx` | 256 px | the smallest size that reliably **finds** a distant athlete on a whole frame; the floor for full-frame passes |
| `yolo11n-pose-320.onnx` | 320 px | tracking on a faster laptop, or with `--latency-budget` raised |
| `yolo11n-pose.onnx` | 640 px | the accuracy reference the tests compare against; too slow to serve from (~60 ms on 2 cores) |

`TRACKING_FIXES.md` (repo root) has the measurements behind the split
between "finding" and "tracking", and why a model small enough to hit the
latency budget on a whole frame cannot see someone standing three metres
away.

## If some of these are missing

The app runs with any subset — it uses the largest model inside the
latency budget for tracking and the smallest one at or above 256 px for
finding. With fewer files it simply has fewer choices:

* **192 + 256 only** (the default download, kept small enough to email):
  everything works. A fast laptop tracks at 256 px instead of 320.
* **256 only:** works; tracking costs whatever 256 px costs on that
  machine, which may be over the 8 ms budget — the startup line says so.
* **no model at or above 256 px:** the server warns that a distant
  athlete may not be found at all. Don't ship this.

The 320 px and 640 px exports are in the separate extras archive. Drop
them into this folder — nothing else to do — to widen the auto-picker's
choices and to enable the reference-accuracy test
(`server/tests/test_pose_engine.py::test_keypoints_match_a_bigger_reference`)
and section 3 of `tools/verify_tracking.py`.

## Re-exporting them yourself

```bash
pip install ultralytics
python -c "
from ultralytics import YOLO
m = YOLO('yolo11n-pose.pt')
for s in (192, 256, 320, 640):
    m.export(format='onnx', imgsz=s, opset=12, simplify=True)
"
```

Then rename each output to `yolo11n-pose-<size>.onnx` and drop it here.
Exports are static-shaped on purpose: a dynamic-axis export of these
weights has hardcoded reshape nodes and does not work.
