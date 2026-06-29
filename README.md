# room-reconstruction

Turn an ARKit room scan into a **metric 3D reconstruction**: run LingBot-Map dense
reconstruction on the captured frames, then fuse it with the scan's ARKit poses + measured floor plan
so the dense geometry comes out **to scale, gravity-aligned, in real metres**.

This is a slimmed Apache-2.0 derivative of LingBot-Map (inference code only) plus our fusion step.
See [`NOTICE`](NOTICE) for attribution and the list of modifications.

## What the scanning app gives you

Each scan exports a zip with:

```
frames/000001.jpg …        upright portrait JPEGs (native res)
manifest.json              per-frame metric camera pose + intrinsics + the measured floor plan
```

`manifest.json` schema (the parts fusion uses):

```jsonc
{
  "device": "iPhone12,1",
  "imageWidth": 1440, "imageHeight": 1920,
  "worldAlignment": "gravity", "convention": "arkit",
  "intrinsics": { "fx": …, "fy": …, "cx": …, "cy": … },
  "frames": [
    { "index": 1, "file": "frames/000001.jpg",
      "transform": [16 floats],   // row-major 4x4, world <- camera (c2w), METRIC metres
      "tracking": "normal" }
  ],
  "floorPlan": {                  // measured in the SAME world as `transform`
    "floorY": -1.382, "roomHeight": 2.4,
    "corners": [[x,y,z], …],
    "walls":   [{ "index":0, "start":[x,y,z], "end":[x,y,z], "length": 1.77 }, …],
    "openings":[{ "kind":"door|window|opening", "wallIndex":0, "centerDistance":…,
                  "width":…, "bottomHeight":…, "height":… }, …]
  }
}
```

The point: `floorPlan` and every frame `transform` share one metric ARKit world, so once lingbot is
aligned to that trajectory, its dense cloud and the measured plan live in the same coordinates.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[vis]"          # lingbot deps + matplotlib (fuse.py) + viewer
pip install torch                # platform-specific; see https://pytorch.org
# Optional, CUDA only — much faster streaming attention:
#   pip install flashinfer-python
```

Download the checkpoint (≈4.3 GB) into `checkpoints/` — see the upstream LingBot-Map release.

## Pipeline

```bash
# 1. dense reconstruction -> geometry npz
#    On CUDA, drop --use_sdpa for the faster FlashInfer path.
python demo.py \
  --model_path checkpoints/lingbot-map-long.pt \
  --image_folder /path/to/<scan>/frames \
  --use_sdpa \
  --save_npz out/<scan>_lingbot.npz

# 2. fuse with the ARKit manifest -> metric top-down PNG + metric PLY + report
python fuse.py \
  --manifest /path/to/<scan>/manifest.json \
  --npz out/<scan>_lingbot.npz \
  --conf 3 --voxel 0.02 \
  --out-prefix out/<scan>_fused
```

A streaming npz stores depth (not a prebuilt cloud), so `fuse.py` back-projects it and **cleans** it:
`--conf` (depth_conf gate; ~1–14 for streaming, use **3** — higher prunes the ceiling/upper walls),
`--edge` (drop flying pixels at depth edges), `--voxel` (downsample, uniformises density),
`--sor-k`/`--sor-std` (statistical outlier removal). Defaults are sensible; `--conf 3 --voxel 0.02`
is the recommended starting point for streaming captures.

`fuse.py` prints the **recovered scale (m/unit)** and the **trajectory residual (cm)** — a low
residual means the lingbot and ARKit camera trajectories genuinely agree, i.e. the metric fusion is
trustworthy. Outputs: `<prefix>_topdown.png` (dense footprint with the measured walls/openings
overlaid) and `<prefix>_metric.ply` (dense cloud in metres, ARKit world).

## Validate the fusion math without a GPU

```bash
python fuse.py --selftest
```

Hides a known Sim(3) in synthetic data and asserts `fuse.py` recovers the scale, residual ≈ 0, and
the room footprint/height exactly.

## Notes

- **`--use_sdpa` is required on non-CUDA machines** (Apple Silicon, CPU). FlashInfer is CUDA-only;
  without the flag the model aborts with `FlashInfer is not available`.
- lingbot estimates its own poses and **ignores the manifest** — the manifest is the metric anchor
  applied afterward by `fuse.py`, not an input to the model.
