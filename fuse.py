#!/usr/bin/env python3
"""Metric fusion: lingbot dense geometry + ARKit metric manifest -> one metric world.

The two captures of the same room are complementary:
  - lingbot predicts dense per-frame geometry (cabinets, counters, real wall shape) AND its own
    per-frame camera poses, but everything is UP-TO-SCALE in an arbitrary frame.
  - the ARKit manifest carries metric, gravity-aligned per-frame camera poses (for the SAME frames)
    plus the measured floor plan (corners / walls / openings / floorY / roomHeight).

Because both sides give a camera pose per frame, we solve a single Sim(3) (Umeyama, 1991) that maps
lingbot's whole camera trajectory onto ARKit's. That recovers, in one shot and for ANY camera motion
(walking, not just an in-place pivot): the metric scale, the rotation into ARKit's gravity world, and
the translation to ARKit's origin. Applying it to lingbot's point cloud puts the dense geometry into
the same metric coordinates as the measured floor plan — best of both worlds.

This supersedes the earlier pivot-circle heuristic (footprint-spike/extract_footprint.py), which
needed the camera to trace a clean horizontal circle and broke on a walking scan.

Usage:
  fuse.py --manifest manifest.json --npz lingbot.npz [--out-prefix out/kitchen]
          [--conf 1.5] [--max-points 600000]
  fuse.py --selftest                      # validate the math on synthetic data (no lingbot needed)

The lingbot .npz is produced by `demo.py --save_npz lingbot.npz` (extrinsic, world_points,
world_points_conf, paths).
"""
import argparse
import json
import os
import sys

import numpy as np


# ── core math ────────────────────────────────────────────────────────────────

def umeyama(src, dst):
    """Least-squares Sim(3) mapping src -> dst (both (N,3)). Returns (s, R, t) with s*R@p+t."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    Sigma = (Xd.T @ Xs) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (Xs ** 2).sum() / n
    s = float((D * np.diag(S)).sum() / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(s, R, t, P):
    """Apply a Sim(3) to points P (...,3)."""
    return s * (P @ R.T) + t


def trajectory_residual(s, R, t, src, dst):
    pred = apply_sim3(s, R, t, src)
    return float(np.sqrt(((pred - dst) ** 2).sum(1).mean()))


# ── IO ───────────────────────────────────────────────────────────────────────

def arkit_centers(manifest):
    """ARKit camera centres (N,3) and the matching frame basenames, from a manifest dict."""
    centers, names = [], []
    for f in manifest["frames"]:
        T = f["transform"]            # row-major 4x4 c2w; translation is column 3
        centers.append([T[3], T[7], T[11]])
        names.append(os.path.basename(f["file"]))
    return np.array(centers, float), names


def unproject_depth(npz, ext, conf_min=0.0, edge_rel=0.05, max_depth=None):
    """Build the dense cloud (M,3) + per-point conf (M,) from per-frame depth + intrinsics + c2w.

    Streaming inference returns depth (not a prebuilt world_points), so we back-project: a pixel (u,v)
    with depth d maps to camera point ((u-cx)/fx·d, (v-cy)/fy·d, d) (OpenCV pinhole), then to lingbot
    world via the c2w extrinsic. Points stay in lingbot's frame/scale — the SAME frame as the camera
    centres used for alignment — so the recovered Sim(3) applies to them unchanged.

    Cleanup applied per frame, at source:
      - conf_min   : drop pixels with depth_conf below this.
      - edge_rel   : drop "flying pixels" at depth discontinuities — where the local depth gradient
                     exceeds edge_rel·depth. These boundary pixels are what render as radial streaks.
      - max_depth  : drop pixels farther than this (lingbot units) — far background seen through
                     doorways/windows that doesn't belong to the room shell.
    """
    D = np.asarray(npz["depth"], float)
    if D.ndim == 4:
        D = D[..., 0]                                 # (N,H,W,1) -> (N,H,W)
    K = np.asarray(npz["intrinsic"], float)           # (N,3,3) for the depth resolution
    C = np.asarray(npz["depth_conf"], float) if "depth_conf" in npz.files else None
    N, H, W = D.shape
    grid_v, grid_u = np.mgrid[0:H, 0:W].astype(float)
    pts_all, conf_all = [], []
    for i in range(N):
        d = D[i]                                       # (H,W)
        gy, gx = np.gradient(d)
        grad = np.hypot(gx, gy)
        keep = d > 0
        keep &= grad <= edge_rel * np.maximum(d, 1e-3)
        if max_depth is not None:
            keep &= d <= max_depth
        if C is not None:
            keep &= C[i] >= conf_min
        if not keep.any():
            continue
        du = d[keep]
        fx, fy, cx, cy = K[i, 0, 0], K[i, 1, 1], K[i, 0, 2], K[i, 1, 2]
        cam = np.stack([(grid_u[keep] - cx) / fx * du,
                        (grid_v[keep] - cy) / fy * du, du], 1)
        world = cam @ ext[i, :3, :3].T + ext[i, :3, 3]
        pts_all.append(world)
        conf_all.append(C[i][keep] if C is not None else np.ones(len(du)))
    return np.concatenate(pts_all), np.concatenate(conf_all)


def lingbot_from_npz(npz, edge_rel=0.05, max_depth=None):
    """lingbot camera centres (N,3), frame basenames, dense points (M,3), conf (M,)."""
    ext = np.asarray(npz["extrinsic"], float)        # (N,3,4) c2w; centre is column 3
    centers = ext[:, :3, 3]
    names = [os.path.basename(str(p)) for p in npz["paths"]]
    if "world_points" in npz.files:
        pts = np.asarray(npz["world_points"], float).reshape(-1, 3)
        conf = (np.asarray(npz["world_points_conf"], float).reshape(-1)
                if "world_points_conf" in npz.files else np.ones(len(pts)))
    else:                                             # streaming npz: derive cloud from depth
        pts, conf = unproject_depth(npz, ext, edge_rel=edge_rel, max_depth=max_depth)
    return centers, names, pts, conf


def match_by_name(names_a, names_b):
    """Index pairs (ia, ib) for frames present in both, in a-order."""
    idx_b = {n: i for i, n in enumerate(names_b)}
    pairs = [(i, idx_b[n]) for i, n in enumerate(names_a) if n in idx_b]
    return pairs


def write_ply(path, pts, conf=None):
    """Binary little-endian PLY: x y z (+ optional float 'confidence')."""
    pts = np.asarray(pts, np.float32)
    has_conf = conf is not None
    with open(path, "wb") as fh:
        hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {len(pts)}",
               "property float x", "property float y", "property float z"]
        if has_conf:
            hdr.append("property float confidence")
        hdr += ["end_header", ""]
        fh.write(("\n".join(hdr)).encode())
        if has_conf:
            rec = np.empty((len(pts), 4), np.float32)
            rec[:, :3] = pts
            rec[:, 3] = np.asarray(conf, np.float32)
        else:
            rec = pts
        fh.write(rec.tobytes())


# ── floor plan geometry (from the manifest) ──────────────────────────────────

def wall_xz(wall):
    return np.array([wall["start"][0], wall["start"][2]]), np.array([wall["end"][0], wall["end"][2]])


def opening_segment(fp, op):
    """The opening as a footprint segment (2 endpoints, x-z) centred on its wall."""
    w = fp["walls"][op["wallIndex"]]
    a, b = wall_xz(w)
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-6:
        return a, b
    d = d / L
    c = a + d * op["centerDistance"]
    half = op["width"] / 2
    return c - d * half, c + d * half


# ── cloud cleaning (metric space, after alignment) ───────────────────────────

def voxel_downsample(pts, conf, voxel):
    """Keep the highest-confidence point per `voxel`-metre cell. Uniformises the density (kills the
    near-camera over-density that renders as radial fans) and shrinks the cloud cheaply."""
    keys = np.floor(pts / voxel).astype(np.int64)
    keys -= keys.min(0)
    dims = keys.max(0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    order = np.lexsort((-conf, flat))          # within each voxel, highest conf sorts first
    flat_o = flat[order]
    first = np.empty(len(flat_o), bool)
    first[0] = True
    first[1:] = flat_o[1:] != flat_o[:-1]
    sel = order[first]
    return pts[sel], conf[sel]


def remove_outliers(pts, conf, k=16, std_ratio=2.0):
    """Statistical outlier removal: drop points whose mean distance to their k nearest neighbours is
    an outlier (> mean + std_ratio·std). Clears isolated flecks and the thin streak tails."""
    from scipy.spatial import cKDTree
    if len(pts) <= k:
        return pts, conf
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1, workers=-1)
    md = d[:, 1:].mean(1)
    keep = md <= md.mean() + std_ratio * md.std()
    return pts[keep], conf[keep]


def clean_cloud(pts, conf, voxel=0.02, sor_k=16, sor_std=2.0):
    if voxel and voxel > 0:
        pts, conf = voxel_downsample(pts, conf, voxel)
    if sor_k and sor_k > 0:
        pts, conf = remove_outliers(pts, conf, sor_k, sor_std)
    return pts, conf


def robust_ceiling(height_above):
    """Ceiling height as the histogram mode of the upper points, not p99 — immune to a thin tail of
    stray high points that otherwise inflates the reported room height."""
    hi = height_above[(height_above > 1.6) & (height_above < 3.2)]
    if len(hi) < 50:
        return float(np.percentile(height_above, 99))
    h, e = np.histogram(hi, bins=80)
    c = (e[:-1] + e[1:]) / 2
    return float(c[h.argmax()])


# ── reporting + render ───────────────────────────────────────────────────────

def fuse(manifest, lb_centers, lb_names, pts, conf, conf_thresh, max_points,
         voxel=0.02, sor_k=16, sor_std=2.0, out_prefix=None, quiet=False):
    """Run the alignment and (optionally) render/export. Returns a result dict."""
    ak_c, ak_names = arkit_centers(manifest)
    pairs = match_by_name(ak_names, lb_names)
    if len(pairs) < 3:
        raise SystemExit(f"need >=3 matched frames, got {len(pairs)} "
                         f"(ARKit {len(ak_names)} vs lingbot {len(lb_names)})")
    ia = [p[0] for p in pairs]
    ib = [p[1] for p in pairs]
    s, R, t = umeyama(lb_centers[ib], ak_c[ia])
    resid = trajectory_residual(s, R, t, lb_centers[ib], ak_c[ia])

    # dense cloud -> metric ARKit world, then clean (voxel downsample + outlier removal)
    keep = conf >= conf_thresh
    pts_m = apply_sim3(s, R, t, pts[keep])
    conf_m = conf[keep]
    raw_n = len(pts_m)
    pts_m, conf_m = clean_cloud(pts_m, conf_m, voxel=voxel, sor_k=sor_k, sor_std=sor_std)
    if max_points and len(pts_m) > max_points:
        sel = np.random.default_rng(0).choice(len(pts_m), max_points, replace=False)
        pts_m, conf_m = pts_m[sel], conf_m[sel]

    fp = manifest.get("floorPlan")
    floor_y = fp["floorY"] if fp else float(np.percentile(pts_m[:, 1], 2))
    height_above = pts_m[:, 1] - floor_y
    ceil_y = floor_y + (fp["roomHeight"] if (fp and fp.get("roomHeight")) else
                        robust_ceiling(height_above))

    res = {
        "matched_frames": len(pairs),
        "scale_m_per_unit": s,
        "trajectory_residual_cm": resid * 100,
        "cloud_points_raw": raw_n,
        "cloud_points": len(pts_m),
        "floorY": floor_y,
        "recon_room_height_m": robust_ceiling(height_above),
    }
    # wall band (drop floor/ceiling clutter) for footprint stats + plot
    band = (pts_m[:, 1] > floor_y + 0.2) & (pts_m[:, 1] < ceil_y - 0.2)
    Xb, Zb = pts_m[band, 0], pts_m[band, 2]
    if len(Xb):
        xlo, xhi = np.percentile(Xb, [1, 99])
        zlo, zhi = np.percentile(Zb, [1, 99])
        res["footprint_bbox_m"] = (float(xhi - xlo), float(zhi - zlo))
    if fp:
        res["measured_wall_lengths_m"] = [round(w["length"], 3) for w in fp["walls"]]

    if not quiet:
        print("── metric fusion ───────────────────────────────────────────")
        print(f"matched frames        : {res['matched_frames']}")
        print(f"recovered scale       : {s:.4f} m / lingbot-unit")
        print(f"trajectory fit (RMS)  : {res['trajectory_residual_cm']:.2f} cm   "
              f"(lower = better alignment)")
        print(f"dense points          : {res['cloud_points']:,} "
              f"(from {res['cloud_points_raw']:,} raw, conf>={conf_thresh})")
        if "footprint_bbox_m" in res:
            bb = res["footprint_bbox_m"]
            print(f"recon footprint bbox  : {bb[0]:.2f} × {bb[1]:.2f} m")
        print(f"recon room height     : {res['recon_room_height_m']:.2f} m"
              + (f"   (measured {fp['roomHeight']:.2f} m)" if (fp and fp.get('roomHeight')) else ""))

    if out_prefix:
        _render(out_prefix + "_topdown.png", Xb, Zb, fp, res, s)
        write_ply(out_prefix + "_metric.ply", pts_m, conf_m)
        if not quiet:
            print(f"wrote {out_prefix}_topdown.png and {out_prefix}_metric.ply")
    return res


def _render(path, Xb, Zb, fp, res, scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    if len(Xb):
        ax.hexbin(Xb, Zb, gridsize=320, bins="log", cmap="inferno")
    if fp:
        for w in fp["walls"]:
            a, b = wall_xz(w)
            ax.plot([a[0], b[0]], [a[1], b[1]], color="cyan", lw=1.6, alpha=0.9)
        cs = np.array([[c[0], c[2]] for c in fp["corners"]])
        ax.scatter(cs[:, 0], cs[:, 1], c="cyan", s=18, zorder=5)
        kcol = {"door": "lime", "window": "deepskyblue", "opening": "orange"}
        for op in fp.get("openings", []):
            p, q = opening_segment(fp, op)
            ax.plot([p[0], q[0]], [p[1], q[1]], color=kcol.get(op["kind"], "white"), lw=4, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.grid(True, color="cyan", alpha=0.25, lw=0.5)
    title = (f"Metric fusion  scale={scale:.3f} m/u  fit={res['trajectory_residual_cm']:.1f} cm\n"
             f"room height ≈ {res['recon_room_height_m']:.2f} m  "
             f"(cyan = measured walls, segments = openings)")
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


# ── self-test (no lingbot needed) ────────────────────────────────────────────

def selftest():
    rng = np.random.default_rng(7)

    # A measured ARKit plan: a 3.2 x 2.4 m room, floor at y=-1.4, with a door and a window.
    floorY, roomH = -1.4, 2.4
    corners = [[0, floorY, 0], [3.2, floorY, 0], [3.2, floorY, 2.4], [0, floorY, 2.4]]
    walls = []
    for i in range(4):
        a = corners[i]; b = corners[(i + 1) % 4]
        walls.append({"index": i, "start": a, "end": b,
                      "length": float(np.hypot(b[0] - a[0], b[2] - a[2]))})
    fp = {"floorY": floorY, "roomHeight": roomH, "corners": corners, "walls": walls,
          "openings": [{"kind": "door", "wallIndex": 0, "centerDistance": 1.0,
                        "width": 0.9, "bottomHeight": 0.0, "height": 2.0},
                       {"kind": "window", "wallIndex": 2, "centerDistance": 1.6,
                        "width": 1.2, "bottomHeight": 1.0, "height": 1.0}]}

    # A walking camera path (NOT a pivot) at ~1.4 m height.
    n = 40
    th = np.linspace(0, 1.8 * np.pi, n)
    ak_c = np.c_[1.6 + 0.9 * np.cos(th) + 0.1 * rng.standard_normal(n),
                 np.full(n, floorY + 1.4),
                 1.2 + 0.6 * np.sin(th) + 0.1 * rng.standard_normal(n)]
    transforms = []
    for c in ak_c:
        T = np.eye(4); T[:3, 3] = c
        transforms.append(T.reshape(-1).tolist())
    manifest = {"frames": [{"file": f"frames/{i+1:06d}.jpg", "transform": transforms[i]}
                           for i in range(n)],
                "floorPlan": fp}

    # A dense metric room shell: 4 walls (floor->ceiling) + floor + ceiling planes.
    pts_metric = []
    for w in walls:
        a = np.array(w["start"]); b = np.array(w["end"])
        for f in np.linspace(0, 1, 120):
            base = a + (b - a) * f
            for h in np.linspace(0, roomH, 60):
                pts_metric.append([base[0], floorY + h, base[2]])
    gx, gz = np.meshgrid(np.linspace(0, 3.2, 80), np.linspace(0, 2.4, 60))
    for yy in (floorY, floorY + roomH):                # floor and ceiling planes
        for x, z in zip(gx.reshape(-1), gz.reshape(-1)):
            pts_metric.append([x, yy, z])
    pts_metric = np.array(pts_metric)

    # Hide a known Sim(3): build a synthetic lingbot frame as its inverse.
    s0 = 0.6
    ang = 0.7
    R0 = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    # extra tilt so "up" isn't axis-aligned in lingbot frame
    tlt = 0.25
    Rt = np.array([[1, 0, 0], [0, np.cos(tlt), -np.sin(tlt)], [0, np.sin(tlt), np.cos(tlt)]])
    R0 = R0 @ Rt
    t0 = np.array([5.0, -2.0, 3.0])
    inv = lambda P: (1.0 / s0) * (P - t0) @ R0   # inverse of s0*R0@p+t0  (R0 orthonormal)

    lb_c = inv(ak_c)
    lb_pts = inv(pts_metric)
    ext = np.zeros((n, 3, 4))
    ext[:, :3, :3] = np.eye(3)        # Umeyama uses centres only
    ext[:, :3, 3] = lb_c
    conf = np.full(len(lb_pts), 3.0)

    s, R, t = umeyama(lb_c, ak_c)
    resid = trajectory_residual(s, R, t, lb_c, ak_c)
    recon = apply_sim3(s, R, t, lb_pts)
    max_err = float(np.abs(recon - pts_metric).max())

    print(f"hidden scale s0={s0}  recovered s={s:.5f}   |Δ|={abs(s-s0):.2e}")
    print(f"trajectory residual = {resid*100:.4f} cm")
    print(f"max cloud reprojection error = {max_err*1000:.3f} mm")

    ok = abs(s - s0) < 1e-3 and resid < 1e-3 and max_err < 1e-3
    # exercise render + ply + reporting end-to-end
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        res = fuse(manifest, lb_c, [f"{i+1:06d}.jpg" for i in range(n)],
                   lb_pts, conf, conf_thresh=1.5, max_points=0,
                   out_prefix=os.path.join(d, "selftest"), quiet=False)
        ok = ok and os.path.exists(os.path.join(d, "selftest_topdown.png"))
        ok = ok and os.path.exists(os.path.join(d, "selftest_metric.ply"))
        bb = res.get("footprint_bbox_m", (0, 0))
        ok = ok and abs(bb[0] - 3.2) < 0.05 and abs(bb[1] - 2.4) < 0.05
        ok = ok and abs(res["recon_room_height_m"] - roomH) < 0.05
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ── cli ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest")
    ap.add_argument("--npz")
    ap.add_argument("--out-prefix", default=None,
                    help="write <prefix>_topdown.png and <prefix>_metric.ply")
    ap.add_argument("--conf", type=float, default=1.5,
                    help="confidence threshold (world_points_conf, or depth_conf for streaming npz)")
    ap.add_argument("--edge", type=float, default=0.05,
                    help="depth-cloud edge filter: drop pixels whose depth gradient exceeds edge·depth "
                         "(removes flying pixels / radial streaks). Streaming npz only.")
    ap.add_argument("--max-depth", type=float, default=None,
                    help="drop depth-cloud points farther than this (lingbot units ~ metres) — "
                         "far background through openings. Streaming npz only.")
    ap.add_argument("--voxel", type=float, default=0.02,
                    help="voxel downsample size in metres (0 = off)")
    ap.add_argument("--sor-k", type=int, default=16,
                    help="statistical outlier removal: neighbours per point (0 = off)")
    ap.add_argument("--sor-std", type=float, default=2.0,
                    help="statistical outlier removal: keep points within mean+std·this")
    ap.add_argument("--max-points", type=int, default=600000, help="cap cloud size (0 = no cap)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not (args.manifest and args.npz):
        ap.error("need --manifest and --npz (or --selftest)")

    manifest = json.load(open(args.manifest))
    npz = np.load(args.npz, allow_pickle=True)
    lb_c, lb_names, pts, conf = lingbot_from_npz(npz, edge_rel=args.edge, max_depth=args.max_depth)
    fuse(manifest, lb_c, lb_names, pts, conf,
         conf_thresh=args.conf, max_points=args.max_points,
         voxel=args.voxel, sor_k=args.sor_k, sor_std=args.sor_std, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()
