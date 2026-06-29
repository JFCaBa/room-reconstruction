#!/usr/bin/env python3
"""Local 3D viewer (viser) for a fused metric reconstruction.

Shows the dense metric point cloud (height-coloured) together with the measured floor-plan wireframe
(walls + openings) and the camera path — all in the same ARKit metres. Serves a local web page like
the lingbot demo viewer did, but for the FUSED result.

Usage:
  view.py <metric.ply> [--manifest manifest.json] [--port 8080] [--point-size 0.01]

Then open the printed http://localhost:<port> URL.
"""
import argparse
import json

import numpy as np
import trimesh
import viser


def height_colors(pts):
    import matplotlib.cm as cm
    y = pts[:, 1]
    t = (y - y.min()) / (np.ptp(y) + 1e-9)
    return (cm.viridis(t)[:, :3] * 255).astype(np.uint8)


def wall_wireframe(fp):
    """Walls as line segments (floor edge, ceiling edge, corner posts). Returns (M,2,3), (M,2,3)."""
    floorY = fp["floorY"]
    H = fp.get("roomHeight") or 2.4
    segs = []
    for w in fp["walls"]:
        a = np.array(w["start"], float); b = np.array(w["end"], float)
        af = [a[0], floorY, a[2]]; bf = [b[0], floorY, b[2]]
        ac = [a[0], floorY + H, a[2]]; bc = [b[0], floorY + H, b[2]]
        segs += [[af, bf], [ac, bc], [af, ac]]
    seg = np.array(segs, np.float32)
    col = np.tile(np.array([0, 255, 255], np.uint8), (len(seg), 2, 1))
    return seg, col


def wall_dim_labels(fp):
    """One length label per wall (measured metres), placed at the wall midpoint, nudged outward from
    the room centre and up to floor+0.1 m — the same dimension read-out as the iOS footprint."""
    floorY = fp["floorY"]
    cen = np.array([np.mean([c[0] for c in fp["corners"]]),
                    np.mean([c[2] for c in fp["corners"]])])
    out = []
    for i, w in enumerate(fp["walls"]):
        a = np.array(w["start"], float); b = np.array(w["end"], float)
        L = w.get("length") or float(np.hypot(*(b - a)[[0, 2]]))
        mid = (a + b) / 2
        m2 = np.array([mid[0], mid[2]])
        d = m2 - cen
        d = d / (np.linalg.norm(d) + 1e-9)
        p = m2 + d * 0.12                                   # nudge just outside the wall
        out.append((f"wall{i}", f"{L:.2f} m", (p[0], floorY + 0.1, p[1])))
    return out


def opening_wireframe(fp):
    """Openings as coloured rectangles on their wall. Returns (M,2,3),(M,2,3) or (None,None)."""
    floorY = fp["floorY"]
    kcol = {"door": [80, 255, 80], "window": [0, 180, 255], "opening": [255, 160, 0]}
    segs, cols = [], []
    for op in fp.get("openings", []):
        w = fp["walls"][op["wallIndex"]]
        a = np.array(w["start"], float); b = np.array(w["end"], float)
        d = b - a; L = np.linalg.norm(d)
        d = d / (L + 1e-9)
        c = a + d * op["centerDistance"]; half = op["width"] / 2
        y0 = floorY + op["bottomHeight"]; y1 = y0 + op["height"]
        p = c - d * half; q = c + d * half
        P0 = [p[0], y0, p[2]]; Q0 = [q[0], y0, q[2]]
        P1 = [p[0], y1, p[2]]; Q1 = [q[0], y1, q[2]]
        segs += [[P0, Q0], [P1, Q1], [P0, P1], [Q0, Q1]]
        cols += [kcol.get(op["kind"], [255, 255, 255])] * 4
    if not segs:
        return None, None
    seg = np.array(segs, np.float32)
    col = np.tile(np.array(cols, np.uint8)[:, None, :], (1, 2, 1))
    return seg, col


def floor_grid(fp, step=0.5):
    """A faint reference grid in the floor plane (y = floorY) over the room's footprint bbox."""
    floorY = fp["floorY"]
    cs = np.array([[c[0], c[2]] for c in fp["corners"]])
    x0, z0 = np.floor(cs.min(0) / step) * step
    x1, z1 = np.ceil(cs.max(0) / step) * step
    segs = []
    for x in np.arange(x0, x1 + step / 2, step):
        segs.append([[x, floorY, z0], [x, floorY, z1]])
    for z in np.arange(z0, z1 + step / 2, step):
        segs.append([[x0, floorY, z], [x1, floorY, z]])
    seg = np.array(segs, np.float32)
    col = np.tile(np.array([90, 110, 120], np.uint8), (len(seg), 2, 1))
    return seg, col


def camera_centers(manifest):
    return np.array([[f["transform"][3], f["transform"][7], f["transform"][11]]
                     for f in manifest["frames"]], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--point-size", type=float, default=0.012)
    ap.add_argument("--no-clip", action="store_true",
                    help="don't clip the cloud to the measured room box")
    args = ap.parse_args()

    mesh = trimesh.load(args.ply, process=False)
    pts = np.asarray(mesh.vertices, np.float32)
    vc = getattr(mesh, "colors", None)
    if vc is None or len(vc) != len(pts):
        vc = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if vc is not None and len(vc) == len(pts):
        cloud_colors = np.asarray(vc)[:, :3].astype(np.uint8)   # real photo colour
        kind = "photo-coloured"
    else:
        cloud_colors = height_colors(pts)
        kind = "height-coloured"
    print(f"loaded {len(pts):,} points from {args.ply} ({kind})")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    try:
        server.scene.set_up_direction("+y")   # ARKit world: +Y is up
    except Exception:
        pass

    manifest = json.load(open(args.manifest)) if args.manifest else None
    fp = manifest.get("floorPlan") if manifest else None

    # Clip the cloud to the measured room box (drops above-ceiling noise + stray points outside the
    # walls) so the view is clean. The cloud's own floor is wherever lingbot reconstructed to.
    if fp and not args.no_clip:
        floorY = fp["floorY"]; ceil = floorY + (fp.get("roomHeight") or 2.4)
        cs = np.array([[c[0], c[2]] for c in fp["corners"]])
        lo = cs.min(0) - 0.3; hi = cs.max(0) + 0.3
        m = ((pts[:, 1] > floorY - 0.1) & (pts[:, 1] < ceil + 0.1)
             & (pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0])
             & (pts[:, 2] > lo[1]) & (pts[:, 2] < hi[1]))
        pts, cloud_colors = pts[m], cloud_colors[m]
        print(f"clipped to room box: {m.sum():,} points")

    server.scene.add_point_cloud("/cloud", points=pts, colors=cloud_colors,
                                 point_size=args.point_size)

    if fp:
        seg, col = wall_wireframe(fp)
        server.scene.add_line_segments("/walls", points=seg, colors=col, line_width=3.0)
        oseg, ocol = opening_wireframe(fp)
        if oseg is not None:
            server.scene.add_line_segments("/openings", points=oseg, colors=ocol, line_width=5.0)
        gseg, gcol = floor_grid(fp)
        server.scene.add_line_segments("/floor", points=gseg, colors=gcol, line_width=1.0)
        # dimension labels — wall lengths + room height, like the iOS footprint
        for name, text, pos in wall_dim_labels(fp):
            server.scene.add_label(f"/dims/{name}", text, position=pos,
                                   anchor="bottom-center")
        H = fp.get("roomHeight") or 2.4
        cy = fp["corners"][0]
        server.scene.add_label("/dims/height", f"H {H:.2f} m",
                               position=(cy[0], fp["floorY"] + H / 2, cy[2]),
                               anchor="center-center")
    if manifest:
        cams = camera_centers(manifest)
        server.scene.add_point_cloud("/cameras", points=cams,
                                     colors=np.tile([255, 60, 60], (len(cams), 1)).astype(np.uint8),
                                     point_size=0.04)

    print(f"\n  3D viewer running →  http://localhost:{args.port}\n  (Ctrl-C to stop)")
    server.sleep_forever()


if __name__ == "__main__":
    main()
