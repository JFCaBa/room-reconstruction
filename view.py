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


def camera_centers(manifest):
    return np.array([[f["transform"][3], f["transform"][7], f["transform"][11]]
                     for f in manifest["frames"]], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--point-size", type=float, default=0.012)
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

    server.scene.add_point_cloud("/cloud", points=pts, colors=cloud_colors,
                                 point_size=args.point_size)

    if args.manifest:
        manifest = json.load(open(args.manifest))
        fp = manifest.get("floorPlan")
        if fp:
            seg, col = wall_wireframe(fp)
            server.scene.add_line_segments("/walls", points=seg, colors=col, line_width=3.0)
            oseg, ocol = opening_wireframe(fp)
            if oseg is not None:
                server.scene.add_line_segments("/openings", points=oseg, colors=ocol, line_width=5.0)
        cams = camera_centers(manifest)
        server.scene.add_point_cloud("/cameras", points=cams,
                                     colors=np.tile([255, 60, 60], (len(cams), 1)).astype(np.uint8),
                                     point_size=0.04)

    print(f"\n  3D viewer running →  http://localhost:{args.port}\n  (Ctrl-C to stop)")
    server.sleep_forever()


if __name__ == "__main__":
    main()
