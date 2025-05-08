#!/usr/bin/env python3
# -------------------------------------------------------------
#  find_safe_direction.py
#
#  Given:
#     • a watertight scene mesh (.ply, .obj, …)
#     • the 3-D XYZ of the grasp/pick point
#
#  this script finds the direction d̂ (unit vector) that yields
#  the *widest, obstacle-free* straight-line corridor of length L
#  and radius R by “sweeping” a cylinder and counting vertex hits.
#
#  © 2025  BSD-3-Clause
# -------------------------------------------------------------
import argparse
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ---------- Fibonacci-sphere directions ---------------------------------
def fibonacci_sphere(n_dirs: int = 64) -> np.ndarray:
    phi = (1 + 5 ** 0.5) / 2          # golden ratio
    dirs = []
    for i in range(n_dirs):
        z = 1 - 2 * (i + 0.5) / n_dirs
        r = (1 - z * z) ** 0.5
        theta = 2 * math.pi * (i / phi)
        dirs.append([r * math.cos(theta), r * math.sin(theta), z])
    return np.asarray(dirs, dtype=np.float64)


# ---------- cylinder score for *one* direction --------------------------
def score_direction(d: np.ndarray,
                    p: np.ndarray,
                    vertices: np.ndarray,
                    kdtree: cKDTree,
                    R: float,
                    L: float) -> tuple[int, np.ndarray]:
    """
    Returns (hit_count, direction)
    """
    # 1) quick reject with bounding-sphere around the cylinder
    bb_center = p + d * (L / 2)
    bb_radius = (L / 2) + R
    idx = kdtree.query_ball_point(bb_center, bb_radius)
    if not idx:          # empty list ⇒ zero hits
        return 0, d

    vecs = vertices[idx] - p                    # (K,3)
    axial = vecs @ d                           # projection on d
    mask_axial = (axial >= 0.0) & (axial <= L)
    if not mask_axial.any():
        return 0, d

    vecs = vecs[mask_axial]
    axial = axial[mask_axial][:, None]
    radial2 = np.sum((vecs - axial * d) ** 2, axis=1)
    hits = int(np.count_nonzero(radial2 <= R * R))
    return hits, d


# ---------- CLI ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Find safest straight-line approach direction "
                    "for a grasp point inside a cavity.")
    ap.add_argument("--mesh_path", default="/home/exx/datasets/aria/real/spot_room_v1/vol_fusion_v1/mesh_234.ply", type=Path,
                    help="Watertight mesh file of the *environment* (.ply, .obj, …)", required=False)
    ap.add_argument("--pick_xyz", default="-0.9074752,0.18065326,-0.50076877", type=str,
                    help="Comma-separated XYZ of grasp pose, e.g. 0.5,1.0,0.8", required=False)
    # ap.add_argument("--pick_xyz", default="1.5,-1.0,-0.50076877", type=str, required=False)
    ap.add_argument("--radius", type=float, default=0.15,
                    help="Cylinder radius [m] (>= gripper radius + margin)")
    ap.add_argument("--length", type=float, default=0.30,
                    help="Cylinder length / plunge depth [m]")
    ap.add_argument("--n_dirs", type=int, default=64,
                    help="Number of test directions (Fibonacci-sphere)")
    ap.add_argument("--standoff", type=float, default=0.5,
                    help="Distance [m] to place standoff pose back along safest dir")
    return ap.parse_args()


# ---------- main --------------------------------------------------------
def main() -> None:
    args = parse_args()

    # -- load mesh -------------------------------------------------------
    mesh = trimesh.load_mesh(args.mesh_path, force="mesh")
    if mesh.is_empty or mesh.vertices.size == 0:
        sys.exit("ERROR: empty mesh or failed to load.")

    vertices = mesh.vertices.view(np.ndarray)    # (M,3) float64
    kdtree = cKDTree(vertices)

    # -- grasp position --------------------------------------------------
    try:
        p = np.fromstring(args.pick_xyz, sep=",", dtype=np.float64)
        assert p.shape == (3,)
    except Exception:
        sys.exit("ERROR: pick_xyz format must be e.g. 0.5,1.0,0.8")

    # -- sample directions & score --------------------------------------
    dirs = fibonacci_sphere(args.n_dirs)

    # Parallel scoring → list of (hits, dir)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        scores = pool.starmap(
            score_direction,
            [(d, p, vertices, kdtree, args.radius, args.length) for d in dirs])

    best_hits, best_dir = min(scores, key=lambda t: t[0])
    best_dir /= np.linalg.norm(best_dir)          # guard numerical drift

    # -- standoff pose ---------------------------------------------------
    standoff_xyz = p + args.standoff * best_dir

    # -- output ----------------------------------------------------------
    np.set_printoptions(precision=3, suppress=True)
    print(f"best_dir     = {best_dir}")
    print(f"hits         = {best_hits}")
    print(f"standoff_xyz = {standoff_xyz}")


if __name__ == "__main__":
    main()
