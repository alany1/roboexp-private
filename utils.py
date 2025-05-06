from scipy.ndimage import binary_dilation
import numpy as np
import torch
import math
import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon   # pip install scikit-image

import numpy as np
import numpy as np
import numpy as np
import numpy as np
from scipy.optimize import least_squares
def apply_depth_hyperbola(est_depth: np.ndarray,
                          S: float,
                          c: float,
                          out: np.ndarray | None = None):
    """
    Convert a normalised depth image E to metric depth D̂ using

        D̂  =  S / (E + c)

    Parameters
    ----------
    est_depth : (H,W) float32/float64 array   – normalised depth E
    S, c      : scalars returned by `fit_depth_hyperbola`
    out       : optional pre-allocated array for the result

    Returns
    -------
    metric_depth : (H,W) float64 array  – estimated metric depth
    """
    if out is None:
        out = np.empty_like(est_depth, dtype=np.float64)
    else:
        if out.shape != est_depth.shape:
            raise ValueError("`out` must have the same shape as est_depth")
        out = out.astype(np.float64, copy=False)

    # The +1e-12 prevents accidental division by zero at E ≈ −c
    np.divide(S, est_depth + c + 1e-12, out=out, where=np.isfinite(est_depth))

    # optional: mark invalid pixels (where E was inf/NaN) as NaN in the output
    out[~np.isfinite(est_depth)] = np.nan
    return out
def fit_depth_hyperbola(
    metric_depth,
    est_depth,
    mask=None,
    depth_min=0.0,
    depth_max=np.inf,
    train_split=0.8,
    n_samples=200_000,   # subsample cap (keeps it fast)
    init_S=None, init_c=None,
    robust_loss="soft_l1",
    f_scale=1.0,
    random_state=0,
):
    """
    Robustly fit   D ≈ S / (E + c)   using scipy.least_squares.

    Returns
    -------
    S, c              : floats
    val_mae, val_rel  : validation error on the held-out set
    """

    rng = np.random.default_rng(random_state)

    D = metric_depth.astype(np.float64)
    E = est_depth.astype(np.float64)

    if mask is None:
        mask = (np.isfinite(D) & np.isfinite(E) &
                (D > depth_min) & (D < depth_max))
    else:
        mask = mask & np.isfinite(D) & np.isfinite(E) & \
                (D > depth_min) & (D < depth_max)

    if mask.sum() < 1000:
        raise ValueError("Too few valid pixels after masking")

    # ------------------------------------------------------------------ #
    # 1)  random subsample to keep LM fast                               #
    # ------------------------------------------------------------------ #
    idxs = np.column_stack(np.nonzero(mask))
    if idxs.shape[0] > n_samples:
        idxs = idxs[rng.choice(idxs.shape[0], n_samples, replace=False)]

    # shuffle & split
    rng.shuffle(idxs)
    split = int(idxs.shape[0] * train_split)
    train, val = idxs[:split], idxs[split:]

    E_train = E[train[:, 0], train[:, 1]]
    D_train = D[train[:, 0], train[:, 1]]
    E_val   = E[val[:, 0], val[:, 1]]
    D_val   = D[val[:, 0], val[:, 1]]

    # ------------------------------------------------------------------ #
    # 2)  least-squares with robust loss                                 #
    # ------------------------------------------------------------------ #
    def depth_model(p, x):        # p = [S, c]
        return p[0] / (x + p[1])

    def residuals(p, x, y):
        return depth_model(p, x) - y

    # initial guess ---------------------------------------------------- #
    if init_S is None or init_c is None:
        # crude but safe: pick two quantiles and solve S & c analytically
        q10, q90 = np.quantile(E_train, [0.1, 0.9])
        D_q10 = np.median(D_train[(E_train >= q10*0.95) & (E_train <= q10*1.05)])
        D_q90 = np.median(D_train[(E_train >= q90*0.95) & (E_train <= q90*1.05)])
        # Solve S/(q10+c)=D_q10  and  S/(q90+c)=D_q90
        c_init = (q10*D_q10 - q90*D_q90) / (D_q90 - D_q10 + 1e-12)
        S_init = D_q10 * (q10 + c_init)
        if not np.isfinite(S_init) or S_init <= 0:
            S_init, c_init = 1.0, 0.1
    else:
        S_init, c_init = init_S, init_c

    res = least_squares(
        fun=residuals,
        x0=[S_init, c_init],
        args=(E_train, D_train),
        loss=robust_loss,
        f_scale=f_scale,
        bounds=([1e-6, -0.99], [np.inf, 10.0])  # keep denominator positive
    )

    S, c = res.x

    # ------------------------------------------------------------------ #
    # 3)  validation error                                               #
    # ------------------------------------------------------------------ #
    D_hat_val = depth_model(res.x, E_val)
    val_mae   = np.abs(D_hat_val - D_val).mean()
    val_rel   = (np.abs(D_hat_val - D_val) / D_val).mean()

    print("Robust hyperbola fit:")
    print(f"  S = {S:.6f} ,  c = {c:.6f}")
    print(f"  validation  MAE = {val_mae*1e2:.2f} cm   REL = {val_rel*1e2:.2f}%")

    return float(S), float(c), float(val_mae), float(val_rel)



def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    mask : (H,W) np.ndarray of bool / {0,1}
        Input binary mask whose foreground pixels define the point set.

    Returns
    -------
    filled : (H,W) np.ndarray of bool
        Binary mask where the entire convex hull of the input foreground is 1/True.
        If the input mask is empty, returns an all-False array.
    """
    # 1) coordinates of foreground pixels ---------------------------------
    ys, xs = np.nonzero(mask)          # row = y, col = x
    if len(xs) == 0:                   # empty mask → nothing to do
        return np.zeros_like(mask, dtype=bool)

    pts = np.column_stack([xs, ys])    # (N,2) with (x,y) pairs

    # 2) convex hull in image coords --------------------------------------
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]      # ordered vertices of the hull

    # 3) rasterise / fill the polygon -------------------------------------
    rr, cc = polygon(hull_pts[:, 1],   # rows (y)
                     hull_pts[:, 0],   # columns (x)
                     shape=mask.shape)

    filled = np.zeros_like(mask, dtype=bool)
    filled[rr, cc] = True
    return filled

def backproject_points_cam(*, depth, pose, fx, fy, cx, cy, depth_range=(0.1, 5.0), seg=None, device="cuda", aria_rot=True):
    """
    Backproject the depth image to 3D points in camera space.

    :param depth: np.ndarray (HxW depth image).
    :param pose: np.ndarray (4x4 camera-to-world transform).
    :return: np.ndarray (Nx3 points in world space).
    """

    depth_near, depth_far = depth_range

    if seg is not None:
        mask = seg == 12  # human

        structure = np.ones((10, 10), dtype=bool)
        mask = binary_dilation(mask, structure=structure, iterations=5)

        depth[mask] = 0.0

    if aria_rot:
        depth = np.rot90(depth, k=1).copy()
    depth = torch.from_numpy(depth).to(device).float()

    height, width = depth.shape
    c2w = pose

    R_np, t_np = c2w[:3, :3], c2w[:3, 3:]

    R = torch.from_numpy(R_np).float().to(device)  # Shape: [3, 3]
    t = torch.from_numpy(t_np).float().to(device)  # Shape: [3, 1]

    # Create coordinate grids
    u = torch.arange(0, width, device=device).unsqueeze(0).repeat(height, 1)  # Shape: [H, W]
    v = torch.arange(0, height, device=device).unsqueeze(1).repeat(1, width)  # Shape: [H, W]

    # Apply valid depth conditions
    valid_mask = (depth > depth_near) & (depth < depth_far)
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z = depth[valid_mask]

    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy

    points_camera = torch.stack((x, y, z), dim=1)  # Shape: [N, 3]

    points_world = (R @ points_camera.T + t).T  # Shape: [N, 3]

    return_points = torch.zeros(height, width, 3, device=device)
    return_points[v_valid, u_valid] = points_camera

    # to match the original image's orientation
    if aria_rot:
        return_points = torch.rot90(return_points, k=3).clone()

    return return_points.to(device)

def rotate_points_batched(
    points: torch.Tensor,  # shape: (M, 3)
    pivots: torch.Tensor,  # shape: (B, 3)
    axes: torch.Tensor,  # shape: (B, 3)
    angles: torch.Tensor,  # shape: (B,)
) -> torch.Tensor:
    """
    Rotate the same set of `points` (M,3) around each (pivot, axis, angle) in a batched way.

    Returns a tensor of shape (B, M, 3) where each of the B rows is a rotated version of 'points'.
    """

    # points: (M,3)
    # We broadcast so pivot->(B,1,3), axis->(B,1,3), etc. and produce (B,M,3) output.

    B = pivots.shape[0]
    M = points.shape[0]

    # Expand points to shape (1, M, 3) so that we can broadcast over the batch dimension
    points_1m3 = points.unsqueeze(0)  # => (1, M, 3)

    # Reshape pivots to (B, 1, 3) for broadcasting
    pivots_b13 = pivots.unsqueeze(1)  # => (B, 1, 3)

    # Shift points so pivot is at origin => (B, M, 3)
    shifted = points_1m3 - pivots_b13

    # Normalize axes => shape (B, 3) => (B, 1, 3)
    axis_norm = axes / (axes.norm(dim=1, keepdim=True) + 1e-9)
    axis_norm_b13 = axis_norm.unsqueeze(1)  # => (B, 1, 3)

    # Dot and cross products
    dot = (shifted * axis_norm_b13).sum(dim=2, keepdim=True)  # (B, M, 1)
    cross = torch.cross(axis_norm_b13.expand(-1, M, -1), shifted, dim=2)  # (B, M, 3)

    # angles => (B,)
    cos_t = torch.cos(angles)  # (B,)
    sin_t = torch.sin(angles)  # (B,)
    one_minus_cos = 1 - cos_t

    # Broadcast them to (B,1,1) so we can multiply with (B,M,3)
    cos_t_b11 = cos_t.view(B, 1, 1)
    sin_t_b11 = sin_t.view(B, 1, 1)
    omc_b11 = one_minus_cos.view(B, 1, 1)

    # Rodrigues' rotation formula
    rotated = shifted * cos_t_b11 + cross * sin_t_b11 + axis_norm_b13 * (dot * omc_b11)  # shape => (B, M, 3)

    # Shift back
    rotated = rotated + pivots_b13  # => (B, M, 3)

    return rotated
def rotate_pcd(*, pcd, hinge_axis, hinge_pivot, rad):
    
    obj_pcd_t = torch.from_numpy(pcd).cpu().float()
    
    hinge_pivot = hinge_pivot[None, ...]
    hinge_pivot_t = torch.tensor(hinge_pivot).cpu().float()
    
    hinge_axis = hinge_axis[None, ...]
    hinge_axis_t = torch.tensor(hinge_axis).cpu().float()
    
    target_angle = rad

    rotated_pts = (
        rotate_points_batched(
            obj_pcd_t, hinge_pivot_t, hinge_axis_t, torch.tensor(target_angle)
        )
        .cpu()
        .numpy()
    )[0]
    
    return rotated_pts
def translate_pcd(*, pcd, direction, amount):
    t_axis = direction[None, ...]
    return pcd + t_axis * amount


import open3d as o3d
import numpy as np
import math


def _poisson_surface_samples(hull_mesh, d, oversample=2):
    """Poisson-disk samples on the *surface* (works on any Open3D version)."""
    area = hull_mesh.get_surface_area()
    n    = math.ceil(4.0 * area / (math.sqrt(3) * d**2))

    init = hull_mesh.sample_points_uniformly(n * oversample)
    try:  # Open3D ≥ 0.18
        pcd = hull_mesh.sample_points_poisson_disk(n, init=init, pcl=init)
    except TypeError:  # older releases
        pcd = hull_mesh.sample_points_poisson_disk(n, init_factor=5, pcl=init)
    return np.asarray(pcd.points)


def _points_in_convex_hull(hull_mesh, query_pts):
    """
    Vectorised ‘point ∈ convex-hull?’ test using half-space inequalities.
    Works even if Open3D lacks mesh.is_point_inside().
    """
    try:  # Open3D ≥ 0.17
        return np.asarray(hull_mesh.is_point_inside(
            o3d.utility.Vector3dVector(query_pts)
        ))
    except AttributeError:
        # --- manual half-space test ---------------------------------------
        tris = np.asarray(hull_mesh.triangles)
        verts = np.asarray(hull_mesh.vertices)

        # plane eqns  n·x ≤ c  for each face, where n points *outwards*
        n_vecs = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]],
                          verts[tris[:, 2]] - verts[tris[:, 0]])
        n_norm = np.linalg.norm(n_vecs, axis=1, keepdims=True)
        n_vecs /= n_norm + 1e-12
        c_vals = (n_vecs * verts[tris[:, 0]]).sum(axis=1)

        # ensure normals face outwards
        dots = (verts @ n_vecs.T)   # (V,F)
        wrong = dots.max(axis=0) > c_vals + 1e-6
        n_vecs[wrong] *= -1
        c_vals[wrong] *= -1

        # point-inside test
        return ((query_pts @ n_vecs.T) <= c_vals + 1e-9).all(axis=1)


def sample_convex_hull_dense_volume(points, d, oversample_surface=2):
    """
    Densely samples *both* the surface and the **interior** of the
    convex hull so that every location is ≤ d from some sample.

    Parameters
    ----------
    points : (N,3) array-like
    d      : max allowed hole radius
    oversample_surface : Poisson oversampling factor for the surface

    Returns
    -------
    (M,3) ndarray of sample locations
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")

    # 1) convex hull -------------------------------------------------------
    pcd   = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    hull, _ = pcd.compute_convex_hull()
    hull.remove_duplicated_vertices()

    # 2) SURFACE   (Poisson-disk) -----------------------------------------
    surf_pts = _poisson_surface_samples(hull, d, oversample_surface)

    # 3) INTERIOR  (regular grid + inside-test) ---------------------------
    #
    # grid step so that any location is ≤ d from a grid centre:
    # farthest distance in a cube of edge s is √3·s/2 — set to d.
    step = 2 * d / math.sqrt(3)

    min_bb = np.min(points, axis=0) - step
    max_bb = np.max(points, axis=0) + step
    xs     = np.arange(min_bb[0], max_bb[0] + step * 0.5, step)
    ys     = np.arange(min_bb[1], max_bb[1] + step * 0.5, step)
    zs     = np.arange(min_bb[2], max_bb[2] + step * 0.5, step)

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)

    inside_mask = _points_in_convex_hull(hull, grid)
    interior_pts = grid[inside_mask]

    # 4) merge & return ----------------------------------------------------
    return np.vstack((surf_pts, interior_pts))

def axis_aligned_bbox(pts: np.ndarray):
    """
    Parameters
    ----------
    pts : (N, 3) float array

    Returns
    -------
    center  : (3,)   – (x,y,z) of box centre
    extent  : (3,)   – side lengths (dx, dy, dz)
    corners : (8, 3) – the 8 corner points
    """
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must be an (N,3) array")

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    extent = maxs - mins

    # enumerate corners (binary combinations of mins/maxs)
    corners = np.array([[x, y, z]
                        for x in (mins[0], maxs[0])
                        for y in (mins[1], maxs[1])
                        for z in (mins[2], maxs[2])])

    return center, extent, corners

import numpy as np

def iou_aabb(box1, box2):
    """
    Intersection-over-Union for two axis-aligned 3-D bounding boxes.

    Parameters
    ----------
    box1, box2 : array-like, shape (6,) or ((3,), (3,))
        Either
          • [xmin, ymin, zmin, xmax, ymax, zmax],  or
          • ((xmin, ymin, zmin), (xmax, ymax, zmax))

    Returns
    -------
    float
        IoU ∈ [0, 1].
    """
    # unpack -> (min, max)  each of shape (3,)
    b1_min, b1_max = (np.asarray(box1[:3]), np.asarray(box1[3:])) if len(box1) == 6 else map(np.asarray, box1)
    b2_min, b2_max = (np.asarray(box2[:3]), np.asarray(box2[3:])) if len(box2) == 6 else map(np.asarray, box2)

    # intersection box
    inter_min = np.maximum(b1_min, b2_min)
    inter_max = np.minimum(b1_max, b2_max)
    inter_sizes = np.maximum(0.0, inter_max - inter_min)    # clamp negatives → 0
    inter_vol = np.prod(inter_sizes)

    if inter_vol == 0.0:
        return 0.0                                           # no overlap

    # volumes
    vol1 = np.prod(np.maximum(0.0, b1_max - b1_min))
    vol2 = np.prod(np.maximum(0.0, b2_max - b2_min))

    return inter_vol / (vol1 + vol2 - inter_vol)

def inersection_aabb(box1, box2):
    """
    Intersection-over-Union for two axis-aligned 3-D bounding boxes.

    Parameters
    ----------
    box1, box2 : array-like, shape (6,) or ((3,), (3,))
        Either
          • [xmin, ymin, zmin, xmax, ymax, zmax],  or
          • ((xmin, ymin, zmin), (xmax, ymax, zmax))

    Returns
    -------
    float
        IoU ∈ [0, 1].
    """
    # unpack -> (min, max)  each of shape (3,)
    b1_min, b1_max = (np.asarray(box1[:3]), np.asarray(box1[3:])) if len(box1) == 6 else map(np.asarray, box1)
    b2_min, b2_max = (np.asarray(box2[:3]), np.asarray(box2[3:])) if len(box2) == 6 else map(np.asarray, box2)

    # intersection box
    inter_min = np.maximum(b1_min, b2_min)
    inter_max = np.minimum(b1_max, b2_max)
    inter_sizes = np.maximum(0.0, inter_max - inter_min)    # clamp negatives → 0
    inter_vol = np.prod(inter_sizes)

    return inter_vol

def pick(d, *keys, strict=False):
    """Pick keys"""
    _d = {}
    for k in keys:
        if k in d:
            _d[k] = d[k]
        elif strict:
            raise KeyError(k)
    return _d


def sample_bbox_dense_volume(points, d, oversample_surface: int = 2):
    """
    Sample a regular 3-D grid that *densely* covers the **axis-aligned bounding-box**
    (AABB) of the input points, plus a denser 2-D grid on each of the six faces.

    Any location inside the AABB is guaranteed to lie within ≤ d of at least one
    returned sample.

    Parameters
    ----------
    points : (N, 3) array-like
        Input point cloud (only its AABB is used).
    d : float
        Maximum admissible “hole” radius (coverage distance).
    oversample_surface : int, default=2
        How many times **denser** the face grids should be, relative to the interior
        spacing.  (1 ⇒ same spacing as interior; 2 ⇒ twice as dense, etc.)

    Returns
    -------
    samples : (M, 3) `np.ndarray`
        The concatenated 3-D interior samples and the 2-D face samples.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("`points` must be an (N, 3) array")

    # ---------------------------------------------------------------------
    # 1) grid spacing for full 3-D coverage
    #    – farthest distance from a grid centre in a cube of edge s is √3·s/2
    #      ⇒ choose s so that √3·s/2 ≤ d  ⇒  s = 2d / √3
    step_3d = 2.0 * d / math.sqrt(3)

    # ---------------------------------------------------------------------
    # 2) axis-aligned bounding box (no padding)
    bb_min = points.min(axis=0)
    bb_max = points.max(axis=0)

    # Ensure the extremes are included by adding half a step
    xs = np.arange(bb_min[0], bb_max[0] + step_3d * 0.5, step_3d)
    ys = np.arange(bb_min[1], bb_max[1] + step_3d * 0.5, step_3d)
    zs = np.arange(bb_min[2], bb_max[2] + step_3d * 0.5, step_3d)

    # 3-D interior grid ----------------------------------------------------
    grid_xyz = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    # ---------------------------------------------------------------------
    # 3) face sampling (denser 2-D grids)
    step_2d = step_3d / oversample_surface
    faces = []

    # For each pair of axes, fix one coordinate at min or max and sample a 2-D grid
    def _surf_grid(fixed_axis, fixed_val):
        a, b = [ax for ax in range(3) if ax != fixed_axis]
        u = np.arange(bb_min[a], bb_max[a] + step_2d * 0.5, step_2d)
        v = np.arange(bb_min[b], bb_max[b] + step_2d * 0.5, step_2d)
        m, n = np.meshgrid(u, v, indexing="ij")
        pts = np.zeros((m.size, 3))
        pts[:, fixed_axis] = fixed_val
        pts[:, a] = m.ravel()
        pts[:, b] = n.ravel()
        return pts

    for axis in range(3):
        faces.append(_surf_grid(axis, bb_min[axis]))  # “minus” face
        faces.append(_surf_grid(axis, bb_max[axis]))  # “plus”  face

    surf_pts = np.vstack(faces)

    # ---------------------------------------------------------------------
    # 4) merge & return
    return np.vstack((grid_xyz, surf_pts))

if __name__ == '__main__':
    import pickle
    with open(f"/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/debug_vol_fusion/full/identified_objects.pkl", "rb") as f:
        identified_objects = pickle.load(f)

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()    
    x = sample_bbox_dense_volume(identified_objects["object_0"]["current_pointcloud"], 0.01)
    pcd.points = o3d.utility.Vector3dVector(x)
    
    o3d.io.write_point_cloud("/home/exx/Downloads/identified_objects.ply", pcd)
    
