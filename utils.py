from scipy.ndimage import binary_dilation
import numpy as np
import torch


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
