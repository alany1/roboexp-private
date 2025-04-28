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
