import glob
import os
import pickle

import numpy as np
import open3d as o3d  # pip install open3d
from scipy.spatial import cKDTree


def sample_mesh_surface(
    filename,
    num_points=100000,  # Number of points to sample
    voxel_size=None,
):  # E.g., 0.05 for 5 cm
    """
    Loads a mesh from a file (e.g., PLY) with Open3D, samples a dense
    point cloud on its surface, and optionally applies voxel downsampling.

    Parameters
    ----------
    filename : str
        Path to the mesh file.
    num_points : int
        Number of points to sample on the mesh surface.
    voxel_size : float or None
        If provided, use voxel downsampling at this size (in meters).
        For instance, 0.05 -> 5 cm.

    Returns
    -------
    pcd : o3d.geometry.PointCloud
        A (possibly) downsampled point cloud sampled from the mesh surface.
    """
    # Read the mesh
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_triangle_normals()  # Make sure normals are computed

    # Sample the mesh's surface. Poisson disk yields a more uniform distribution:
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

    # If you want even denser sampling, you can also use sample_points_uniformly
    # pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    # Optionally downsample to a specific spacing
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return pcd

def evaluate_background_mesh(gt_ply_path: str, est_ply_path, thresh: float = 0.20):
    """
    Parameters
    ----------
    gt_ply_path   : path to the ground-truth mesh/point-cloud (.ply)
    est_ply_path  : path to the Khronos reconstructed mesh (.ply)
    thresh        : distance threshold in **metres** (default 0.20 m)

    Returns
    -------
    dict  {'precision': P, 'recall': R, 'F1': F}
    """

    # ---------- load vertices ----------
    gt_mesh = o3d.io.read_triangle_mesh(gt_ply_path)

    if type(est_ply_path) == str:
        est_mesh = o3d.io.read_triangle_mesh(est_ply_path)
    elif type(est_ply_path) == dict:
        est_mesh = o3d.geometry.TriangleMesh()
        est_mesh.vertices = o3d.utility.Vector3dVector(est_ply_path["vertices"])
        est_mesh.triangles = o3d.utility.Vector3iVector(est_ply_path["faces"])
    elif type(est_ply_path) == np.ndarray:
        est_mesh = o3d.geometry.TriangleMesh()
        est_mesh.vertices = o3d.utility.Vector3dVector(est_ply_path)
    else:
        raise ValueError("oops")

    gt_xyz = np.asarray(gt_mesh.vertices, dtype=np.float32)
    est_xyz = np.asarray(est_mesh.vertices, dtype=np.float32)

    if len(gt_xyz) == 0 or len(est_xyz) == 0:
        raise ValueError("One of the meshes has no vertices.")

    # ---------- nearest-neighbour look-ups ----------
    gt_kdtree = cKDTree(gt_xyz)
    est_kdtree = cKDTree(est_xyz)

    # each reconstructed vertex → nearest GT
    dist_pred_to_gt, _ = gt_kdtree.query(est_xyz)
    tp_pred = np.count_nonzero(dist_pred_to_gt <= thresh)

    # each GT vertex → nearest reconstruction
    dist_gt_to_pred, _ = est_kdtree.query(gt_xyz)
    tp_gt = np.count_nonzero(dist_gt_to_pred <= thresh)

    # ---------- metrics ----------
    precision = tp_pred / (len(est_xyz) + 1e-9)
    recall = tp_gt / (len(gt_xyz) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {"precision": precision, "recall": recall, "F1": f1}

if __name__ == '__main__':
    static_objects_map = {}
    dynamic_objects_map = {}
    mesh_map = {}

    final_state = "/home/exx/Downloads/spark_states_v9/final_state.pkl"
    
    with open(final_state, "rb") as f:
        final_state = pickle.load(f)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_state["points"])
    o3d.io.write_point_cloud("/home/exx/Downloads/tmp_pcd.ply", pcd)
    
    
    gt_mesh_path = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/mesh_labels/background_mesh.ply"
    gt_pcd_path = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/mesh_labels/pcd_background_mesh.ply"
    
    if not os.path.exists(gt_pcd_path):
        # sample it
        print(f"Sampling {gt_mesh_path} to {gt_pcd_path}...")
        out = sample_mesh_surface(gt_mesh_path, num_points=100_000)
        o3d.io.write_point_cloud(gt_pcd_path, out)

    scores = evaluate_background_mesh(gt_pcd_path,
                                      final_state["points"],
                                      thresh=0.2)  # 20 cm as in the paper


    print(f"Average metrics: {scores}")
