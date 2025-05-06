#!/usr/bin/env python3
"""
hierarchy_to_ply.py
-------------------
Visualise an object-to-bbox hierarchy as a single PLY mesh.

Input
-----
hierarchy = {
    "all_instance_ids": [id_0, id_1, ...],          # list[str]
    "instance_id_to_bbox": {
        id_k: {"center": np.ndarray(3,), "size": np.ndarray(3,)},
        ...
    }
}

Output
------
A PLY file that you can inspect in MeshLab / Open3D viewer, with
  • coloured boxes on z = 0            (layer 0)
  • matching spheres above at +gap_z   (layer 1)
  • skinny cylinders linking sphere → centre of its box

Instances that lack a bounding box still get a sphere; they are
laid out in a tidy row and have no link cylinder.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import open3d as o3d

def cylinder_between(p0: np.ndarray,
                     p1: np.ndarray,
                     radius: float) -> o3d.geometry.TriangleMesh:
    """
    Return a cylinder whose bottom end is at p0 and top end is at p1.
    """
    p0 = p0.astype(float)
    p1 = p1.astype(float)
    vec    = p1 - p0
    length = np.linalg.norm(vec)

    if length == 0:       # degenerate: render a tiny sphere
        return o3d.geometry.TriangleMesh.create_sphere(radius)

    # 1 build a unit-radius cylinder centred at the origin, axis = +Z
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius, height=length, resolution=16)
    cyl.compute_vertex_normals()

    # 2 rotate so its +Z axis aligns with vec
    z_axis = np.array([0., 0., 1.])
    v      = vec / length
    axis   = np.cross(z_axis, v)
    s      = np.linalg.norm(axis)
    if s > 1e-6:
        c     = np.dot(z_axis, v)
        angle = np.arctan2(s, c)
        R     = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / s * angle)
        cyl.rotate(R, center=np.zeros(3))

    # 3 translate to the segment mid-point  ➜ bottom sits at p0, top at p1
    cyl.translate((p0 + p1) / 2.0)

    return cyl
# ---------------------------------------------------------------------
# upgraded hierarchy_to_mesh  –  now handles "contains" & "constrained"
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# hierarchy_to_mesh  —  now with `show_boxes`
# ---------------------------------------------------------------------
def hierarchy_to_mesh(
    hierarchy: Dict,
    *,
    show_boxes: bool = True,            # ← draw boxes?
    restrict_relations: bool = False,   # ← keep only related objects?
    gap_z: float = 1.0,
    parent_gap_z: float = 0.6,
    node_radius: float = 0.06,
    parent_radius: float = 0.08,
    line_radius: float = 0.015,
    edge_radius: float = 0.01,
    force_node_z: float | None = None,
) -> o3d.geometry.TriangleMesh:
    """
    Visualise a 3-layer scene graph.

    Parameters
    ----------
    show_boxes        : draw axis-aligned box meshes if True
    restrict_relations: if True, keep only object nodes that appear as a
                        *child* in either 'contains' or 'constrained'
                        (and their boxes if `show_boxes` is True)
    """

    # ------------------------------------------------------------------
    # 0.  unpack
    # ------------------------------------------------------------------
    ids: list[str]          = hierarchy["all_instance_ids"]
    id2box: Dict[str, Dict] = hierarchy["instance_id_to_bbox"]
    contains_map            = hierarchy.get("contains", {})
    constrained_map         = hierarchy.get("constrained", {})

    # children that participate in relations
    related_ids = {cid
                   for rel in (contains_map, constrained_map)
                   for cids in rel.values()
                   for cid in cids}

    # filter objects if requested
    obj_ids = [i for i in ids] if not restrict_relations else [
        i for i in ids if i in related_ids
    ]

    merged = o3d.geometry.TriangleMesh()

    # ------------------------------------------------------------------
    # 1.  layer-0  (boxes)
    # ------------------------------------------------------------------
    top_z = -np.inf
    drawn_boxes: set[str] = set()

    if show_boxes:
        for inst_id in obj_ids:
            if inst_id not in id2box:
                continue
            centre = np.asarray(id2box[inst_id]["center"], float)
            size   = np.asarray(id2box[inst_id]["size"],   float)
            size   = np.maximum(size, 0.1)                       # avoid degenerate

            m = o3d.geometry.TriangleMesh.create_box(*size)
            m.translate(centre - size / 2)
            m.paint_uniform_color(np.random.rand(3))
            merged += m
            drawn_boxes.add(inst_id)
            top_z = max(top_z, centre[2] + size[2] / 2)

    if top_z == -np.inf:          # no boxes at all
        top_z = 0.0

    node_z   = force_node_z if force_node_z is not None else (top_z + gap_z)
    parent_z = node_z + parent_gap_z

    # ------------------------------------------------------------------
    # 2.  bounds for orphan layout
    # ------------------------------------------------------------------
    if id2box:
        all_c = np.vstack([b["center"] for b in id2box.values()])
        span_x, span_y = all_c[:, 0].ptp(), all_c[:, 1].ptp()
        min_x, max_y   = all_c[:, 0].min(), all_c[:, 1].max()
    else:
        span_x = span_y = 0.0
        min_x  = max_y = 0.0

    orphan_dx  = max(span_x, 1.0) * 0.15
    orphan_row = max_y + max(span_y, 1.0) * 0.2
    orphan_col = 0

    # ------------------------------------------------------------------
    # 3.  layer-1  (object nodes)
    # ------------------------------------------------------------------
    node_pos: Dict[str, np.ndarray] = {}

    for inst_id in obj_ids:
        has_box = inst_id in id2box
        centre  = np.asarray(id2box[inst_id]["center"], float) if has_box else None

        if has_box:
            pos = np.array([centre[0], centre[1], node_z])
        else:                                  # orphan object, still allowed
            pos = np.array([min_x + orphan_col * orphan_dx,
                            orphan_row,
                            node_z])
            orphan_col += 1

        node_pos[inst_id] = pos

        sphere = o3d.geometry.TriangleMesh.create_sphere(node_radius, 20)
        sphere.translate(pos)
        sphere.paint_uniform_color([0.1, 0.8, 0.1] if has_box else [0.6, 0.6, 0.6])
        merged += sphere

        if has_box:
            line = cylinder_between(pos, centre, line_radius)
            line.paint_uniform_color([0.2, 0.2, 0.2])
            merged += line

    # ------------------------------------------------------------------
    # 4.  layer-2  (parent nodes)  + edges
    # ------------------------------------------------------------------
    parent_nodes = set(contains_map) | set(constrained_map)
    parent_orphan_col = 0

    for parent in parent_nodes:
        children_xy = [node_pos[c][:2] for rel in (contains_map, constrained_map)
                       for c in rel.get(parent, []) if c in node_pos]

        if children_xy:
            xy = np.mean(children_xy, axis=0)
            pos = np.array([xy[0], xy[1], parent_z])
        else:                                   # parent with unseen children
            pos = np.array([min_x + parent_orphan_col * orphan_dx,
                            orphan_row + orphan_dx,
                            parent_z])
            parent_orphan_col += 1

        node_pos[parent] = pos

        sphere = o3d.geometry.TriangleMesh.create_sphere(parent_radius, 20)
        sphere.translate(pos)
        sphere.paint_uniform_color([1.0, 0.2, 0.2])    # red parent
        merged += sphere

    def add_edges(rel_map: Dict[str, List[str]], colour):
        nonlocal merged
        for parent, child_ids in rel_map.items():
            p = node_pos.get(parent)
            if p is None:
                continue
            for cid in child_ids:
                c = node_pos.get(cid)
                if c is None:
                    continue
                e = cylinder_between(p, c, edge_radius)
                e.paint_uniform_color(colour)
                merged += e

    add_edges(contains_map,    [1.0, 0.6, 0.0])   # orange  = contains
    add_edges(constrained_map, [0.0, 0.4, 1.0])   # blue    = constrained

    # ------------------------------------------------------------------
    # 5.  reference point cloud (unchanged)
    # ------------------------------------------------------------------
    pts, cols = hierarchy.get("points"), hierarchy.get("colors")
    if pts is not None and len(pts):
        mesh_pcd = o3d.geometry.TriangleMesh()
        mesh_pcd.vertices  = o3d.utility.Vector3dVector(np.asarray(pts))
        mesh_pcd.triangles = o3d.utility.Vector3iVector(          # keep every vertex
            np.repeat(np.arange(len(pts))[:, None], 3, axis=1)
        )
        if cols is not None and len(cols) == len(pts):
            mesh_pcd.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cols))
        else:
            mesh_pcd.vertex_colors = o3d.utility.Vector3dVector(
                np.full((len(pts), 3), 0.5))
        merged += mesh_pcd

    merged.compute_vertex_normals()
    return merged


def save_hierarchy_as_ply(hierarchy: Dict,
                          out_file: str | Path,
                          **kwargs) -> None:
    """
    Convenience wrapper: build → write (PLY, ASCII)
    """
    mesh = hierarchy_to_mesh(hierarchy, **kwargs)
    out_file = Path(out_file).with_suffix(".ply")
    o3d.io.write_triangle_mesh(str(out_file), mesh, write_ascii=True)
    print(f"✓ Wrote hierarchy visualisation ➜ {out_file}")


# ---------------------------------------------------------------------
# quick demonstration if you run this file directly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo_hierarchy = {
        "all_instance_ids": ["drawer", "bottle", "plate", "missing_box"],
        "instance_id_to_bbox": {
            "drawer":  {"center": np.array([0., 0., 0.]),   "size": np.array([0.4, 0.6, 0.2])},
            "bottle":  {"center": np.array([1.2, 0.5, 0.1]), "size": np.array([0.1, 0.1, 0.3])},
            "plate":   {"center": np.array([-0.8, 0.3, 0.05]),"size": np.array([0.3, 0.3, 0.02])},
        },
    }
    
    import pickle
    with open("/home/exx/Downloads/tmp_viz_info.pkl", "rb") as f:
        demo_hierarchy = pickle.load(f)
        
    instance_id_to_box = demo_hierarchy["instance_id_to_bbox"]
    
    len(demo_hierarchy["all_instance_ids"])
    len(instance_id_to_box)
    
    for inst_id, box in instance_id_to_box.items():
        if not np.all(box["size"]>0):
            print('wtf')

    # save_hierarchy_as_ply(demo_hierarchy, "/home/exx/Downloads/hierarchy_vis.ply",
    #                       gap_z=5, node_radius=0.05, line_radius=0.01)
    mesh_pcd = o3d.geometry.TriangleMesh()
    mesh_pcd.vertices = o3d.utility.Vector3dVector(np.asarray(demo_hierarchy["points"]))
    mesh_pcd.vertex_colors = o3d.utility.Vector3dVector(np.asarray(demo_hierarchy["colors"]))
    o3d.io.write_triangle_mesh(
        "/home/exx/Downloads/tmp_mesh.ply", mesh_pcd, write_ascii=False
    )
    # exit()
    demo_hierarchy.keys()
    
    save_hierarchy_as_ply(
        demo_hierarchy,
        "/home/exx/Downloads/hierarchy_vis.ply",
        gap_z=2.0,
        parent_gap_z=2.0,
        node_radius=0.05,
        parent_radius=0.07,
        line_radius=0.012,
        edge_radius=0.008,
        show_boxes=False,
    )
