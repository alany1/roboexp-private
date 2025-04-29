import pickle
from asyncio import sleep

import numpy as np
import open3d as o3d
import torch
from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, PointCloud, TriMesh, ImageBackground, Box, group
import trimesh


app = Vuer(queries=dict(show_grid=False))

hierarchy_vis_path = "/home/exx/Downloads/hierarchy_vis.ply"
hierarchy_mesh = o3d.io.read_triangle_mesh(hierarchy_vis_path)
hierarchy_vertices = np.asarray(hierarchy_mesh.vertices)
hierarchy_faces = np.asarray(hierarchy_mesh.triangles)
hierarchy_colors = (np.asarray(hierarchy_mesh.vertex_colors)*255).astype(np.uint8)

pcd_path = "/home/exx/Downloads/tmp_mesh.ply"
pcd_mesh = o3d.io.read_point_cloud(pcd_path)
pcd_points = np.asarray(pcd_mesh.points)
pcd_colors = (np.asarray(pcd_mesh.colors)*255).astype(np.uint8)

@app.spawn(start=True)
async def main(sess: VuerSession):
    sess.set @ DefaultScene(
        TriMesh(vertices=hierarchy_vertices, faces=hierarchy_faces, colors=hierarchy_colors),
        PointCloud(vertices=pcd_points, colors=pcd_colors, size=0.1 ),
        rotation=[-np.pi/2, 0, 0],
    )

    while True:

        await sleep(1 / 24)

    print("Done")

