from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, ImageBackground, Sphere, PointCloud
from asyncio import sleep
from utils import backproject_points_cam

app = Vuer()

import pickle
import numpy as np

with open("/home/exx/Downloads/obs.pkl", "rb") as f:
    d = pickle.load(f)
    
before_pcd_cam = np.load("/home/exx/Downloads/before_pcd_cam.npy")
after_pcd_cam = np.load("/home/exx/Downloads/after_pcd_cam.npy")

@app.spawn(start=True)
async def main(sess: VuerSession):
    sess.set @ DefaultScene()
    
    # before_pcd_cam = np.load("/home/exx/Downloads/before_pcd_cam.npy")
    # after_pcd_cam = np.load("/home/exx/Downloads/after_pcd_cam.npy")

    before_obj_depth = np.load("/home/exx/Downloads/obj_depth.npy")
    before_obj_depth_mask = before_obj_depth > 0
    
    pts = backproject_points_cam(depth=d["obs"]["depths"], pose=d["obs"]["c2w"], fx=381.31536171,
                           fy=381.31536171, cx=352, cy=352, device="cpu", aria_rot=False)
    pts = pts[d["mask"]]
    
    w, h = int(d["obs"]["intrinsic"][0, 2]), int(d["obs"]['intrinsic'][1, 2])
    depth_check_mask = np.zeros((h, w), dtype=bool)
    depth_check_mask = np.logical_and(d["mask"], before_obj_depth_mask)
    
    # for every point in the before_pcd, get the corresponding pixel
    from matplotlib import pyplot as plt
    plt.imshow(depth_check_mask); plt.show()
    d["obs"]["depths"][depth_check_mask] = before_obj_depth[depth_check_mask]
    
    plt.imshow(d["mask"]); plt.show()
    
    pts_ranges = np.linalg.norm(pts, axis=1)
    pts_ranges.mean()
    
    
    
    before_viz = PointCloud(key="before", vertices=before_pcd_cam, color="red", size=0.05)
    after_viz = PointCloud(key="after", vertices=after_pcd_cam, color="blue", size=0.05)
    
    sess.upsert @ before_viz
    sess.upsert @ after_viz
    
    sess.upsert @ PointCloud(key="object", vertices=pts.cpu().numpy(), color="yellow", size=0.05)
    
    while True:
        await sleep(0.1)
