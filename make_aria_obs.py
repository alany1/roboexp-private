from PIL import Image
import numpy as np
import pickle
from transformers import pipeline

from utils import fit_depth_hyperbola, apply_depth_hyperbola

USE_EST_DEPTH = True

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device="cuda")

root = "/home/exx/datasets/aria/real/kitchen_v2/vol_fusion_v3_hand_detector_combination"
# for masking out the human
instances_dir = "/home/exx/datasets/aria/real/kitchen_v2/processed/instances"

# all_ts = [[30, 60, 90, 120, 150], [200, 210, 220, 230], [340, 350, 360, 380], [505, 525, 535, 550], [626, 630, 635], [675, 700, 740, 770], [825, 850, 875, 890], [1000, 1050, 1100, 1150], [1260, 1350, 1400, 1450], [1550, 1555, 1560, 1565]]

with open(f"{root}/keyframes.pkl", "rb") as f:
    keyframes = pickle.load(f)
    
all_ts = []
obs_per_kf = 15
for start, end in keyframes:
    all_ts.append(list(range(start, end+1, int((end-start+1)/(obs_per_kf-1)))))

# all_ts = all_ts[-2:]
    
print(all_ts)

background_classes = [0, 1, 3, 5, 15, 45, ]

def make(ts):
    rgb = f"{root}/sg_obs/rgb/rgb_{ts:04d}.png"
    rgb_pil = Image.open(rgb)
    rgb = np.array(rgb_pil) / 255
    rgb = np.rot90(rgb, 1).copy()
    
    instance = f"{instances_dir}/instance_{ts:07d}.png"
    instance = Image.open(instance)
    instance = np.array(instance)
    filter_mask = instance == 12
    # downsample
    filter_mask = np.array(Image.fromarray(filter_mask).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
    filter_mask = np.rot90(filter_mask, 1).copy()
    
    depth = f"{root}/sg_obs/depth/depth_{ts:04d}.png"
    depth = Image.open(depth)
    depth = np.array(depth) / 1000
    # remove human
    depth[filter_mask] = 0
    
    fit_mask = np.isin(instance, background_classes)
    fit_mask = np.array(Image.fromarray(fit_mask).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
    
    if USE_EST_DEPTH:
        est_depth = pipe(rgb_pil)["depth"]
        est_depth = np.array(est_depth) / 255
        
        try:
            S, c, mae, rel = fit_depth_hyperbola(depth, est_depth, mask=fit_mask, depth_min=1, depth_max=3)
        except ValueError:
            S = 1
            c = 1
            mae = 1_000
            rel = 1_000
        est_metric_depth = apply_depth_hyperbola(est_depth, S, c,)
    else:
        est_metric_depth = depth
        mae = 0
        
    est_metric_depth = np.rot90(est_metric_depth, 1).copy()
        
    with open(f"{root}/sg_obs/camera_info.pkl", "rb") as f:
        camera_info = pickle.load(f)
        
    intrinsics = np.array(camera_info["intrinsics"])
    c2w = np.array(camera_info["poses"][ts])

    mask = np.logical_and(depth > 0, depth < 4.5)    
    from utils import backproject_points_cam
    pts = backproject_points_cam(depth=est_metric_depth,
                       pose=c2w,
                        fx=intrinsics[0, 0],
                        fy=intrinsics[1, 1],
                        cx=intrinsics[0, 2],
                        cy=intrinsics[1, 2],
                        device="cpu",
                             aria_rot=False).numpy()

    dist_coef = np.zeros(5,)
    
    fake_obs = dict(
        position=pts,
        rgb=rgb,
        depths=est_metric_depth,
        mask=mask,
        c2w=c2w,
        intrinsic=intrinsics,
        dist_coef=dist_coef,
        filter_mask=filter_mask,
    )
    return fake_obs, mae


from tqdm import tqdm
import os

save_dir = "/home/exx/Downloads/aria_obs_est"
os.makedirs(save_dir, exist_ok=True)
for i, ts_batch in tqdm(enumerate(all_ts)):
    fake_obs = dict()
    for j, ts in enumerate(ts_batch):
        result, mae = make(ts)
        if mae > 0.08:
            print(f"skipping {ts} with mae {mae}")
            continue
        fake_obs[f"fake_{ts}"] = result

    # all_pts_world = []
    # for name, obs in fake_obs.items():
    #     c2w = obs["c2w"]
    #     pts_camera = obs["position"]
    #     mask = (~obs["filter_mask"] & obs["mask"])
    # 
    #     pts_camera = pts_camera[mask].reshape(-1, 3)
    #     # homogenize
    #     pts_camera = np.concatenate([pts_camera, np.ones((pts_camera.shape[0], 1))], axis=1)
    #     pts_world = (c2w @ pts_camera.T).T
    #     all_pts_world.append(pts_world)
    # 
    # all_pts_world = np.concatenate(all_pts_world, axis=0)
    # 
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(all_pts_world[:, :3])
    # o3d.io.write_point_cloud(f"/home/exx/Downloads/proj.ply", pcd)
        
    
    if len(fake_obs) == 0:
        print(f"No valid observations for {i}")
        exit()
    
    with open(f"{save_dir}/tmp_{i}.pkl", "wb") as f:
        pickle.dump(fake_obs, f)
