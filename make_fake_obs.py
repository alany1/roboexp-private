from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import os

root = "/home/exx/datasets/aria/blender_eval/bedroom"


with open(f"{root}/keyframes.pkl", "rb") as f:
    keyframes = pickle.load(f)

save_dir = f"{root}/sg_obs"
os.makedirs(save_dir, exist_ok=True)

all_ts = []
frames = []
obs_per_kf = 10
for start, end in keyframes:
    all_ts.append(list(range(start, end+1, int((end-start+1)/(obs_per_kf-1)))))
    frames.extend(all_ts[-1])
    
with open(f"{save_dir}/all_ts.json", "w") as f:
    import json
    json.dump(frames, f)

# with open(f"{save_dir}/last.json", "w") as f:
#     import json
#     json.dump(all_ts[-1], f)

print(all_ts)
def make(ts):
    rgb = f"{root}/renders/rgb/{ts:04d}.jpg"
    rgb = Image.open(rgb)
    rgb = np.array(rgb) / 255
    
    semantics = f"{root}/renders/semantics_vis/semantic_{ts:04d}.vis.png"
    semantics = Image.open(semantics)
    semantics = np.array(semantics)
    
    import OpenEXR, Imath
    def load_exr_depth(path):
        f = OpenEXR.InputFile(path)
        dw = f.header()["dataWindow"]  # (min.x, min.y)-(max.x, max.y)
        H, W = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
        # Blender’s File‑Output node writes the slot into channel **R**
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        depth = np.frombuffer(f.channel("R", FLOAT), dtype=np.float32).reshape(H, W)
        return depth
    
    depth = f"{root}/renders/depth/Depth{ts:04d}.exr"
    depth = load_exr_depth(depth)
    import json
    with open(f"{root}/trajectory.json", "r") as f:
        camera_info = json.load(f)
    intrinsics = np.array(camera_info["K"])
    c2w = np.array(camera_info["poses"][ts])

    R_bl = np.array(
        [
            [1, 0, 0],  # mine X (down)  →  –Y in Blender
            [0, -1, 0],  # mine Y (left)  →  –X
            [0, 0, -1],
        ]
    )
    R4 = np.eye(4, dtype=float)
    R4[:3, :3] = R_bl  # embed as 4×4

    c2w = c2w @ R4

    mask = depth < 4.5
    
    from utils import backproject_points_cam
    pts = backproject_points_cam(depth=depth,
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
        depths=depth,
        semantics=semantics,
        mask=mask,
        c2w=c2w,
        intrinsic=intrinsics,
        dist_coef=dist_coef,
    )
    return fake_obs


for i, ts_batch in tqdm(enumerate(all_ts)):
    fake_obs = dict()
    for ts in ts_batch:
        fake_obs[f"fake_{ts}"] = make(ts)
    
    with open(f"{save_dir}/tmp_{i}.pkl", "wb") as f:
        pickle.dump(fake_obs, f)
