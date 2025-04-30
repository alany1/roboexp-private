from PIL import Image
import numpy as np
import pickle

root = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901"

# all_ts = [[30, 60, 90, 120, 150], [200, 210, 220, 230], [340, 350, 360, 380], [505, 525, 535, 550], [626, 630, 635], [675, 700, 740, 770], [825, 850, 875, 890], [1000, 1050, 1100, 1150], [1260, 1350, 1400, 1450], [1550, 1555, 1560, 1565]]

with open(f"{root}/keyframes.pkl", "rb") as f:
    keyframes = pickle.load(f)
    
all_ts = []
frames_per_obs = 10
for start, end in keyframes[:-1]:
    all_ts.append(list(range(start, end, frames_per_obs)))
    
all_ts.append([1550, 1555, 1560, 1565])

def make(ts):
    rgb = f"{root}/renders/rgb/{ts:04d}.jpg"
    rgb = Image.open(rgb)
    rgb = np.array(rgb) / 255
    
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

    # R_bl = np.array(
    #     [
    #         [0, -1, 0],  # mine X (down)  →  –Y in Blender
    #         [-1, 0, 0],  # mine Y (left)  →  –X
    #         [0, 0, -1],
    #     ]
    # )  # mine Z (fwd)   →  –Z
    R_bl = np.array(
        [
            [1, 0, 0],  # mine X (down)  →  –Y in Blender
            [0, -1, 0],  # mine Y (left)  →  –X
            [0, 0, -1],
        ]
    )  # mine Z (fwd)   →  –Z
    R4 = np.eye(4, dtype=float)
    R4[:3, :3] = R_bl  # embed as 4×4

    # post-multiply so the rotation acts in camera space
    c2w = c2w @ R4  # NOT R4 @ pose

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
        mask=mask,
        c2w=c2w,
        intrinsic=intrinsics,
        dist_coef=dist_coef,
    )
    return fake_obs

for i, ts_batch in enumerate(all_ts):
    fake_obs = dict()
    for ts in ts_batch:
        fake_obs[f"fake_{ts}"] = make(ts)
    
    with open(f"/home/exx/Downloads/even_denser/tmp_{i}.pkl", "wb") as f:
        pickle.dump(fake_obs, f)
