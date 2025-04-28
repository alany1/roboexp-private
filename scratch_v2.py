import pickle

from roboexp import (
    RobotExplorationReal,
    RoboMemory,
    RoboPercept,
    RoboActReal,
)

REPLAY_FLAG = True
robo_exp = RobotExplorationReal(gripper_length=0.285, REPLAY_FLAG=REPLAY_FLAG)

robo_memory = RoboMemory(
    lower_bound=[-10, -10, -10],
    higher_bound=[10, 10, 10],
    voxel_size=0.02,
    real_camera=True,
    base_dir=None,
    similarity_thres=0.95,
    iou_thres=0.01,
)

# Set the labels
object_level_labels = [
    "background",
    "basket",
    "fruit",
    "vegetable",
    "bowl",
    "handle",
    "ceiling",
    "miscprop",
    "bottle",
    "coffeemaker",
    "mug",
    "coffeepot",
    "countertop",
    "dishwasher",
    "door",
    "doorframe",
    "dryer",
    "floor",
    "fork",
    "refrigerator",
    "cabinet",
    "knife",
    "microwave",
    "oven",
    "extractorhood",
    "plate",
    "poweroutlet",
    "soapdispenser",
    "spatula",
    "wall",
    "washingmachine",
    "window",
    "windowframe",
    "windowsill",
    "stool",
    "table",
]
part_level_labels = ["handle"]
grounding_dict = (
    " . ".join(object_level_labels) + " . " + " . ".join(part_level_labels)
)
robo_percept = RoboPercept(grounding_dict=grounding_dict, lazy_loading=False)
robo_act = RoboActReal(
    robo_exp,
    robo_percept,
    robo_memory,
    object_level_labels,
    REPLAY_FLAG=REPLAY_FLAG,
    base_dir=None,
)


with open(
    "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/debug_vol_fusion/full/identified_objects.pkl",
    "rb",
) as f:
    objects_list = pickle.load(f)
with open(
    "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/debug_vol_fusion/full/events.pkl",
    "rb",
) as f:
    events = pickle.load(f)

def compare(*, old_instances, robo_memory):
    old_instance_ids = [instance.instance_id for instance in old_instances]
    
    new_instances = []
    for instance in robo_memory:
        if instance.instance_id not in old_instance_ids:
            new_instances.append(instance)
        
    return new_instances

ts_batches = [f"/home/exx/Downloads/tmp_{i}.pkl" for i in range(9)]
fake_obs_batches = []
for ts_batch in ts_batches:
    with open(ts_batch, "rb") as f:
        fake_obs = pickle.load(f)
        fake_obs_batches.append(fake_obs)

kwargs_batches = [{}, 
                  dict(articulate_object=objects_list["object_0"], event=events[0]),
                  dict(articulate_object=objects_list["object_1"], event=events[1]),
                  {},
                  {},
                  dict(articulate_object=objects_list["object_4"], event=events[4]),
                  {},
                    dict(articulate_object=objects_list["object_6"], event=events[6]),
                    {},
                  ]

# kwargs_batches = [{}, 
#                   dict(articulate_object=objects_list["object_6"], event=events[6]),
#                   {},
#                   ]
from copy import deepcopy
contains_dict = dict()
constrained_dict = dict()

current_instances = []
for fake_obs, kwargs in zip(fake_obs_batches, kwargs_batches):
    if kwargs:
        contains, constrained = robo_act.alan_get_observations_update_memory(fake_obs, **kwargs)
        contains_newlist = compare(old_instances=current_instances, robo_memory=contains)
        constrained_newlist = compare(old_instances=contains, robo_memory=constrained)
        # contains
        contains_dict[kwargs["articulate_object"]["name"]] = deepcopy(contains_newlist)
        constrained_dict[kwargs["articulate_object"]["name"]] = deepcopy(constrained_newlist)
        print(f"contains | {[x.label for x in contains_newlist]}")
        print(f"constrained | {[x.label for x in constrained_newlist]}")
    else:
        robo_act.alan_get_observations_update_memory(fake_obs)
    current_instances = deepcopy(robo_memory.memory_instances)
    
import open3d as o3d
import numpy as np
points = robo_memory.index_to_pcd(np.array(list(robo_memory.memory_scene.keys())))
colors = np.array(list(robo_memory.memory_scene.values()))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("/home/exx/Downloads/tmp_scene.ply", pcd)


mesh_pcd = o3d.geometry.TriangleMesh()
mesh_pcd.vertices = o3d.utility.Vector3dVector(np.asarray(points))
mesh_pcd.vertex_colors = o3d.utility.Vector3dVector(
                    np.asarray(colors)
                )
o3d.io.write_triangle_mesh("/home/exx/Downloads/tmp_mesh.ply", mesh_pcd, write_ascii=False)

def write_viz_info(current_instances, contains_dict, constrained_dict, points, colors):
    contains_info = {k: [x.instance_id for x in v] for k, v in contains_dict.items()}
    constrained_info = {k: [x.instance_id for x in v] for k, v in constrained_dict.items()}
    
    all_instances = []
    parented_instance_ids = set()
    for k, v in contains_dict.items(): 
        for instance in v:
            if instance.instance_id not in parented_instance_ids:
                all_instances.append(instance)        
                parented_instance_ids.add(instance.instance_id)

    for k, v in constrained_dict.items(): 
        for instance in v:
            if instance.instance_id not in parented_instance_ids:
                all_instances.append(instance)    
                parented_instance_ids.add(instance.instance_id)

    parentless_instances = []
    for instance in current_instances:
        if instance.instance_id not in parented_instance_ids:
            parentless_instances.append(instance.instance_id)
            all_instances.append(instance)
            
    instance_id_to_bbox = {instance.instance_id: instance.get_attributes() for instance in all_instances} #  if np.all(instance.get_attributes()["size"]>0)} 
    print(len(all_instances), len(instance_id_to_bbox))    
    with open("/home/exx/Downloads/tmp_viz_info.pkl", "wb") as f:
        pickle.dump(
            {
                "contains": contains_info,
                "constrained": constrained_info,
                "parentless_instances": parentless_instances,
                "all_instance_ids": [instance.instance_id for instance in all_instances],
                "instance_id_to_bbox": instance_id_to_bbox,
                "points": points,
                "colors": colors,
            },
            f,
        )
    print(f"Done writing!")

write_viz_info(current_instances, contains_dict, constrained_dict, points, colors)
exit()

d = [x.get_attributes() for x in current_instances if np.all(x.get_attributes()["size"]>0)]
"door_1_instance" in [x.instance_id for x in current_instances]
import open3d as o3d

for instance in current_instances:
    instance_pcd = instance.index_to_pcd(instance.voxel_indexes)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(instance_pcd)
    o3d.io.write_point_cloud(f"/home/exx/Downloads/tmp_pcds/{instance.instance_id}.ply", pcd)
    
save_boxes_as_ply(d, "/home/exx/Downloads/tmp_boxes.ply")

print('done')

from pathlib import Path
from typing import List, Dict

import numpy as np
import open3d as o3d


#!/usr/bin/env python3
"""
boxes_to_ply_fixed.py  –  save a list of axis-aligned boxes to one PLY mesh
boxes = [{"center": np.ndarray(3,), "size": np.ndarray(3,)}]
"""

from pathlib import Path
from typing import List, Dict
import numpy as np
import open3d as o3d


def boxes_to_mesh(boxes: List[Dict[str, np.ndarray]]) -> o3d.geometry.TriangleMesh:
    merged = o3d.geometry.TriangleMesh()

    for box in boxes:
        center = np.asarray(box["center"], dtype=float).reshape(3)
        size   = np.asarray(box["size"],   dtype=float).reshape(3)

        # 1. make a box spanning [0,size] along x,y,z
        m = o3d.geometry.TriangleMesh.create_box(*size)

        # 2. shift it so its *centre* coincides with `center`
        m.translate(center - size / 2.0)          # relative=True by default

        # (optional) uniform colour so you can tell them apart
        m.paint_uniform_color(np.random.rand(3))

        merged += m                               # merge handles index bookkeeping

    merged.compute_vertex_normals()
    return merged


def save_boxes_as_ply(boxes: List[Dict[str, np.ndarray]], out_path: str | Path) -> None:
    mesh = boxes_to_mesh(boxes)
    out_path = Path(out_path).with_suffix(".ply")
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=True)
    print(f"Wrote {len(boxes)} boxes ➜ {out_path}")


# ------------------------------------------------------------------------
# quick check: run `python boxes_to_ply_fixed.py` to see a sample
# ------------------------------------------------------------------------
if __name__ == "__main__":
    demo = [
        {"center": np.array([0.0, 0.0, 0.0]), "size": np.array([1.0, 1.0, 2.0])},
        {"center": np.array([2.0, 1.0, 0.5]), "size": np.array([0.5, 2.0, 1.0])},
        {"center": np.array([-1.5, -0.5, 1.0]), "size": np.array([1.2, 0.8, 0.8])},
    ]
    save_boxes_as_ply(demo, "scene_boxes_fixed.ply")
