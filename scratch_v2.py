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

ts_batches = [f"/home/exx/Downloads/tmp_{i}.pkl" for i in range(6,9)]
fake_obs_batches = []
for ts_batch in ts_batches:
    with open(ts_batch, "rb") as f:
        fake_obs = pickle.load(f)
        fake_obs_batches.append(fake_obs)

# kwargs_batches = [{}, 
#                   dict(articulate_object=objects_list["object_0"], event=events[0]),
#                   dict(articulate_object=objects_list["object_1"], event=events[1]),
#                   {},
#                   {},
#                   dict(articulate_object=objects_list["object_4"], event=events[4]),
#                   ]

kwargs_batches = [{}, 
                  dict(articulate_object=objects_list["object_6"], event=events[6]),
                  {},
                  ]
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

key = "object_6"
[x.instance_id for x in contains_dict[key]]
[x.instance_id for x in constrained_dict[key]]
# 
"door_1_instance" in [x.instance_id for x in current_instances]
import open3d as o3d

for instance in current_instances:
    instance_pcd = instance.index_to_pcd(instance.voxel_indexes)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(instance_pcd)
    o3d.io.write_point_cloud(f"/home/exx/Downloads/tmp_pcds/{instance.instance_id}.ply", pcd)
    

print('done')

# with open("/home/exx/Downloads/tmp.pkl", "rb") as f:
#     fake_obs = pickle.load(f)
# 
# robo_decision = RoboDecision(robo_memory, None, REPLAY_FLAG=REPLAY_FLAG)
# from copy import deepcopy
# # discovery step
# robo_act.alan_get_observations_update_memory(fake_obs)
# newlist = compare(old_instances=[], robo_memory=robo_memory)
# print([x.label for x in newlist])
# current_instances = deepcopy(robo_memory.memory_instances)
# 
# with open("/home/exx/Downloads/tmp_2.pkl", "rb") as f:
#     fake_obs_2 = pickle.load(f)
# 
# robo_act.alan_get_observations_update_memory(fake_obs_2, articulate_object=objects_list["object_0"], event=events[0])
# newlist = compare(old_instances=current_instances, robo_memory=robo_memory)
# print([x.label for x in newlist])
# current_instances = deepcopy(robo_memory.memory_instances)
# 
# with open("/home/exx/Downloads/tmp_3.pkl", "rb") as f:
#     fake_obs_3 = pickle.load(f)
# 
# robo_act.alan_get_observations_update_memory(fake_obs_3, articulate_object=objects_list["object_1"], event=events[1])
# newlist = compare(old_instances=current_instances, robo_memory=robo_memory)
# print([x.label for x in newlist])
# current_instances = deepcopy(robo_memory.memory_instances)
# 

