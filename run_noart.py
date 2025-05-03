import pickle

from roboexp import (
    RobotExplorationReal,
    RoboMemory,
    RoboPercept,
    RoboActReal,
)

from utils import translate_pcd, rotate_pcd, pick
import numpy as np
from copy import deepcopy
import os

def run():
    REPLAY_FLAG = True
    robo_exp = RobotExplorationReal(gripper_length=0.285, REPLAY_FLAG=REPLAY_FLAG)
    
    robo_memory = RoboMemory(
        lower_bound=[-10, -10, -10],
        higher_bound=[10, 10, 10],
        voxel_size=0.037,
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
    
    ts_batches = [f"/home/exx/Downloads/semantics_test/tmp_{i}.pkl" for i in range(9)]
    fake_obs_batches = []
    for ts_batch in ts_batches:
        with open(ts_batch, "rb") as f:
            fake_obs = pickle.load(f)
            fake_obs_batches.append(fake_obs)
    
    kwargs_batches = [
                      {}, 
                      dict(articulate_object=objects_list["object_0"], event=events[0], discovery=True),
                      dict(articulate_object=objects_list["object_1"], event=events[1], discovery=True),
                      dict(articulate_object=objects_list["object_0"], event=events[2], discovery=False),
                      dict(articulate_object=objects_list["object_1"], event=events[3], discovery=False),
                      dict(articulate_object=objects_list["object_4"], event=events[4], discovery=True),
                      dict(articulate_object=objects_list["object_4"], event=events[5], discovery=False),
                      dict(articulate_object=objects_list["object_6"], event=events[6], discovery=True),
                      dict(articulate_object=objects_list["object_6"], event=events[7], discovery=False),
                      ]
    
    if len(fake_obs_batches) != len(kwargs_batches):
        kwargs_batches = kwargs_batches[:len(fake_obs_batches)]
        print(f"Warning: fake_obs_batches and kwargs_batches are not the same length. Truncating kwargs_batches to match fake_obs_batches.")
    
    def dump_state(*, current_instances, contains_dict, constrained_dict, robo_memory, save_path):
        points = robo_memory.index_to_pcd(np.array(list(robo_memory.memory_scene.keys())))
        colors = np.array(list(robo_memory.memory_scene.values()))
        # save the full state
        with open(save_path, "wb") as f:
            data = dict(
                current_instances=[x.get_dict_spark() for x in current_instances],
                contains_dict={k: [x.get_dict_spark() for x in v] for k, v in contains_dict.items()},
                constrained_dict={k: [x.get_dict_spark() for x in v] for k, v in constrained_dict.items()},
                points=points,
                colors=colors,
            )
            pickle.dump(data, f)
    
    contains_dict = dict()
    constrained_dict = dict()
    
    log_root = "/home/exx/Downloads/noart_states_v9"
    os.makedirs(log_root, exist_ok=True)
    current_instances = []
    for kfid, (fake_obs, kwargs) in enumerate(zip(fake_obs_batches, kwargs_batches)):
        if kwargs and kwargs["discovery"]:
                contains = robo_act.no_art_get_observations_update_memory(fake_obs, **kwargs)
                contains_newlist = compare(old_instances=current_instances, robo_memory=contains)
                contains_dict[kwargs["articulate_object"]["name"]] = deepcopy(contains_newlist)
                print(f"contains | {[x.label for x in contains_newlist]}")
        else:
            robo_act.no_art_get_observations_update_memory(fake_obs, **kwargs)
        
        current_instances = deepcopy(robo_memory.memory_instances)

        print(f"saved state {kfid}")
    
    robo_act.prune_memory(objects_list, contains_dict, constrained_dict)
    dump_state(current_instances=robo_memory.memory_instances,
               contains_dict=contains_dict,
               constrained_dict=constrained_dict,
               robo_memory=robo_memory,
               save_path=os.path.join(log_root, "final_state.pkl"))
    print("saved final state")

if __name__ == '__main__':
    run()
