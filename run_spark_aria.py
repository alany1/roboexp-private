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
        voxel_size=0.02,
        real_camera=True,
        base_dir=None,
        similarity_thres=0.95,
        iou_thres=0.01,
    )
    
    # Set the labels
    object_level_labels = [
        "cup",
        "fruit",
        # "vegetable",
        "bowl",
        "bottle",
        # "coffeemaker",
        # "mug",
        # "dishwasher",
        # "door",
        # "fork",
        # "knife",
        # "microwave",
        # "oven",
        # "plate",
        # "spatula",
        "pillow",
        "stool",
        "monitor",
        "scissors",
        "keyboard",
        "handle",
    ]
    # part_level_labels = ["handle"]
    part_level_labels = []
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
    
    
    # vol_fusion_root = "/home/exx/datasets/aria/real/kitchen_v2/vol_fusion_v3_hand_detector_combination"
    # aria_obs_dir = "/home/exx/Downloads/aria_obs_est"
    
    vol_fusion_root = "/home/exx/datasets/aria/real/spot_room_v1/vol_fusion_v1"
    aria_obs_dir = "/home/exx/Downloads/spot_room_v1_obs/"
    
    with open(
        f"{vol_fusion_root}/identified_objects.pkl",
        "rb",
    ) as f:
        objects_list = pickle.load(f)
    with open(
        f"{vol_fusion_root}/events.pkl",
        "rb",
    ) as f:
        events = pickle.load(f)
    with open(f"{vol_fusion_root}/keyframes.pkl", "rb") as f:
        keyframes = pickle.load(f)
    
    def compare(*, old_instances, robo_memory):
        old_instance_ids = [instance.instance_id for instance in old_instances]
        
        new_instances = []
        for instance in robo_memory:
            if instance.instance_id not in old_instance_ids:
                new_instances.append(instance)
            
        return new_instances
    
    ts_batches = [f"{aria_obs_dir}/tmp_{i}.pkl" for i in range(7)]
    fake_obs_batches = []
    for ts_batch in ts_batches:
        with open(ts_batch, "rb") as f:
            fake_obs = pickle.load(f)
            fake_obs_batches.append(fake_obs)
            print(f"{len(fake_obs)} observations loaded from {ts_batch}")
    
    events_copy = deepcopy(events)
    kwargs_batches = []
    discovered_objects = set()
    for start, end in keyframes:
        if events_copy and events_copy[0]["start_ts"] <= start:
            event = events_copy.pop(0)
            discovery = False
            if event["object_name"] not in discovered_objects:
                discovery = True
                discovered_objects.add(event["object_name"])
            kwargs_batches.append(dict(event=event,
                                       articulate_object=objects_list[event["object_name"]],
                                        discovery=discovery))
        else:
            kwargs_batches.append({})
    
    # kwargs_batches = [
    #     {},
    #     dict(
    #         articulate_object=objects_list["object_0"], event=events[0], discovery=True
    #     ),
    #     dict(
    #         articulate_object=objects_list["object_0"], event=events[1], discovery=False
    #     ),
    #     dict(
    #         articulate_object=objects_list["object_2"], event=events[2], discovery=True
    #     ),
    #     dict(
    #         articulate_object=objects_list["object_2"], event=events[3], discovery=False
    #     ),
    #     dict(
    #         articulate_object=objects_list["object_4"], event=events[4], discovery=True
    #     ),
    #     dict(
    #         articulate_object=objects_list["object_4"], event=events[5], discovery=False
    #     ),
    # ]
    
    # assert len(fake_obs_batches) == len(kwargs_batches), "fake_obs and kwargs batches must be the same length"
    if len(fake_obs_batches) != len(kwargs_batches):
        kwargs_batches = kwargs_batches[:len(fake_obs_batches)]
        print("Warning: fake_obs_batches and kwargs_batches are not the same length. Truncating kwargs_batches to match fake_obs_batches.")
    
    # fake_obs_batches = fake_obs_batches[-2:]
    # kwargs_batches = kwargs_batches[-2:]
    
    
    def set_parent_tf(*, articulate_object, event, related_objects):
        # store the objects position in the world frame, but when the parent object is in its canonical pose.
        # retrieve the objects angle from the current event, and then rotate it into canonical position.
        articulation_type = articulate_object["joint_type"]
        
        assert articulation_type in ["prismatic", "revolute"]
        
        articulation_kwargs = {}
        func = translate_pcd
        if articulation_type == "prismatic":
            articulation_kwargs["direction"] = articulate_object["articulation_params"]["translation_dir"]
            articulation_kwargs["amount"] = -event["art_params"]["translation"]
        else:
            articulation_kwargs["hinge_axis"] = articulate_object["articulation_params"]["rotation_dir"]
            articulation_kwargs["hinge_pivot"] = articulate_object["articulation_params"]["rotation_point"]
            articulation_kwargs["rad"] = -event["art_params"]["angle"]
            
            func = rotate_pcd
        
        for obj in related_objects:
            obj_current_pcd = obj.index_to_pcd(obj.voxel_indexes)
            obj.canonical_pcd = func(pcd=obj_current_pcd,
                                     **articulation_kwargs)
        
    
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
    
    log_root = "/home/exx/Downloads/spot_room_v1_sg_est"
    os.makedirs(log_root, exist_ok=True)
    current_instances = []
    for kfid, (fake_obs, kwargs) in enumerate(zip(fake_obs_batches, kwargs_batches)):
        if kwargs and kwargs["discovery"]:
                contains, constrained = robo_act.alan_get_observations_update_memory(fake_obs, visualize=True, **kwargs)
                contains_newlist = compare(old_instances=current_instances, robo_memory=contains)
                constrained_newlist = compare(old_instances=contains, robo_memory=constrained)            
                set_parent_tf(related_objects=constrained_newlist, **pick(kwargs, "articulate_object", "event"))
                contains_dict[kwargs["articulate_object"]["name"]] = deepcopy(contains_newlist)
                constrained_dict[kwargs["articulate_object"]["name"]] = deepcopy(constrained_newlist)
                print(f"contains | {[x.label for x in contains_newlist]}")
                print(f"constrained | {[x.label for x in constrained_newlist]}")
        else:
            robo_act.alan_get_observations_update_memory(fake_obs, visualize=True, **kwargs)
        
        current_instances = deepcopy(robo_memory.memory_instances)
        
        print(f"Finished state {len(current_instances)}")
    
    robo_act.prune_memory(objects_list, contains_dict, constrained_dict)
    dump_state(current_instances=robo_memory.memory_instances,
               contains_dict=contains_dict,
               constrained_dict=constrained_dict,
               robo_memory=robo_memory,
               save_path=os.path.join(log_root, "final_state.pkl"))
    print("saved final state")

if __name__ == '__main__':
    run()
