import pickle
import numpy as np

from utils import rotate_pcd, translate_pcd

def build_from_keyframes(object_events, canonical_pcd, parent_object):
    canonical_bbox = (np.min(canonical_pcd, axis=0), np.max(canonical_pcd, axis=0))
    canonical_centroid = np.mean(canonical_bbox, axis=0)
    
    centroid_keyframes = dict()
    art_params = parent_object["articulation_params"]
    art_type = parent_object["joint_type"]
    current_object_state = 0
    for event in sorted(object_events, key=lambda x: x['start_ts']):
        if art_type == "revolute":
            centroid_keyframes[event['start_ts']] = rotate_pcd(pcd=canonical_centroid[None, ...],
                                                               hinge_axis=art_params["rotation_dir"],
                                                               hinge_pivot=art_params["rotation_point"],
                                                               rad=current_object_state,)[0]
            current_object_state = event["art_params"]["angle"] 
            centroid_keyframes[event['end_ts']] = rotate_pcd(pcd=canonical_centroid[None, ...],
                                                             hinge_axis=art_params["rotation_dir"],
                                                             hinge_pivot=art_params["rotation_point"],
                                                             rad=current_object_state,)[0]
            
        else:
            centroid_keyframes[event['start_ts']] = translate_pcd(pcd=canonical_centroid[None, ...],
                                                                   direction=art_params["translation_dir"],
                                                                   amount=current_object_state)[0]
            current_object_state = event["art_params"]["translation"]
            centroid_keyframes[event['end_ts']] = translate_pcd(pcd=canonical_centroid[None, ...],
                                                                 direction=art_params["translation_dir"],
                                                                 amount=current_object_state)[0]
    return dict(
        canonical_pcd=canonical_pcd,
        canonical_bbox=canonical_bbox,
        trajectory_keyframes=centroid_keyframes,
    )

def build(*, final_state, events, objects_list):
    """
    returns a dict mapping from instance id to 
    {canonical_pcd: canonical_pcd, canonical_bbox: canonical_bbox, trajectory_keyframes: {keyframe: centroid_pos}}
    
    First: do constrained objects
    Then: do identified objects.
    """
    object_event_dict = {}
    for event in events:
        if event["object_name"] not in object_event_dict:
            object_event_dict[event["object_name"]] = []
        object_event_dict[event["object_name"]].append(event)
    
    constrained_dict = final_state["constrained_dict"]
    dynamic_objects = {}
    
    for k, v in constrained_dict.items():
        articulate_part = k
        
        # build identified object
        dynamic_objects[articulate_part] = build_from_keyframes(object_event_dict[articulate_part],
                                                                objects_list[articulate_part]["pointcloud"],
                                                                objects_list[articulate_part])
        for i, obj in enumerate(v):
            child_obj_pcd = obj["canonical_pcd"]
            dynamic_objects[obj["instance_id"]] = build_from_keyframes(object_event_dict[articulate_part],
                                                                       child_obj_pcd,
                                                                       objects_list[articulate_part])
            

    return dynamic_objects

if __name__ == '__main__':

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

    with open("/home/exx/Downloads/spark_states_v9/final_state.pkl", "rb") as f:
        final_state = pickle.load(f)

    build(final_state=final_state, events=events, objects_list=objects_list)
