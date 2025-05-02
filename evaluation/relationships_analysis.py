import numpy as np
import json
import pickle
from spark_sg.build_dynamic_objects import build
from evaluation.eval_static_objects import bbox_iou


if __name__ == '__main__':
    keyframes_path = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/keyframes.pkl"
    gt_dynamic_objects_json = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/object_labels/dynamic_objects.json"

    gt_dynamic_bboxes_json = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/object_labels/dynamic_bboxes.json"
    gt_bone_bboxes_json = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/object_labels/bone_bboxes.json"

    final_state = "/home/exx/Downloads/spark_states_v9/final_state.pkl"
    events = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/debug_vol_fusion/full/events.pkl"
    identified_objects = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/debug_vol_fusion/full/identified_objects.pkl"
    
    with open(final_state, "rb") as f:
        final_state = pickle.load(f)
    with open(events, "rb") as f:
        events = pickle.load(f)
    with open(identified_objects, "rb") as f:
        identified_objects = pickle.load(f)

    for k in final_state["contains_dict"]:
        print(k, [x["instance_id"] for x in final_state["contains_dict"][k]])

    for k in final_state["constrained_dict"]:
        print(k, [x["instance_id"] for x in final_state["constrained_dict"][k]])
