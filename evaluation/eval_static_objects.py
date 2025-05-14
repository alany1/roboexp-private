
import numpy as np
import json
import glob
import pickle
import yaml
# from scene_graph_eval.eval_utils import bbox_iou
from collections import defaultdict
import numpy as np

def bbox_iou(b1, b2):
    """3-D IoU between two AABBs  (b = [[min],[max]] in meters)."""
    overlap_min = np.maximum(b1[0], b2[0])
    overlap_max = np.minimum(b1[1], b2[1])
    inter_dim   = np.clip(overlap_max - overlap_min, a_min=0, a_max=None)
    inter_vol   = inter_dim.prod()
    vol1        = (b1[1] - b1[0]).prod()
    vol2        = (b2[1] - b2[0]).prod()
    return inter_vol / (vol1 + vol2 - inter_vol + 1e-12)

class StaticObjectEvaluator:
    """
    Compare a *predicted* DSG layer-2 (static objects) against GT JSON
    {name:{bbox:[min,max], class_id:int, pose:[xyz, quat]}}.
    """

    def __init__(self, obj_nodes, gt_dict, label_id2name):
        self.gt = gt_dict

        self.label_id2name = label_id2name
        
        self.gt.keys()
        # instance and semantic id have the same meaning -- objects with the same instance id should be grouped as ONE GT isntance.
        # we populate the bounding box attribute for each of these.
        # key is the instance id
        instance_bbox_mapping = dict()
        for name, obj in self.gt.items():
            # bbox is defined as the maximum bounding box of all the objects with the same instance id
            bbox = obj["bbox"]
            object_id = obj["class_id"]
            if object_id not in instance_bbox_mapping:
                instance_bbox_mapping[object_id] = {
                    "bbox": bbox,
                    "class_id": obj["class_id"],
                }
            else:
                # maximum
                instance_bbox_mapping[object_id]["bbox"] = [
                    np.minimum(instance_bbox_mapping[object_id]["bbox"][0], bbox[0]),
                    np.maximum(instance_bbox_mapping[object_id]["bbox"][1], bbox[1]),
                ]
        
        self.gt = instance_bbox_mapping
            
        
        
        
        self.pred_nodes = []
        for node in obj_nodes:
            if node["instance_id"] not in [x["instance_id"] for x in self.pred_nodes] and node["attributes"]["label"] not in ["background", "wall", "ceiling", "floor", "countertop", "door", "doorframe", "window", "windowframe", "windowsill", "cabinet"]:
                self.pred_nodes.append(node)

    def evaluate(self):
        tp, fp, fn = 0, 0, 0

        matched_gt = set()
        for n in self.pred_nodes:
            # if n["attributes"]["label"] in ["background", "wall", "ceiling", "floor", "countertop", "door", "doorframe", "window", "windowframe", "windowsill", "cabinet"]:
            #     continue

            pred_size = n["attributes"]["size"]
            pred_size = np.maximum(pred_size, 0.01)

            pred_center = n["attributes"]["center"]
            pred_bbox = pred_center - pred_size / 2, pred_center + pred_size / 2

            # find best-IoU GT object of same semantic label
            cands = [name for name in self.gt if name not in matched_gt and n["label"] == self.gt[name]["class_id"]]
            if not cands:
                # print(n)
                fp += 1
                continue

            best = min(cands, key =lambda g: np.linalg.norm(np.mean(np.array(self.gt[g]["bbox"]), axis=0) - pred_center))

            name, gt_bbox, gt_pose = best, self.gt[best]["bbox"], None
            iou = bbox_iou(pred_bbox, np.array(gt_bbox))

            if iou <= 0.0:
                fp += 1
                continue

            # === counted as a true positive ===
            tp += 1
            matched_gt.add(name)

        fp = len(self.pred_nodes) - len(matched_gt)
        fn = len(self.gt) - len(matched_gt)

        metrics = {
            "precision": tp / (tp + fp + 1e-9),
            "recall": tp / (tp + fn + 1e-9),
            "F1": 2 * tp / (2 * tp + fp + fn + 1e-9),
        }

        return metrics

def eval_static_objects(state, label_id2name):
    all_static_objects = []
    for k, v in state["contains_dict"].items():
        all_static_objects.extend(v)

    all_static_objects.extend(state["current_instances"])

    evaluator = StaticObjectEvaluator(all_static_objects, gt_static_objects, label_id2name)
    metrics = evaluator.evaluate()

    return metrics

if __name__ == '__main__':
    # keyframes_path = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/keyframes.pkl"
    # 
    # gt_static_objects_json = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/object_labels/static_objects.json"
    # gt_dynamic_objects_json = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/object_labels/dynamic_objects.json"
    # 
    # gt_dynamic_bboxes_json = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/object_labels/dynamic_bboxes.json"
    # gt_bone_bboxes_json = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/object_labels/bone_bboxes.json"
    # 
    # final_state = "/home/exx/Downloads/spark_states_v8/final_state.pkl"
    # 
    # label_space_path = "/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/semantic_labeling/label_space.yaml"


    root = "/home/exx/datasets/aria/blender_eval/bedroom"
    keyframes_path = f"{root}/keyframes.pkl"

    gt_static_objects_json = f"{root}/object_labels/static_objects.json"

    final_state = f"{root}/debug_vol_fusion/full/sg_est/final_state.pkl"

    label_space_path = f"{root}/semantic_labeling/label_space.yaml"
    
    with open(label_space_path, "r") as f:
        label_space = yaml.safe_load(f)

    label_id2name = {x['label']: x['name'].replace("_", "") for x in label_space["label_names"]}

    with open(gt_static_objects_json, "r") as f:
        gt_static_objects = json.load(f)

    with open(keyframes_path, "rb") as f:
        keyframes = pickle.load(f)

    # flatten, ignore first
    keyframes = [item for sublist in keyframes for item in sublist][1:]

    static_objects_map = {}
    dynamic_objects_map = {}
    mesh_map = {}

    with open(final_state, "rb") as f:
        state = pickle.load(f)

    metrics = eval_static_objects(state, label_id2name)
    print(metrics)
