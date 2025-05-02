import numpy as np
import json
import pickle
from spark_sg.build_dynamic_objects import build
from evaluation.eval_static_objects import bbox_iou

IOU_THRESH = 0.0                 # “detected” if 3-D IoU ≥ 0.0

def corners_to_bbox(corners: np.ndarray):
    """
    corners : (8, 3) ndarray – each row is one XYZ corner of the box
    returns : (bbox_min, bbox_max) where each is shape (3,)
    """
    if corners.shape != (8, 3):
        raise ValueError("Expected an (8, 3) array of corners.")

    bbox_min = corners.min(axis=0)  # (x_min, y_min, z_min)
    bbox_max = corners.max(axis=0)  # (x_max, y_max, z_max)
    return bbox_min, bbox_max

class DynamicObjectEvaluator:
    """
    Compare a *predicted* DSG layer-2 (static objects) against GT JSON
    {name:{bbox:[min,max], class_id:int, pose:[xyz, quat]}}.
    """

    def __init__(self, obj_nodes, gt_dict):
        self.pred_nodes = obj_nodes
        self.gt = gt_dict

        # get bounding box of all objects at the given keyframe
        self.max_matching_distance = 0.5

    def interpolate_centroid(self, trajectory_keyframes, kf):
        if kf in trajectory_keyframes.keys():
            return trajectory_keyframes[kf]
        
        if kf < min(trajectory_keyframes.keys()):
            return trajectory_keyframes[min(trajectory_keyframes.keys())]
        if kf > max(trajectory_keyframes.keys()):
            return trajectory_keyframes[max(trajectory_keyframes.keys())]
        
        closest_below = max(
            [x for x in trajectory_keyframes.keys() if x < kf]
        )
        closest_above = min(
            [x for x in trajectory_keyframes.keys() if x > kf]
        )
        
        # linear interpolation
        t = (kf - closest_below) / (closest_above - closest_below)
        pred_centroid = (
            trajectory_keyframes[closest_below] * (1 - t)
            + trajectory_keyframes[closest_above] * t
        )
        return pred_centroid
        
    
    def evaluate(self, keyframe):
        tp, fp, fn = 0, 0, 0

        matched_gt = dict()
        for n in self.pred_nodes:
            if n == "12_0_instance":
                print('wtf')
            pred_centroid = self.interpolate_centroid(
                self.pred_nodes[n]["trajectory_keyframes"], keyframe
            )

            # no semantics constraint for dynamic objects
            cands = [name for name in self.gt[keyframe] if name not in matched_gt]

            if not cands:
                fp += 1
                continue
            
            pred_bbox_bounds = np.array(self.pred_nodes[n]["canonical_bbox"])
            # center it
            pred_bbox_bounds_centered = pred_bbox_bounds - np.mean(pred_bbox_bounds, axis=0)[None, ...]
            
            # translate bbox without rotation
            pred_bbox_bounds_centered += pred_centroid[None, ...] 

            # best = max(
            #     cands,
            #     key=lambda g: bbox_iou(
            #         pred_bbox_bounds_centered, np.array(self.gt[keyframe][g])
            #     ),
            #     default=None,
            # )

            best = min(
                cands,
                key=lambda g: np.linalg.norm(pred_centroid - np.mean(self.gt[keyframe][g] , axis=0)),
                default=None,
            )

            gt_centroid = np.mean(np.array(self.gt[keyframe][best]), axis=0)

            (
                name,
                gt_bbox,
            ) = best, self.gt[keyframe][best]
            iou = bbox_iou(pred_bbox_bounds_centered, np.array(gt_bbox))
            
            # bbox_iou(pred_bbox_bounds_centered, np.array(self.gt[keyframe]["Fork"]))
            # if iou <= IOU_THRESH:
            #     print('missed', iou, name, n)
            #     fp += 1
            #     continue

            if np.linalg.norm(pred_centroid - gt_centroid) > 0.1:
                fp += 1
                continue

            # === counted as a true positive ===
            tp += 1
            matched_gt[name] = n

        fn = len(self.gt[keyframe]) - len(matched_gt)
        print(set(self.gt[keyframe].keys()) - set(matched_gt.keys()))
        print(matched_gt)

        metrics = {
            "precision": tp / (tp + fp + 1e-9),
            "recall": tp / (tp + fn + 1e-9),
            "F1": 2 * tp / (2 * tp + fp + fn + 1e-9),
        }
        return metrics

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

    with open(gt_dynamic_objects_json, "r") as f:
        gt_dynamic_objects = json.load(f)
    with open(gt_dynamic_bboxes_json, "r") as f:
        gt_dynamic_bboxes = json.load(f)

    with open(gt_bone_bboxes_json, "r") as f:
        gt_bone_bboxes = json.load(f)

    with open(keyframes_path, "rb") as f:
        keyframes = pickle.load(f)

    # flatten, ignore first
    keyframes = [item for sublist in keyframes for item in sublist][1:]
    temp = []
    
    pred_dynamic_objects = build(final_state=final_state, 
          events=events,
          objects_list=identified_objects)

    all_metrics = dict()

    all_dynamic_bbox = {}
 
    from copy import deepcopy

    for k in keyframes:
        _d = deepcopy(gt_dynamic_bboxes[str(k)])
        _d.update(deepcopy(gt_bone_bboxes[str(k)]))
        all_dynamic_bbox[k] = deepcopy(_d)

    evaluator = DynamicObjectEvaluator(pred_dynamic_objects, all_dynamic_bbox)
    for k in keyframes:
        metrics = evaluator.evaluate(k)

        all_metrics[k] = metrics

    average_metric = {
        "precision": np.mean([x["precision"] for x in all_metrics.values()]),
        "recall": np.mean([x["recall"] for x in all_metrics.values()]),
        "F1": np.mean([x["F1"] for x in all_metrics.values()]),
    }

    print(f"Average metrics: {average_metric}")
