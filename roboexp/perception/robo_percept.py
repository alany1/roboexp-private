from .models import MyGroundingSegment, MyDenseClip
from roboexp.utils import display_image
import torch
import numpy as np
from spark_sg import USE_SEMANTICS

class RoboPercept:
    def __init__(self, grounding_dict, lazy_loading=False, color_to_instance=None, device="cuda"):
        # When lazy loading, the model will be loaded when it is used; warning: it will decrease the inference speed
        self.grounding_dict = grounding_dict
        self.lazy_loading = lazy_loading
        self.device = device
        
        if USE_SEMANTICS:
            from spark_sg import COLOR_INSTANCE_MAPPING
            self.color_to_instance = COLOR_INSTANCE_MAPPING

        if not self.lazy_loading:
            # Used to get the grounding masks
            self.my_grounding_sam = MyGroundingSegment(device=self.device)
            # Used to get the pixel-wise clip features
            self.my_dense_clip = MyDenseClip(device=self.device)
    
    def get_mask_bbox(self, mask):
        # Get the bounding box of the mask
        y, x = np.where(mask)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        return x_min, y_min, x_max, y_max
                 
    def process_semantics(self, semantics):
        # assuming input is an instance map
        colors = np.unique(semantics.reshape(-1, 3), axis=0)
        pred_phrases = []
        pred_masks = []
        pred_boxes = []
        for c in colors:
            k = self.color_to_instance[tuple(c.tolist())]
            if k == 0:
                # skip background
                continue
            confidence=1.0
            phrase = f"{k}:{confidence}"
            pred_mask = np.all(semantics == c, axis=-1)
            
            pred_masks.append(pred_mask)
            pred_boxes.append(self.get_mask_bbox(pred_mask))
            pred_phrases.append(phrase)

        pred_masks = np.array(pred_masks)
        pred_boxes = np.array(pred_boxes)
        return pred_boxes, pred_phrases, pred_masks
        
    def get_attributes_from_observations(self, observations, visualize=False):
        observation_attributes = {}
        for name, obs in observations.items():
            # obs includes "rgb", "position", "mask", "c2w"
            rgb = obs["rgb"]
            img = (rgb * 255).clip(0, 255).astype("uint8")
            
            if USE_SEMANTICS:
                semantics = obs["semantics"]
                pred_boxes, pred_phrases, pred_masks = self.process_semantics(semantics)

                mask_feats = np.zeros((pred_masks.shape[0], 1024), dtype=np.float16)
            else:
                pred_boxes, pred_phrases, pred_masks = self.get_grounding_masks(
                    img, only_max=True
                )
                
                if pred_boxes is None:
                    observation_attributes[name] = {
                        "pred_boxes": None,
                        "pred_phrases": None,
                        "pred_masks": None,
                        "mask_feats": None,
                    }
                    continue
                    
                mask_feats = self.get_dense_clip(
                    img, pred_boxes, pred_masks, pred_phrases, per_mask=True
                )
                
            # Visualize the segmentation
            if visualize:
                display_image(img, boxes=pred_boxes, labels=pred_phrases, masks=pred_masks)


            observation_attributes[name] = {
                "pred_boxes": pred_boxes,
                "pred_phrases": pred_phrases,
                "pred_masks": pred_masks,
                "mask_feats": mask_feats,
            }
        return observation_attributes

    def get_grounding_masks(self, img, only_max=True):
        # only_max: only return the label with the highest confidence score for each mask
        if not self.lazy_loading:
            my_grounding_sam = self.my_grounding_sam
        else:
            my_grounding_sam = MyGroundingSegment(device=self.device)
        # The example of the text can be "chair. table ."
        pred_boxes, pred_phrases, pred_masks = my_grounding_sam.run(
            img, self.grounding_dict, only_max=only_max
        )
        # Clean the GPU memory
        if self.lazy_loading:
            del my_grounding_sam
            torch.cuda.empty_cache()
        return pred_boxes, pred_phrases, pred_masks

    def get_dense_clip(self, img, boxes, masks, phrases, per_mask=True):
        if not self.lazy_loading:
            my_dense_clip = self.my_dense_clip
        else:
            my_dense_clip = MyDenseClip(device=self.device)
        # The phrases include the confidence score, the example format is "table:0.7834 chair:0.2333" for each mask
        results = my_dense_clip.run(img, boxes, masks, phrases, per_mask=per_mask)
        # Clean the GPU memory
        if self.lazy_loading:
            del my_dense_clip
            torch.cuda.empty_cache()
        if not per_mask:
            dense_feats, dense_confidences = results 
            return dense_feats.numpy(), dense_confidences.numpy()
        else:
            mask_feats = results
            return mask_feats.numpy()
