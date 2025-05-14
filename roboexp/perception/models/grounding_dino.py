import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import torch
from PIL import Image
import numpy as np
def undo_clockwise90_square(boxes_rot):
    """
    Invert a 90°-clockwise rotation for a *square* image.

    Args
    ----
    boxes_rot : (N, 4) tensor
        Normalised [cx, cy, w, h] in the *rotated* frame.

    Returns
    -------
    boxes_orig : (N, 4) tensor
        Normalised [cx, cy, w, h] in the **original** frame.
    """
    cx_r, cy_r, w_r, h_r = boxes_rot.unbind(-1)  # (N,)

    cx_o = cy_r  # xₒ =  yᵣ               (swap axes)
    cy_o = 1.0 - cx_r  # yₒ = 1 – xᵣ            (horizontal flip)
    w_o = h_r  # width  ↔ height
    h_o = w_r

    return torch.stack([cx_o, cy_o, w_o, h_o], dim=-1)


# The code is modified based on https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam_demo.py
class MyGroundingDINO:
    def __init__(
        self,
        config_path,
        checkpoint_path,
        device="cuda",
    ):
        self.device = device
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.custom_box_thresholds = {
            "handle": 0.1,
        }
            
        # Load the model
        args = SLConfig.fromfile(config_path)
        args.device = device
        self.model = build_model(args)
        # Load the pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.eval()
        self.model.to(self.device)

    # def get_grounding_output(self, img, text, with_logits=True, only_max=True):
    #     # The example of the text can be "chair. table ."
    #     _img = self._process_image(img)
    #     _text = self._process_text(text)
    #     _img = _img.to(self.device)
    #     with torch.no_grad():
    #         outputs = self.model(_img[None], captions=[_text])
    #     logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    #     boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    # 
    #     # filter output
    #     logits_filt = logits.clone()
    #     boxes_filt = boxes.clone()
    #     filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
    #     logits_filt = logits_filt[filt_mask]  # num_filt, 256
    #     boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    #     logits_filt.shape[0]
    #     # get phrase
    #     tokenlizer = self.model.tokenizer
    #     tokenized = tokenlizer(_text)
    #     # build pred
    #     pred_phrases = []
    #     for logit, box in zip(logits_filt, boxes_filt):
    #         # Support multiple confidences for different labels (modify the code based on the original function get_phrases_from_posmap())
    #         if only_max:
    #             # Pick the label that the sublabel can have the highest probability
    #             label_index = torch.argmax(logit)
    #             # Find the starting point of this label and the ending point of this label
    #             start_label = int(label_index)
    #             while True:
    #                 if (
    #                     tokenized["input_ids"][start_label - 1] == 101
    #                     or tokenized["input_ids"][start_label - 1] == 102
    #                     or tokenized["input_ids"][start_label - 1] == 1012
    #                 ):
    #                     break
    #                 start_label -= 1
    #             end_label = int(label_index)
    #             while True:
    #                 if (
    #                     tokenized["input_ids"][end_label + 1] == 102
    #                     or tokenized["input_ids"][end_label + 1] == 101
    #                     or tokenized["input_ids"][end_label + 1] == 1012
    #                 ):
    #                     break
    #                 end_label += 1
    #             pred_label = tokenlizer.decode([tokenized["input_ids"][i] for i in range(start_label, end_label + 1)])
    #             pred_logit = logit[label_index].item()
    #             pred_phrases.append(f"{pred_label}:{str(pred_logit)[:4]}")
    #         else:
    #             posmap = logit > self.text_threshold
    #             non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
    #             token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
    #             pred_labels = tokenlizer.decode(token_ids).split()
    #             pred_logits = [logit[i].item() for i in non_zero_idx]
    #             if with_logits:
    #                 output = [
    #                     f"{pred_labels[i]}:{str(pred_logits[i])[:4]}"
    #                     for i in range(len(pred_labels))
    #                 ]
    #                 pred_phrases.append(" ".join(output))
    #             else:
    #                 pred_phrases.append(" ".join(pred_labels))
    # 
    #     return boxes_filt, pred_phrases
    
    def get_grounding_output(self, img, text, with_logits=True, only_max=True):
        """
        Same interface as the original but with word-specific thresholds:
          • a box survives if ANY token inside it clears its personal cut-off
          • the arg-max label is kept only if *its* score clears that same cut-off
        """

        # ------------------------------------------------------------------ forward
        assert only_max, "only_max must be True for this function"
        # rotate 90 degrees clockwise
        img = np.rot90(img, k=3).copy()
        _img = self._process_image(img).to(self.device)
        _text = self._process_text(text)
        with torch.no_grad():
            out = self.model(_img[None], captions=[_text])

        logits = out["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = out["pred_boxes"].cpu()[0]  # (nq, 4)

        boxes = undo_clockwise90_square(boxes)  # back to original frame
        # ------------------------------------------------------- build thr_vec once
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(_text)
        ids_full = tokenized["input_ids"]  # e.g. [101, …, 102]
        thr_vec = torch.full(
            (len(ids_full),),
            fill_value=self.box_threshold,
            dtype=logits.dtype,
        )

        for word, thr in self.custom_box_thresholds.items():
            ids_word = tokenizer(word)["input_ids"][1:-1]  # drop [CLS]/[SEP]
            for i in range(len(ids_full) - len(ids_word) + 1):
                if ids_full[i : i + len(ids_word)] == ids_word:
                    thr_vec[i : i + len(ids_word)] = thr  # lower the bar

        # ---------------------------------------- keep boxes whose ANY token passes
        filt_mask = (logits[:, : thr_vec.size(0)] > thr_vec).any(dim=1)
        logits_filt = logits[filt_mask]  # (#keep , 256)
        boxes_filt = boxes[filt_mask]  # (#keep , 4)

        # ------------------------------------------------------------- decode boxes
        kept_boxes = []
        pred_phrases = []

        for logit, box in zip(logits_filt, boxes_filt):
            # -------- arg-max token & span ---------------------------------
            label_idx = torch.argmax(logit).item()  # 0‥255
            score = logit[label_idx].item()

            # span left ← until boundary token
            start = label_idx
            while ids_full[start - 1] not in (101, 102, 1012):  # [CLS],[SEP],'.'
                start -= 1
            # span right → until boundary token
            end = label_idx
            while ids_full[end + 1] not in (101, 102, 1012):
                end += 1

            label = tokenizer.decode(ids_full[start : end + 1]).strip().lower()
            thr = self.custom_box_thresholds.get(label, self.box_threshold)

            if score >= thr:  # keep this box & label 👍
                kept_boxes += [box]
                pred_phrases.append(f"{label}:{score:.3f}")
            # else: drop the box completely

        # ----------------------------- pack results (torch tensors, like original)
        if kept_boxes:
            boxes_out = torch.stack(kept_boxes)
        else:  # no surviving detections
            boxes_out = torch.empty((0, 4))

        return boxes_out, pred_phrases

    def _process_image(self, img):
        # Apply the normalization to the image
        img_pil = Image.fromarray(img)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        _img, _ = transform(img_pil, None)
        return _img

    def _process_text(self, text):
        # Do processing to the text
        text = text.lower()
        text = text.strip()
        if not text.endswith("."):
            text = text + "."
        return text
