import csv
from typing import Dict, Tuple

def load_colors_by_rgb(
    path: str,
    *,
    with_alpha: bool = False,
) -> Dict[Tuple[int, ...], str]:
    key_fields = ("red", "green", "blue", "alpha") if with_alpha else (
        "red", "green", "blue"
    )

    palette = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            # build the key tuple in the channel order given in the file
            key = tuple(int(row[k]) for k in key_fields)
            id = int(row["id"])
            palette[key] = id

    return palette

USE_SEMANTICS = True

# _colors2name = load_colors_by_rgb("/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/renders/semantics_vis/colors.csv")
# COLOR_INSTANCE_MAPPING = {}
# i = 1
# for color in _colors2name:
#     COLOR_INSTANCE_MAPPING[color] = i
#     i += 1
# 

COLOR_INSTANCE_MAPPING = load_colors_by_rgb("/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/renders/semantics_vis/colors.csv")    
        
