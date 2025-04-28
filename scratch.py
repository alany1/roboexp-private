from roboexp import (
    RobotExplorationReal,
    RoboMemory,
    RoboPercept,
    RoboActReal,
    RoboDecision,
)
from datetime import datetime
import os

# Initialize the modules
base_dir = f"/home/exx/mit/RoboEXP/experiments/demo_1"
REPLAY_FLAG = True
robo_exp = RobotExplorationReal(gripper_length=0.285, REPLAY_FLAG=REPLAY_FLAG)
# Initialize the memory module
robo_memory = RoboMemory(
    # lower_bound=[0, -0.8, -1],
    # higher_bound=[1, 0.5, 2],
    lower_bound=[-10, -10, -10],
    higher_bound=[10, 10, 10],
    # lower_bound=[-3.47571843,  1.20802123,  1.47201896],
    # higher_bound=[0.95888057, 7.9796549 , 8.77895072],
    voxel_size=0.02,
    real_camera=True,
    base_dir=base_dir,
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
    # "refrigerator",
    # "cabinet",
    # "can",
    # "doll",
    # "plate",
    # "spoon",
    # "fork",
    # "hamburger",
    # "condiment",
]
part_level_labels = ["handle"]

grounding_dict = (
    " . ".join(object_level_labels) + " . " + " . ".join(part_level_labels)
)
# Initialize the perception module
robo_percept = RoboPercept(grounding_dict=grounding_dict, lazy_loading=False)
# Initialize the action module
robo_act = RoboActReal(
    robo_exp,
    robo_percept,
    robo_memory,
    object_level_labels,
    base_dir=base_dir,
    REPLAY_FLAG=REPLAY_FLAG,
)
# Initialize the decision module
robo_decision = RoboDecision(robo_memory, base_dir, REPLAY_FLAG=REPLAY_FLAG)
# Get the first observation, visualize the RGBD with semantic masks and the low-level memory
robo_act.get_observations_update_memory(update_scene_graph=True, visualize=True)

# Our system continue making the decisions to explore and take new observations in a closed-loop
# Visualize the RGBD observations, semantic masks, low-level memory and high-level graph
robo_decision.update_action_list()
while robo_decision.is_done() == False:
    action = robo_decision.get_action()
    print(action)
    if action[1] == "open_close":
        robo_act.skill_open_close(action[0], visualize=True)
    elif "pick" in action[1]:
        robo_act.skill_pick(action[0], action[1], visualize=False)
        if action[1] == "pick_away":
            update_scene_graph = True
            scene_graph_option = {
                "type": "pick_away",
                "node": action[0],
            }
            robo_act.get_observations_update_memory(
                update_scene_graph=update_scene_graph,
                scene_graph_option=scene_graph_option,
                visualize=False,
            )
            robo_decision.update_action_list()
        elif action[1] == "pick_back":
            update_scene_graph = True
            scene_graph_option = {
                "type": "pick_back",
                "node": action[0],
            }
            robo_act.get_observations_update_memory(
                update_scene_graph=update_scene_graph,
                scene_graph_option=scene_graph_option,
                visualize=False,
            )
