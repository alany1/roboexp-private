from roboexp.utils import get_pose_from_front_up_end_effector
import numpy as np
import open3d as o3d
from roboexp.utils import visualize_pc
from PIL import Image
from transforms3d.quaternions import mat2quat
from transforms3d.axangles import axangle2mat
from roboexp.utils import get_pose_look_at
import os
import copy
import pickle

COUNT_TIME = False
if COUNT_TIME:
    import time


class RoboAct:
    def __init__(
        self,
        robo_exp,
        robo_percept,
        robo_memory,
        object_level_labels,
        base_dir,
        gripper_length=0.158,
        base_iter=1000,
        wrist_gripper_path=None,
        no_mount_camera=True,
        # parameter for the open action
        open_num=20,
        prismatic_open_unit=0.02,
        revolute_open_unit=5 / 180 * np.pi,
        REPLAY_FLAG=False,
    ):
        # If replaying, no need to do the action, but need to process all the intermediate results
        self.REPLAY_FLAG = REPLAY_FLAG
        self.robo_exp = robo_exp
        self.robo_percept = robo_percept
        self.robo_memory = robo_memory
        self.object_level_labels = object_level_labels
        self.base_dir = base_dir
        # Create the dir for the observations
        dir_name = f"{self.base_dir}/observations"
        if not os.path.exists(dir_name):
            # Create directory if it doesn't exist
            os.makedirs(dir_name)
        # Predefine the labels to explore
        self.gripper_length = gripper_length
        self.base_iter = base_iter
        # The number of iterations to open the handle
        self.open_num = open_num
        # The unit of the movement to open the handle
        self.prismatic_open_unit = prismatic_open_unit
        self.revolute_open_unit = revolute_open_unit
        self.extra_alignment = False
        # Load the gripper mask in the wrist camera
        if wrist_gripper_path is not None:
            self.wrist_gripper_mask = np.array(Image.open(wrist_gripper_path))
            self.wrist_gripper_mask = self.wrist_gripper_mask > 0
        else:
            self.wrist_gripper_mask = None
        self.no_mount_camera = no_mount_camera
        if no_mount_camera:
            camera_positions = {
                "camera_0": np.array([0.2, 0, 0.6]),
                "camera_1": np.array([0.2, 0.4, 0.4]),
                "camera_2": np.array([0.2, -0.4, 0.4]),
            }
            target = np.array([0.8, 0.0, 0.3])
            self.camera_poses = [
                get_pose_look_at(eye=np.array(camera_position), target=target)
                for camera_position in camera_positions.values()
            ]
        # The options to save the observations
        self.save_index = 0
        # Set the predefined idle spaces
        self.idle_spaces = [
            [0.3, 0.6],
            [0.4, 0.6],
            [0.47, 0.6],
            [0.52, 0.6],
            [0.56, 0.6],
        ]
        self.idle_flag = [True, True, True, True, True]
    def get_image_bbox(self, *, w2c, points, K, dist_coef, w, h):
        # Project the points to the image plane
        pc_camera = points @ w2c[:3, :3].T + w2c[:3, 3]
        mask = pc_camera[:, 2] > 0
        px = self.robo_memory._project_point_to_pixel(
            pc_camera[mask], K, dist_coef,
        )
        u, v = px[:, 1], px[:, 0]

        # Get the bounding box of the projected points
        idx = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v = u[idx], v[idx]

        u = u.astype(np.int32)
        v = v.astype(np.int32)

        xmin, xmax = u.min(), u.max()
        ymin, ymax = v.min(), v.max()

        return xmin, ymin, xmax, ymax

    def alan_get_observations_update_memory(
        self,
        fake_obs,
        articulate_object=None,
        event=None,
    ):
        obs_attr = self.robo_percept.get_attributes_from_observations(
            fake_obs, visualize=True
        )
        if articulate_object is None:
            self.robo_memory.update_memory(
                fake_obs,
                obs_attr,
                self.object_level_labels,
                filter_masks=dict(fake=None),
                visualize=True,
            )
        else:
            from utils import rotate_pcd, translate_pcd
            fridge = articulate_object["pointcloud"]
            joint_type = articulate_object["joint_type"]
            
            assert joint_type in ["revolute", "prismatic"]
            
            if joint_type == "revolute": 
                rotated_fridge = rotate_pcd(
                    pcd=fridge,
                    hinge_axis=articulate_object["articulation_params"]["rotation_dir"],
                    hinge_pivot=articulate_object["articulation_params"]["rotation_point"],
                    rad=event["art_params"]["angle"],
                )
            else:
                rotated_fridge = translate_pcd(
                    pcd=fridge,
                    direction=articulate_object["articulation_params"]["translation_dir"],
                    amount=event["art_params"]["translation"],
                )

            contains_obs_attr = dict()
            constrained_obs_attr = dict()
            for view in fake_obs:
                c2w = fake_obs[view]["c2w"]
                w2c = np.linalg.inv(c2w)
                
                K = fake_obs[view]["intrinsic"]
                dist_coef = fake_obs[view]["dist_coef"]
    
                before_bbox = self.get_image_bbox(
                    w2c=w2c,
                    points=fridge,
                    K=K,
                    dist_coef=dist_coef,
                    w=fake_obs[view]["rgb"].shape[1],
                    h=fake_obs[view]["rgb"].shape[0],
                )
                after_bbox = self.get_image_bbox(
                    w2c=w2c,
                    points=rotated_fridge,
                    K=K,
                    dist_coef=dist_coef,
                    w=fake_obs[view]["rgb"].shape[1],
                    h=fake_obs[view]["rgb"].shape[0],
                )
                
                # mask = np.zeros_like(fake_obs[view]["rgb"])
                # mask[after_bbox[1] : after_bbox[3], after_bbox[0] : after_bbox[2]] = 1
                # from matplotlib import pyplot as plt
                # plt.imshow(mask[..., 0].astype(bool)); plt.show()
                # plt.imshow(fake_obs[view]["rgb"])
                # plt.show()                
                # valid bboxes are the ones that fall in the bbox
                contains_list = []
                constrained_list = []
                for i, bbox in enumerate(obs_attr[view]["pred_boxes"]):
                    before_valid = (
                        bbox[0] >= before_bbox[0]
                        and bbox[1] >= before_bbox[1]
                        and bbox[2] <= before_bbox[2]
                        and bbox[3] <= before_bbox[3]
                    )
                    after_valid = (
                        bbox[0] >= after_bbox[0]
                        and bbox[1] >= after_bbox[1]
                        and bbox[2] <= after_bbox[2]
                        and bbox[3] <= after_bbox[3]
                    )
    
                    # visualize after mask
                    mask = np.zeros_like(fake_obs[view]["rgb"])
                    mask[after_bbox[1] : after_bbox[3], after_bbox[0] : after_bbox[2]] = 1
    
                    if before_valid:
                        contains_list.append(i)
                    if after_valid:
                        constrained_list.append(i)
    
                contains_pred_phrases = [
                    p
                    for i, p in enumerate(obs_attr[view]["pred_phrases"])
                    if i in contains_list
                ]
                contains_obs_attr[view] = {
                    "pred_boxes": obs_attr[view]["pred_boxes"][contains_list],
                    "pred_masks": obs_attr[view]["pred_masks"][contains_list],
                    "pred_phrases": contains_pred_phrases,
                    "mask_feats": obs_attr[view]["mask_feats"][contains_list],
                }
    
                constrained_pred_phrases = [
                    obs_attr[view]["pred_phrases"][i] for i in constrained_list
                ]
                constrained_obs_attr[view] = {
                    "pred_boxes": obs_attr[view]["pred_boxes"][constrained_list],
                    "pred_masks": obs_attr[view]["pred_masks"][constrained_list],
                    "pred_phrases": constrained_pred_phrases,
                    "mask_feats": obs_attr[view]["mask_feats"][constrained_list],
                }
            
            # for v in contains_obs_attr:
            #     print(contains_obs_attr[v]['pred_phrases'])
            
            self.robo_memory.update_memory(
                fake_obs,
                contains_obs_attr,
                self.object_level_labels,
                filter_masks=dict(fake=None),
                visualize=True,
            )
            
            contains_instances = copy.deepcopy(self.robo_memory.memory_instances)
            
            self.robo_memory.update_memory(
                fake_obs,
                constrained_obs_attr,
                self.object_level_labels,
                filter_masks=dict(fake=None),
                visualize=True,
            )
            
            constrained_instances = copy.deepcopy(self.robo_memory.memory_instances)
            
            return contains_instances, constrained_instances
            
    def get_observations_update_memory(
        self,
        wrist_only=False,
        direct_move=None,
        update_scene_graph=False,
        scene_graph_option=None,
        visualize=False,
    ):
        # wrist_only specify if we need to get the observations from one wrist camera or need to observe the whole environment
        # When we need to observe the whole environment, no_mount_camera specify if we have access to mounted camera to get multiple observations or need to move the wrist to get
        if COUNT_TIME:
            start = time.time()
        if not self.REPLAY_FLAG:
            if wrist_only:
                observations = self.robo_exp.get_observations(wrist_only=True)
            else:
                if not self.no_mount_camera:
                    observations = self.robo_exp.get_observations(wrist_only=False)
                else:
                    # Move the wrist camera to get multiple observations
                    # The rotation is designed to move the wrist camera to the mounted camera pose
                    rotation_mat = np.zeros((3, 3))
                    rotation_mat[0, 2] = 1
                    rotation_mat[1, 1] = -1
                    rotation_mat[2, 0] = 1
                    # Collect the observations
                    observations = {}
                    for i, camera_pose in enumerate(self.camera_poses):
                        new_q = mat2quat(camera_pose[:3, :3] @ rotation_mat)
                        self.robo_exp.run_action(
                            action_code=1,
                            iteration=self.base_iter,
                            action_parameters=np.concatenate(
                                [
                                    camera_pose[:3, 3],
                                    new_q,
                                ]
                            ),
                            for_camera=True,
                        )
                        observation = self.robo_exp.get_observations(wrist_only=True)[
                            "wrist"
                        ]
                        observations[f"wrist_{i}"] = observation
                    self.robo_exp.run_action(
                        action_code=4, iteration=self.base_iter / 2
                    )
            # Save the observations
            with open(
                f"{self.base_dir}/observations/observations_{self.save_index}.pkl",
                "wb",
            ) as f:
                pickle.dump(observations, f)
            with open(
                f"{self.base_dir}/observations/observations_options_{self.save_index}.pkl",
                "wb",
            ) as f:
                options = {}
                options["direct_move"] = direct_move
                options["update_scene_graph"] = update_scene_graph
                pickle.dump(options, f)
            self.save_index += 1
        else:
            # Read the observations and options from the pickle file
            with open(
                f"{self.base_dir}/observations/observations_{self.save_index}.pkl", "rb"
            ) as f:
                observations = pickle.load(f)
            with open(
                f"{self.base_dir}/observations/observations_options_{self.save_index}.pkl",
                "rb",
            ) as f:
                options = pickle.load(f)
            direct_move = options["direct_move"]
            update_scene_graph = options["update_scene_graph"]
            if direct_move is not None:
                scene_graph_option["move_vec"] = direct_move["move_vec"]
            self.save_index += 1
    
        ##############3
        with open("/home/exx/Downloads/tmp.pkl", "rb") as f:
            fake_obs = pickle.load(f)

        # fake_obs = {"fake_512": fake_obs["fake_512"]}
        obs_attr = self.robo_percept.get_attributes_from_observations(fake_obs, visualize=True)

        self.robo_memory.update_memory(
            fake_obs,
            obs_attr,
            self.object_level_labels,
            filter_masks=dict(fake=None),
            visualize=True,
        )
        # 
        old_instances = self.robo_memory.memory_instances.copy()
        
        with open("/home/exx/Downloads/tmp_2.pkl", "rb") as f:
            fake_obs_2 = pickle.load(f)
            
        obs_attr_2 = self.robo_percept.get_attributes_from_observations(fake_obs_2, visualize=True)
        
        # edit
        with open("/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/debug_vol_fusion/full/identified_objects.pkl", "rb") as f:
            objects_list = pickle.load(f)
        with open("/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/debug_vol_fusion/full/events.pkl", "rb") as f:
            events = pickle.load(f)
            
        from utils import rotate_pcd
        fridge = objects_list["object_0"]["pointcloud"]
        rotated_fridge = rotate_pcd(pcd=fridge, hinge_axis=objects_list["object_0"]['articulation_params']["rotation_dir"], hinge_pivot=objects_list["object_0"]['articulation_params']["rotation_point"], rad=events[0]['art_params']["angle"])
        
        def get_image_bbox(*, w2c, points, K, w, h):
            # Project the points to the image plane
            pc_camera = points @ w2c[:3, :3].T + w2c[:3, 3]
            mask = pc_camera[:, 2] > 0
            px = self.robo_memory._project_point_to_pixel(pc_camera[mask], K, fake_obs_2[view]['dist_coef'])
            u, v = px[:, 1], px[:, 0]

            # Get the bounding box of the projected points
            idx = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u, v = u[idx], v[idx]
            
            u = u.astype(np.int32)
            v = v.astype(np.int32)

            xmin, xmax = u.min(), u.max()
            ymin, ymax = v.min(), v.max()

            return xmin, ymin, xmax, ymax
        
        contains_obs_attr_2 = dict()
        constrained_obs_attr_2 = dict()
        for view in fake_obs_2:
            c2w = fake_obs_2[view]["c2w"]
            w2c = np.linalg.inv(c2w)
            
            K = fake_obs_2[view]["intrinsic"]
            
            before_bbox = get_image_bbox(w2c=w2c, points=fridge, K=K, w=fake_obs_2[view]["rgb"].shape[1], h=fake_obs_2[view]["rgb"].shape[0])
            after_bbox = get_image_bbox(w2c=w2c, points=rotated_fridge, K=K, w=fake_obs_2[view]["rgb"].shape[1], h=fake_obs_2[view]["rgb"].shape[0])

            # valid bboxes are the ones that fall in the bbox
            contains_list = []
            constrained_list = []
            for i, bbox in enumerate(obs_attr_2[view]["pred_boxes"]):
                before_valid = bbox[0] >= before_bbox[0] and bbox[1] >= before_bbox[1] and bbox[2] <= before_bbox[2] and bbox[3] <= before_bbox[3]
                after_valid = bbox[0] >= after_bbox[0] and bbox[1] >= after_bbox[1] and bbox[2] <= after_bbox[2] and bbox[3] <= after_bbox[3]
                
                # visualize after mask
                mask = np.zeros_like(fake_obs_2[view]["rgb"])
                mask[after_bbox[1]:after_bbox[3], after_bbox[0]:after_bbox[2]] = 1

                if before_valid:
                    contains_list.append(i)
                if after_valid:
                    constrained_list.append(i)
            
            
            contains_pred_phrases = [p for i, p in enumerate(obs_attr_2[view]["pred_phrases"]) if i in contains_list]
            contains_obs_attr_2[view] = {
                "pred_boxes": obs_attr_2[view]["pred_boxes"][contains_list],
                "pred_masks": obs_attr_2[view]["pred_masks"][contains_list],
                "pred_phrases": contains_pred_phrases,
                "mask_feats": obs_attr_2[view]["mask_feats"][contains_list],
            }
            
            constrained_pred_phrases = [obs_attr_2[view]["pred_phrases"][i] for i in constrained_list]
            constrained_obs_attr_2[view] = {
                "pred_boxes": obs_attr_2[view]["pred_boxes"][constrained_list],
                "pred_masks": obs_attr_2[view]["pred_masks"][constrained_list],
                "pred_phrases": constrained_pred_phrases,
                "mask_feats": obs_attr_2[view]["mask_feats"][constrained_list],
            }
            
            
            
            
            
        
        self.robo_memory.update_memory(
            fake_obs_2,
            contains_obs_attr_2,
            self.object_level_labels,
            filter_masks=dict(fake=None),
            visualize=True,
        )

        

        old_instance_ids = [instance.instance_id for instance in old_instances]
        new_instances = []
        for instance in self.robo_memory.memory_instances:
            if instance.instance_id not in old_instance_ids:
                new_instances.append(instance)
        
        i = 2        
        new_instances[i].index_to_pcd(new_instances[i].voxel_indexes)
        new_instances[i].label
        len(self.robo_memory.memory_instances), len(old_instances)
        
        
        
        ##################
        
        observation_attributes = self.robo_percept.get_attributes_from_observations(
            observations, visualize=visualize
        )
        
        
        
        if COUNT_TIME:
            print(f"VLM processing takes {time.time() - start}")
            start = time.time()
        # Update the memory based on the observations
        filter_masks = {"wrist": self.wrist_gripper_mask}

        self.robo_memory.update_memory(
            observations,
            observation_attributes,
            self.object_level_labels,
            direct_move=direct_move,
            filter_masks=filter_masks,
            extra_alignment=self.extra_alignment,
            update_scene_graph=update_scene_graph,
            scene_graph_option=scene_graph_option,
            visualize=visualize,
        )
        if not self.REPLAY_FLAG:
            # Save the memory
            dir_name = f"{self.base_dir}/memory"
            if not os.path.exists(dir_name):
                # Create directory if it doesn't exist
                os.makedirs(dir_name)
            self.robo_memory.save_memory(f"{dir_name}/memory_{self.save_index-1}.pkl")
        if COUNT_TIME:
            print(f"Memory updates takes {time.time() - start}")

    def skill_open_close(self, node, visualize=False):
        assert "handle" in node.node_label
        print("TEST: OPEN start")
        # Remove the instance from current memory
        self.robo_memory.remove_instance(node.instance)

        # Keep the memory before opening
        memory_before_open = copy.deepcopy(self.robo_memory.memory_scene)
        if not self.REPLAY_FLAG:
            # Open the stuff
            (
                handle_center,
                close_point,
                close_direciton,
                up_direction,
                joint_type,
                move_vec,
                revolute_info,
            ) = self.open_handle(node)
            print(f"The joint type for this handle is {joint_type}")
        else:
            # Check the handle two directions
            handle_center = node.handle_center

        # Update the memory
        self.get_observations_update_memory(visualize=visualize)

        if not self.REPLAY_FLAG:
            # Take extra observation using the wrist camera movement to see the full interior of the drawer
            wrist_position, wrist_rotation = self.move_wrist(
                handle_center,
                close_point,
                close_direciton,
                up_direction,
                joint_type,
                node,
            )

        self.get_observations_update_memory(wrist_only=True, visualize=visualize)

        if not self.REPLAY_FLAG:
            if joint_type == "revolute":
                # Upward a little a bit
                self.robo_exp.run_action(
                    action_code=1,
                    iteration=self.base_iter / 4,
                    action_parameters=np.concatenate(
                        [
                            wrist_position + np.array([0, 0, 1]) * 0.2,
                            wrist_rotation,
                        ]
                    ),
                    for_camera=True,
                    speed=100,
                )

        # Get the new voxel indexes based on the difference of the voxels between after opening and before opening
        new_voxel_indexes = self.robo_memory.diff_memory(
            memory_before_open, handle_center, care_radius=0.2
        )

        if not self.REPLAY_FLAG:
            self.robo_exp.run_action(action_code=4, iteration=self.base_iter / 2)

            # Close the stuff
            self.close_handle(
                close_point,
                close_direciton,
                up_direction,
                handle_center,
                joint_type,
                revolute_info,
            )

        if not self.REPLAY_FLAG:
            # Update the memory
            if joint_type == "prismatic":
                # For the drawer case, need to further move the memory
                direct_move = {
                    "move_vec": move_vec,
                    "drawer_indexes": new_voxel_indexes,
                }
                scene_graph_option = {
                    "type": "handle",
                    "handle_instance": node.instance,
                    "move_vec": move_vec,
                }
                self.get_observations_update_memory(
                    direct_move=direct_move,
                    update_scene_graph=True,
                    scene_graph_option=scene_graph_option,
                    visualize=visualize,
                )
            else:
                # For the door case, no need further move the memory
                scene_graph_option = {
                    "type": "handle",
                    "handle_instance": node.instance,
                }
                self.get_observations_update_memory(
                    update_scene_graph=True,
                    scene_graph_option=scene_graph_option,
                    visualize=visualize,
                )
        else:
            # Other options will be updated in the called function (it will read the log files to read all the options)
            self.get_observations_update_memory(
                scene_graph_option={
                    "type": "handle",
                    "handle_instance": node.instance,
                },
                visualize=visualize,
            )

    def open_handle(self, node):
        handle_center = node.handle_center
        handle_direction = node.handle_direction
        open_direction = node.open_direction
        joint_type = node.joint_type
        joint_info = {}
        if joint_type == "revolute":
            joint_axis = node.joint_axis
            joint_origin = node.joint_origin
            joint_info["joint_axis"] = joint_axis
            joint_info["joint_origin"] = joint_origin

        (
            close_point,
            close_direction,
            up_direction,
            move_vec,
            revolute_info,
        ) = self._execute_open_handle(
            handle_center, handle_direction, open_direction, joint_type, joint_info
        )

        # Check if robot gripper is in the reset position, if not, move to the reset position
        self.robo_exp.run_action(action_code=4, iteration=self.base_iter / 2)

        return (
            handle_center,
            close_point,
            close_direction,
            up_direction,
            joint_type,
            move_vec,
            revolute_info,
        )

    def _execute_open_handle(
        self, handle_center, handle_direction, open_direction, joint_type, joint_info
    ):
        up_direction = np.cross(handle_direction, open_direction)
        front_direction = -open_direction
        end_effector_position = handle_center + self.gripper_length * open_direction
        end_effector_rotation = get_pose_from_front_up_end_effector(
            front=front_direction, up=up_direction
        )

        self.robo_exp.run_action(
            action_code=1,
            iteration=self.base_iter,
            action_parameters=np.concatenate(
                [
                    end_effector_position + open_direction * 0.03,
                    end_effector_rotation,
                ]
            ),
        )

        self.robo_exp.run_action(
            action_code=1,
            iteration=self.base_iter,
            action_parameters=np.concatenate(
                [end_effector_position, end_effector_rotation]
            ),
            speed=80,
        )
        # Close the gripper
        self.robo_exp.run_action(action_code=3, iteration=self.base_iter / 2)

        move_vec = None
        revolute_info = None
        if joint_type == "prismatic":
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter / 4,
                action_parameters=np.concatenate(
                    [
                        end_effector_position
                        + self.open_num * open_direction * self.prismatic_open_unit,
                        end_effector_rotation,
                    ]
                ),
                speed=80,
            )
            move_vec = -self.open_num * open_direction * self.prismatic_open_unit

            # Open the gripper
            self.robo_exp.run_action(action_code=2, iteration=self.base_iter / 2)

            # Backward and upward a bit to avoid collision
            close_point = (
                end_effector_position
                + self.open_num * open_direction * self.prismatic_open_unit
            )
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        close_point + open_direction * 0.05,
                        end_effector_rotation,
                    ]
                ),
                speed=80,
            )

            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        close_point + open_direction * 0.03 + np.array([0, 0, 1]) * 0.2,
                        end_effector_rotation,
                    ]
                ),
                speed=100,
            )
        elif joint_type == "revolute":
            joint_axis = joint_info["joint_axis"]
            joint_origin = joint_info["joint_origin"]
            trajectory_positions = []
            trjectory_rotation = []
            gripper_position = None
            new_front_direction = None  # Record the last front direction
            revolute_info = {
                "trajectory_positions": [end_effector_position - open_direction * 0.02],
                "trjectory_rotation": [end_effector_rotation],
            }
            for i in range(1, self.open_num + 1):
                rot_mat = axangle2mat(
                    joint_axis, i * self.revolute_open_unit, is_normalized=True
                )
                gripper_position = (
                    np.dot(rot_mat, (handle_center - joint_origin)) + joint_origin
                )
                new_front_direction = np.dot(rot_mat, front_direction)
                end_effector_rotation = get_pose_from_front_up_end_effector(
                    front=new_front_direction, up=up_direction
                )
                end_effector_position = (
                    gripper_position - new_front_direction * self.gripper_length
                )
                trajectory_positions.append(end_effector_position)
                trjectory_rotation.append(end_effector_rotation)

            revolute_info["trajectory_positions"] += trajectory_positions
            revolute_info["trjectory_rotation"] += trjectory_rotation
            # Visualize the trajectory
            if False:
                scene_pcd, scene_color = self.robo_memory.get_scene_pcd()
                # Test the estimated normal of current scen first
                scene_pc = o3d.geometry.PointCloud()
                scene_pc.points = o3d.utility.Vector3dVector(scene_pcd)
                scene_pc.colors = o3d.utility.Vector3dVector(scene_color)
                extra = []
                for i in range(self.open_num):
                    # draw sphere on the trajectory_positions
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                    sphere.translate(trajectory_positions[i])
                    sphere.paint_uniform_color(np.array([1, 0, 0]))
                    extra.append(sphere)
                visualize_pc(scene_pcd, scene_color, extra=extra)

            for i in range(self.open_num):
                self.robo_exp.run_action(
                    action_code=1,
                    iteration=self.base_iter / 4,
                    action_parameters=np.concatenate(
                        [trajectory_positions[i], trjectory_rotation[i]]
                    ),
                    speed=80,
                )

            # Open the gripper
            self.robo_exp.run_action(action_code=2, iteration=self.base_iter / 2)

            # Backward and upward a bit to avoid collision
            close_point = gripper_position + self.gripper_length * open_direction
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        trajectory_positions[-1] - new_front_direction * 0.1,
                        trjectory_rotation[-1],
                    ]
                ),
                speed=80,
            )
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        trajectory_positions[-1]
                        - new_front_direction * 0.1
                        + np.array([0, 0, 1]) * 0.2,
                        trjectory_rotation[-1],
                    ]
                ),
                speed=100,
            )
            revolute_info["trajectory_positions"].append(
                trajectory_positions[-1] - new_front_direction * 0.1
            )
            revolute_info["trjectory_rotation"].append(trjectory_rotation[-1])
            revolute_info["trajectory_positions"].append(
                trajectory_positions[-1]
                - new_front_direction * 0.1
                + np.array([0, 0, 1]) * 0.2
            )
            revolute_info["trjectory_rotation"].append(trjectory_rotation[-1])
        else:
            raise ValueError("Unknown joint type")

        return close_point, -open_direction, up_direction, move_vec, revolute_info

    def close_handle(
        self,
        close_point,
        close_direction,
        up_direction,
        handle_center,
        joint_type,
        revolute_info,
    ):
        end_effector_position = close_point - close_direction * 0.05
        end_effector_rotation = get_pose_from_front_up_end_effector(
            front=close_direction, up=up_direction
        )

        # Close the gripper
        if joint_type == "prismatic":
            self.robo_exp.run_action(
                action_code=2, action_parameters=[True], iteration=self.base_iter / 2
            )
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [end_effector_position, end_effector_rotation]
                ),
                speed=80,
            )

            # Only change the close direction
            handle_end_effector_position = (
                handle_center
                + 0.01 * close_direction
                - self.gripper_length * close_direction
            )
            # target_position is the projection of handle_end_effector_position on the close_direction
            target_position = (
                np.dot(handle_end_effector_position - close_point, close_direction)
                / np.linalg.norm(close_direction)
                * close_direction
            ) + close_point
            # Go to the closing point
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter / 4,
                action_parameters=np.concatenate(
                    [
                        target_position,
                        end_effector_rotation,
                    ]
                ),
                speed=80,
            )
            # Fully open the gripper
            self.robo_exp.run_action(action_code=2, iteration=self.base_iter / 2)
        elif joint_type == "revolute":
            self.robo_exp.run_action(action_code=2, iteration=self.base_iter / 2)
            for i in range(len(revolute_info["trajectory_positions"]) - 1, -1, -1):
                self.robo_exp.run_action(
                    action_code=1,
                    iteration=self.base_iter / 4,
                    action_parameters=np.concatenate(
                        [
                            revolute_info["trajectory_positions"][i],
                            revolute_info["trjectory_rotation"][i],
                        ]
                    ),
                    speed=80,
                )
            target_position = revolute_info["trajectory_positions"][0]
            end_effector_rotation = revolute_info["trjectory_rotation"][0]

        # Backward and upward a little bit to avoid collision
        self.robo_exp.run_action(
            action_code=1,
            iteration=self.base_iter / 4,
            action_parameters=np.concatenate(
                [
                    target_position - close_direction * 0.1,
                    end_effector_rotation,
                ]
            ),
            speed=80,
        )
        self.robo_exp.run_action(
            action_code=1,
            iteration=self.base_iter / 4,
            action_parameters=np.concatenate(
                [
                    target_position - close_direction * 0.1 + np.array([0, 0, 1]) * 0.2,
                    end_effector_rotation,
                ]
            ),
            speed=100,
        )
        # Fully open the gripper
        self.robo_exp.run_action(action_code=2, iteration=self.base_iter / 2)

        # Check if robot gripper is in the reset position, if not, move to the reset position
        self.robo_exp.run_action(action_code=4, iteration=self.base_iter / 2)

    def move_wrist(
        self,
        handle_center,
        close_point,
        close_direciton,
        up_direction,
        joint_type,
        node,
    ):
        camera_up_direction = np.array([0.0, 0.0, 1.0])
        if joint_type == "prismatic":
            wrist_position = close_point + camera_up_direction * 0.4
        else:
            joint_origin = node.joint_origin
            floor_normal_direction = np.array([0.0, 0.0, 1.0])
            viewpoint_position = (
                -np.dot(joint_origin - handle_center, up_direction) * up_direction
                + handle_center
                + 0.03 * floor_normal_direction
            )
            wrist_position = (
                viewpoint_position
                - 0.4 * close_direciton
                + 0.05 * floor_normal_direction
            )

        camera_front_direction = handle_center - wrist_position
        camera_front_direction /= np.linalg.norm(camera_front_direction)
        camera_right_direction = np.cross(camera_front_direction, camera_up_direction)

        wrist_rotation = get_pose_from_front_up_end_effector(
            front=camera_front_direction, up=-camera_right_direction
        )
        self.robo_exp.run_action(
            action_code=1,
            iteration=self.base_iter / 4,
            action_parameters=np.concatenate(
                [
                    wrist_position,
                    wrist_rotation,
                ]
            ),
            for_camera=True,
            speed=80,
        )

        return wrist_position, wrist_rotation

    def skill_pick(self, node, action_type, visualize=False):
        # Pick the object into idle positions
        print(f"TEST: {action_type} start")

        if action_type == "pick_away":
            # Remove the instance from current memory
            self.robo_memory.remove_instance(node.instance)
            # Pick the objects to the idle positions
            object_pcd = node.instance.index_to_pcd(node.instance.voxel_indexes)
            front_direction = np.array([0, 0, -1])
            up_direction = np.array([0, 1, 0])
            # Pick up the object from top center (the most general case)
            pick_point = object_pcd[object_pcd[:, 2].argmax()]
            object_height = object_pcd[:, 2].max() - object_pcd[:, 2].min()

            end_effector_position = (
                pick_point
                + self.gripper_length * (-front_direction)
                + max(0.05 - object_height, 0) * (-front_direction)
            )
            # Get the gripper pose
            end_effector_rotation = get_pose_from_front_up_end_effector(
                front=front_direction, up=up_direction
            )

            # Get the idle position
            idle_index, idle_xy = self._get_idle_position()
            height = end_effector_position[2]
            idle_position = np.array([idle_xy[0], idle_xy[1], height])

            node.pick_back_position = idle_position
            node.pick_back_rotation = end_effector_rotation
            node.original_position = end_effector_position
            node.front_direction = front_direction
            node.up_direction = up_direction
            node.idle_index = idle_index

            if self.REPLAY_FLAG:
                return

            # Open the gripper
            self.robo_exp.run_action(action_code=2, iteration=self.base_iter / 2)
            # Move the gripper to the pick point
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        end_effector_position + 0.03 * (-front_direction),
                        end_effector_rotation,
                    ]
                ),
            )

            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        end_effector_position + 0.05 * front_direction,
                        end_effector_rotation,
                    ]
                ),
                speed=80,
            )

            # Close the gripper
            self.robo_exp.run_action(action_code=3, iteration=self.base_iter / 2)

            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        end_effector_position + 0.03 * (-front_direction),
                        end_effector_rotation,
                    ]
                ),
                speed=80,
            )

            # Move to the idle position
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        idle_position + 0.03 * (-front_direction),
                        end_effector_rotation,
                    ]
                ),
            )

            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        idle_position + 0.05 * front_direction,
                        end_effector_rotation,
                    ]
                ),
                speed=80,
            )

            # Open the gripper
            self.robo_exp.run_action(action_code=2, iteration=self.base_iter / 2)

            # Move a bit higher
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        idle_position + 0.03 * (-front_direction),
                        end_effector_rotation,
                    ]
                ),
                speed=80,
            )
            # Return to the initial position
            self.robo_exp.run_action(action_code=4, iteration=self.base_iter / 2)
        elif action_type == "pick_back":
            action_node = list(node.actions.values())[0]
            for child in action_node.children.values():
                child.instance.no_merge = True
            if self.REPLAY_FLAG:
                return
            front_direction = node.front_direction
            up_direction = node.up_direction
            pick_up_position = node.pick_back_position
            pick_up_rotation = node.pick_back_rotation
            # Move a bit higher
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        pick_up_position + 0.03 * (-front_direction),
                        pick_up_rotation,
                    ]
                ),
            )
            # Open the gripper
            self.robo_exp.run_action(action_code=2, iteration=self.base_iter / 2)
            # Move to the pick up position
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        pick_up_position + 0.05 * front_direction,
                        pick_up_rotation,
                    ]
                ),
                speed=80,
            )
            # Close the gripper
            self.robo_exp.run_action(action_code=3, iteration=self.base_iter / 2)
            # Move a bit higher
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        pick_up_position + 0.03 * (-front_direction),
                        pick_up_rotation,
                    ]
                ),
            )
            # Move to the original position
            original_position = node.original_position
            original_rotation = pick_up_rotation
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        original_position + 0.03 * (-front_direction),
                        original_rotation,
                    ]
                ),
            )
            # Move to the position
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        original_position + 0.05 * front_direction,
                        original_rotation,
                    ]
                ),
                speed=80,
            )
            # Open the gripper
            self.robo_exp.run_action(action_code=2, iteration=self.base_iter / 2)
            # Move a bit higher
            self.robo_exp.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        original_position + 0.03 * (-front_direction),
                        original_rotation,
                    ]
                ),
                speed=80,
            )
            # Return to the initial position
            self.robo_exp.run_action(action_code=4, iteration=self.base_iter / 2)
            idle_index = node.idle_index
            self.idle_flag[idle_index] = True
        else:
            raise NotImplementedError

    def _get_idle_position(self):
        for i, flag in enumerate(self.idle_flag):
            if flag:
                self.idle_flag[i] = False
                return i, self.idle_spaces[i]
        raise ValueError("No idle space available")

    def save_memory(self, save_path):
        self.robo_memory.save_memory(save_path)
