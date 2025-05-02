import pickle
from spark_sg.utils.sg_viz import hierarchy_to_mesh
from utils import rotate_pcd, translate_pcd
import numpy as np
import open3d as o3d

from killport import kill_ports
kill_ports(ports=[8012])

def get_hull(points):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    hull, _ = pcd.compute_convex_hull()
    hull.remove_duplicated_vertices()

    return np.asarray(hull.vertices), np.asarray(hull.triangles)

def extract_viz_data(state):
    all_instances = []
    contains_dict = dict()
    constrained_dict = dict()
    parentless_instances = []
    for k, v in state["contains_dict"].items():
        all_instances.extend(v)
        contains_dict[k] = [x["instance_id"] for x in v]
        
    for k, v in state["constrained_dict"].items():
        all_instances.extend(v)
        constrained_dict[k] = [x["instance_id"] for x in v]
    
    for instance in state["current_instances"]:
        all_instances.append(instance)
        parentless_instances.append(instance["instance_id"])
    
    all_instance_ids = [x["instance_id"] for x in all_instances]
    instance_id_to_bbox = {x["instance_id"]: x["attributes"] for x in all_instances} #  if np.all(x["attributes"]["size"] > 0)}
    
    data = dict(
        all_instance_ids=all_instance_ids,
        contains=contains_dict,
        constrained=constrained_dict,
        parentless_instances=parentless_instances,
        instance_id_to_bbox=instance_id_to_bbox,
        points=state["points"],
        colors=state["colors"],
    )
    
    return data

def get_hierarchy(state,         
                  gap_z=2.0,
        parent_gap_z=2.0,
        node_radius=0.05,
        parent_radius=0.07,
        line_radius=0.012,
        edge_radius=0.008,
        show_boxes=False,):    
    viz_data = extract_viz_data(state)
    hierarchy = hierarchy_to_mesh(viz_data, gap_z=gap_z, parent_gap_z=parent_gap_z,
                                  node_radius=node_radius, parent_radius=parent_radius,
                                  line_radius=line_radius, edge_radius=edge_radius,
                                  show_boxes=show_boxes)
    return np.asarray(hierarchy.vertices), np.asarray(hierarchy.triangles),(np.asarray(hierarchy.vertex_colors)*255).astype(np.uint8)



def get_pcd(state):
    pcd_points = np.asarray(state["points"])
    pcd_colors = (np.asarray(state["colors"])*255).astype(np.uint8)
    return pcd_points, pcd_colors


def load_articulate_parts(identified_objects):
    articulated_parts = dict()
    
    for obj_k, obj_v in identified_objects.items():
        articulated_parts[obj_k] = get_hull(obj_v["pointcloud"])
        
    return articulated_parts

def articulate(key, amount, articulated_parts, identified_objects):
    if key not in articulated_parts:
        return None
    
    vertices, faces = articulated_parts[key]
    articulation_type = identified_objects[key]["joint_type"]
    if articulation_type == "revolute":
        # Rotate the vertices around the hinge axis
        new_vertices = rotate_pcd(pcd=vertices, hinge_axis=identified_objects[key]["articulation_params"]["rotation_dir"], hinge_pivot=identified_objects[key]["articulation_params"]["rotation_point"], rad=amount)
    else:
        new_vertices = translate_pcd(pcd=vertices, direction=identified_objects[key]["articulation_params"]["translation_dir"], amount=amount)
    
    # Create a new TriMesh object with the updated vertices
    return group(TriMesh(vertices=new_vertices, faces=faces, color="red"), key=key)

def load_vuer_parts(articulated_parts):
    vuer_parts = []
    for k, v in articulated_parts.items():
        vuer_parts.append(
            TriMesh(vertices=v[0], faces=v[1], key=k)
        )
    return vuer_parts
        


def articulation_wrapper(i, fps, disp_per_s, *,
                         start=None,        # value at t = 0  (default = lower)
                         lower=0.0, upper=1.0):
    i = np.asarray(i, dtype=float)

    # Handle degenerate cases early
    delta = upper - lower
    if delta == 0 or disp_per_s <= 0:
        return np.full_like(i, lower, dtype=float)

    # Choose starting point
    if start is None:
        start = lower
    # Clamp to range just in case
    start = np.clip(start, lower, upper)

    # --- Compute the “time offset” needed so that pos(0) = start -----------
    # Triangle-wave value in [0,1] that corresponds to 'start'
    triangle0 = (start - lower) / delta          # in [0,1]

    # We’ll *assume* the motion starts by moving upward (lower→upper).  
    # That means phase in [0,1) should equal 'triangle0'.
    phase0 = triangle0                           # phase at t = 0

    travel_time = abs(delta) / disp_per_s        # seconds to go one leg
    offset_time = phase0 * travel_time           # seconds to shift waveform
    # -----------------------------------------------------------------------

    # Elapsed (shifted) time for every frame
    t = i / fps + offset_time                    # seconds
    phase = (t / travel_time) % 2.0              # ∈ [0,2)

    # Triangle wave in [0,1]
    triangle = 1.0 - np.abs(1.0 - phase)

    # Map back to [lower, upper]
    pos = lower + delta * triangle
    return pos if pos.ndim else float(pos)

from asyncio import sleep

from vuer import Vuer, VuerSession
from vuer.schemas import Scene, PointCloud, TriMesh, group


app = Vuer(queries=dict(show_grid=False))


@app.spawn(start=True)
async def main(sess: VuerSession):
    with open("/home/exx/Downloads/spark_states_v8/final_state.pkl", "rb") as f:
        state = pickle.load(f)
    with open("/home/exx/datasets/aria/blender_eval/kitchen_cgtrader_4449901/debug_vol_fusion/full/identified_objects.pkl", "rb") as f:
        identified_objects = pickle.load(f)
    
    # print out constrained and contained ids
    for obj in state["constrained_dict"]:
        obj_constrained = [x["instance_id"] for x in state["constrained_dict"][obj]] 
        print(obj_constrained)
        
    for obj in state["contains_dict"]:
        obj_constrained = [x["instance_id"] for x in state["contains_dict"][obj]] 
        print(obj_constrained)
    
    articulate_parts = load_articulate_parts(identified_objects)
    parts = load_vuer_parts(articulate_parts)
    
    h_v, h_f, h_c = get_hierarchy(state, show_boxes=False)
    pcd_points, pcd_colors = get_pcd(state)
    
    sess.set @ Scene(
        TriMesh(vertices=h_v, faces=h_f, colors=h_c),
        PointCloud(vertices=pcd_points, colors=pcd_colors, size=0.025 ),
        # *parts,
        rotation=[-np.pi/2, 0, 0],
    )
    
    
    # sess.upsert @ articulate("object_6", 0.0, articulate_parts, identified_objects)
    # sess.upsert @ articulate("object_6", 0.5, articulate_parts, identified_objects)
    # sess.upsert @ articulate("object_6", 1.0, articulate_parts, identified_objects)
    # 
    i = 0
    fps = 24
    
    limits = {
        "object_0": {"lower": 0.0, "upper": 2.0, "start": 0.0, "disp_per_s": 2/2},
        "object_1": {"lower": -2.0, "upper": 0.0, "start": 0.0, "disp_per_s": 2/2},
        "object_4": {"lower": -2.0, "upper": 0.0, "start": 0.0, "disp_per_s": 2/2},
        "object_6": {"lower": 0.0, "upper": 0.3, "start": 0.0, "disp_per_s": 0.3/2},
        
    }
    
    while True:
        # for k in identified_objects:
        #     sess.upsert @ articulate(k, articulation_wrapper(i, fps, **limits[k]), articulate_parts, identified_objects)
        #     
        # print(i)
        i += 1
        
        await sleep(1 / fps)

    print("Done")
