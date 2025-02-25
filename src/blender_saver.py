import blenderproc as bproc
import bpy

# import random
# from utils_ema.camera_cv import Camera_cv
# from pathlib import Path
# from typing import List
# from objects.object import Object
# from logging import Logger


# os.makedirs(save_dir, exist_ok=True)
# --- Step 1: Initialize BlenderProc ---
bproc.init()

# --- Step 2: Create a new scene ---
bproc.utility.reset_keyframes()

# plane = bproc.object.create_primitive("PLANE", scale=[5, 5, 0.1])

# # --- Step 3: Create multiple planes ---
# num_planes = 1  # Number of planes to create
#
# for i in range(num_planes):
#     plane = bproc.object.create_primitive("PLANE", scale=[random.uniform(1, 3)] * 2)
#     plane.set_location(
#         [random.uniform(-5, 5), random.uniform(-5, 5), 0]
#     )  # Random XY positions
#     plane.set_name(f"Plane_{i}")
#
# # --- Step 4: Add lighting to the scene ---
# light = bproc.types.Light()
# light.set_type("SUN")
# light.set_location([5, -5, 10])
# light.set_energy(5)
#
# # --- Step 5: Set up the camera ---
# cam = bproc.camera.create_intrinsics(
#     fx=800, fy=800, cx=400, cy=400, height=800, width=800
# )
# cam_pose = bproc.math.build_transformation_mat([0, 0, 10], [0, 0, 0])
# bproc.camera.add_camera_pose(cam_pose)

# --- Step 6: Save the scene as a .blend file ---
# filepath = Path(save_dir) / "scene.blend"

bpy.ops.wm.save_as_mainfile(
    filepath="/home/manu/Desktop/repositories/refactored/opticalib/src/scene.blend"
)


# def blender_save(
#     save_dir: str, objects: List[Object], cameras: List[Camera_cv], logger: Logger
# ):
#
#     # # Clear existing objects
#     # bpy.ops.wm.read_factory_settings(use_empty=True)
#     #
#     # # Link object to the scene
#     # scene = bpy.context.scene
#     #
#     # for obj in objects:
#     #     o = obj.put_blender_obj_in_scene(scene)
#     #     # o.name =
#     #     break
#     #
#     # os.makedirs(save_dir, exist_ok=True)
#     # filepath = Path(save_dir) / "scene.blend"
#     # bpy.ops.wm.save_as_mainfile(filepath=str(filepath))
#     #
#     # exit(1)
#     #
#     # for cam in cameras:
#     #     print(cam)
#     # pass
#
#
